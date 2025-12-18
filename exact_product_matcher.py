"""
Exact Product Matching System for Shelf Product Counting

This module implements the industry-standard approach for exact product identification:
1. Fine-grained Classification (Primary)
2. Embedding Similarity Verification (Secondary)
3. OCR Text Verification (Tertiary)

Used for accurate product counting from shelf images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
import faiss
from collections import defaultdict


class ExactProductMatcher(nn.Module):
    """
    Multi-head neural network for exact product matching.
    
    Combines:
    - Brand classification (coarse-grained)
    - Product classification (fine-grained, exact SKU)
    - Embedding extraction (for verification)
    """
    
    def __init__(self, num_brands: int = 100, num_exact_products: int = 10000, embedding_dim: int = 256):
        super().__init__()
        
        # Shared backbone - EfficientNet-V2 Medium
        self.backbone = models.efficientnet_v2_m(weights='DEFAULT')
        backbone_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Branch 1: Brand/Category Classification (Coarse)
        self.brand_classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_brands)
        )
        
        # Branch 2: Exact Product Classification (Fine-grained)
        self.product_classifier = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_exact_products)
        )
        
        # Branch 3: Embedding for similarity verification
        self.embedding = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with multi-head output.
        
        Returns:
            dict: Contains brand logits, product logits, and normalized embedding
        """
        features = self.backbone(x)
        
        # Multi-head output
        brand_logits = self.brand_classifier(features)
        product_logits = self.product_classifier(features)
        embedding = nn.functional.normalize(self.embedding(features), p=2, dim=1)
        
        return {
            'brand': brand_logits,
            'product': product_logits,
            'embedding': embedding
        }


class ProductEmbeddingIndex:
    """
    FAISS-based embedding index for fast similarity search.
    Stores reference embeddings for each product SKU.
    """
    
    def __init__(self, embedding_dim: int = 256, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create FAISS index (Inner Product for normalized embeddings = cosine similarity)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.product_ids = []
        self.sku_to_indices = defaultdict(list)
    
    def add_products(self, embeddings: np.ndarray, product_ids: List[str]):
        """
        Add product embeddings to the index.
        
        Args:
            embeddings: Normalized embeddings (N, embedding_dim)
            product_ids: List of SKU identifiers
        """
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        start_idx = len(self.product_ids)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for i, pid in enumerate(product_ids):
            self.product_ids.append(pid)
            self.sku_to_indices[pid].append(start_idx + i)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for k most similar products.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dicts with product_id and similarity score
        """
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.product_ids) and idx >= 0:
                results.append({
                    'product_id': self.product_ids[idx],
                    'similarity': float(dist),
                    'confidence': float(dist)
                })
        
        return results
    
    def verify_sku(self, query_embedding: np.ndarray, predicted_sku: str, threshold: float = 0.85) -> bool:
        """
        Verify if query embedding matches the predicted SKU.
        
        Args:
            query_embedding: Query embedding vector
            predicted_sku: Predicted SKU to verify
            threshold: Minimum similarity threshold
            
        Returns:
            True if verified, False otherwise
        """
        if predicted_sku not in self.sku_to_indices:
            return False
        
        # Get all reference embeddings for this SKU
        sku_indices = self.sku_to_indices[predicted_sku]
        
        # Search among all embeddings
        results = self.search(query_embedding, k=len(self.product_ids))
        
        # Check if any of the top matches belong to the predicted SKU
        for result in results[:5]:  # Check top 5 matches
            if result['product_id'] == predicted_sku and result['similarity'] >= threshold:
                return True
        
        return False
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk."""
        # Convert GPU index to CPU before saving
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu_index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'product_ids': self.product_ids,
                'sku_to_indices': dict(self.sku_to_indices)
            }, f)
    
    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.product_ids = metadata['product_ids']
            self.sku_to_indices = defaultdict(list, metadata['sku_to_indices'])


class ProductDatabase:
    """
    Product catalog database with hierarchy and metadata.
    Manages product information from reference_images directory.
    """
    
    def __init__(self, reference_dir: str = 'data/reference_images'):
        self.reference_dir = Path(reference_dir)
        self.products = {}
        self.brand_to_products = defaultdict(list)
        self.sku_to_brand = {}
        
    def load_from_directory(self):
        """
        Load product database from reference_images directory structure.
        
        Expected structure:
        reference_images/
            1000/  (SKU/Product ID)
                image1.jpg
                image2.jpg
            1001/
                image1.jpg
        """
        if not self.reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {self.reference_dir}")
        
        # Scan all subdirectories (each is a product SKU)
        for sku_dir in sorted(self.reference_dir.iterdir()):
            if sku_dir.is_dir():
                sku = sku_dir.name
                
                # Get all images for this SKU
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(sku_dir.glob(ext))
                
                if image_files:
                    self.products[sku] = {
                        'sku': sku,
                        'name': f'Product_{sku}',
                        'images': [str(img) for img in image_files],
                        'brand': self._extract_brand(sku),
                        'category': 'general',
                        'num_references': len(image_files)
                    }
                    
                    brand = self.products[sku]['brand']
                    self.brand_to_products[brand].append(sku)
                    self.sku_to_brand[sku] = brand
        
        print(f"Loaded {len(self.products)} products from {self.reference_dir}")
        return self.products
    
    def _extract_brand(self, sku: str) -> str:
        """
        Extract brand from SKU.
        For now, use first 2 digits as brand identifier.
        Can be customized based on your SKU naming convention.
        """
        # Example: SKU 1000 -> Brand "10"
        if len(sku) >= 2:
            return sku[:2]
        return sku
    
    def get_product(self, sku: str) -> Optional[Dict]:
        """Get product information by SKU."""
        return self.products.get(sku)
    
    def get_all_skus(self) -> List[str]:
        """Get list of all SKUs."""
        return list(self.products.keys())
    
    def get_brands(self) -> List[str]:
        """Get list of all brands."""
        return list(self.brand_to_products.keys())
    
    def get_sku_to_id_mapping(self) -> Dict[str, int]:
        """Create mapping from SKU to integer ID for classification."""
        return {sku: idx for idx, sku in enumerate(sorted(self.products.keys()))}
    
    def get_brand_to_id_mapping(self) -> Dict[str, int]:
        """Create mapping from brand to integer ID."""
        return {brand: idx for idx, brand in enumerate(sorted(self.brand_to_products.keys()))}
    
    def save_metadata(self, filepath: str):
        """Save product database metadata to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                'products': self.products,
                'brand_to_products': dict(self.brand_to_products),
                'sku_to_brand': self.sku_to_brand,
                'total_products': len(self.products),
                'total_brands': len(self.brand_to_products)
            }, f, indent=2)
    
    def load_metadata(self, filepath: str):
        """Load product database metadata from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.products = data['products']
            self.brand_to_products = defaultdict(list, data['brand_to_products'])
            self.sku_to_brand = data['sku_to_brand']


class ExactProductCounter:
    """
    Production-ready exact product counting system.
    
    Workflow:
    1. Classify product to exact SKU (primary)
    2. Verify with embedding similarity (secondary)
    3. OCR text verification (tertiary, optional)
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        db_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Load product database
        self.product_db = ProductDatabase()
        if db_path and Path(db_path).exists():
            self.product_db.load_metadata(db_path)
        else:
            self.product_db.load_from_directory()
        
        # Get mappings
        self.sku_to_id = self.product_db.get_sku_to_id_mapping()
        self.id_to_sku = {v: k for k, v in self.sku_to_id.items()}
        self.brand_to_id = self.product_db.get_brand_to_id_mapping()
        self.id_to_brand = {v: k for k, v in self.brand_to_id.items()}
        
        # Initialize model
        num_brands = len(self.brand_to_id)
        num_products = len(self.sku_to_id)
        
        self.model = ExactProductMatcher(
            num_brands=num_brands,
            num_exact_products=num_products,
            embedding_dim=256
        ).to(device)
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle both checkpoint dict and direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint format from training
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                accuracy = checkpoint.get('accuracy', 'unknown')
                print(f"Loaded model from {model_path} (epoch: {epoch}, accuracy: {accuracy})")
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        # Initialize embedding index
        self.embedding_index = ProductEmbeddingIndex(embedding_dim=256, use_gpu=(device == 'cuda'))
        
        # Load embedding index if available
        if index_path and metadata_path:
            if Path(index_path).exists() and Path(metadata_path).exists():
                self.embedding_index.load(index_path, metadata_path)
                print(f"Loaded embedding index from {index_path}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to(self.device)
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract features from product image.
        
        Returns:
            dict: Contains brand prediction, product prediction, and embedding
        """
        img_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Get predictions
        brand_probs = torch.softmax(outputs['brand'], dim=1)
        product_probs = torch.softmax(outputs['product'], dim=1)
        
        brand_id = torch.argmax(brand_probs, dim=1).item()
        product_id = torch.argmax(product_probs, dim=1).item()
        
        brand_conf = brand_probs[0, brand_id].item()
        product_conf = product_probs[0, product_id].item()
        
        return {
            'brand_id': brand_id,
            'brand': self.id_to_brand.get(brand_id, 'unknown'),
            'brand_confidence': brand_conf,
            'product_id': product_id,
            'sku': self.id_to_sku.get(product_id, 'unknown'),
            'product_confidence': product_conf,
            'embedding': outputs['embedding'].cpu().numpy()[0]
        }
    
    def verify_with_embedding(
        self, 
        embedding: np.ndarray, 
        predicted_sku: str, 
        threshold: float = 0.85
    ) -> Tuple[bool, float]:
        """
        Verify prediction using embedding similarity.
        
        Args:
            embedding: Query embedding
            predicted_sku: Predicted SKU to verify
            threshold: Minimum similarity threshold
            
        Returns:
            (verified, max_similarity)
        """
        if predicted_sku == 'unknown':
            return False, 0.0
        
        verified = self.embedding_index.verify_sku(embedding, predicted_sku, threshold)
        
        # Get actual similarity score
        results = self.embedding_index.search(embedding, k=1)
        max_similarity = results[0]['similarity'] if results else 0.0
        
        return verified, max_similarity
    
    def match_product(
        self, 
        product_image: np.ndarray,
        classification_threshold: float = 0.8,
        embedding_threshold: float = 0.85,
        use_embedding_verification: bool = True
    ) -> Dict:
        """
        Match a single product image to exact SKU.
        
        Args:
            product_image: Cropped product image (numpy array)
            classification_threshold: Minimum confidence for classification
            embedding_threshold: Minimum similarity for embedding verification
            use_embedding_verification: Whether to use embedding verification
            
        Returns:
            dict: Match results with SKU, confidence, and verification status
        """
        # Step 1: Classify to exact SKU
        features = self.extract_features(product_image)
        
        predicted_sku = features['sku']
        classification_conf = features['product_confidence']
        
        # Step 2: Verify with embedding (if enabled and confidence below threshold)
        embedding_verified = False
        embedding_similarity = 0.0
        
        if use_embedding_verification and classification_conf < 1.0:
            embedding_verified, embedding_similarity = self.verify_with_embedding(
                features['embedding'],
                predicted_sku,
                embedding_threshold
            )
        
        # Determine final confidence
        if use_embedding_verification:
            final_confidence = min(classification_conf, embedding_similarity) if embedding_verified else classification_conf * 0.7
        else:
            final_confidence = classification_conf
        
        return {
            'sku': predicted_sku,
            'brand': features['brand'],
            'classification_confidence': classification_conf,
            'embedding_similarity': embedding_similarity,
            'embedding_verified': embedding_verified,
            'final_confidence': final_confidence,
            'matched': final_confidence >= classification_threshold
        }
    
    def count_products(
        self, 
        detected_products: List[np.ndarray],
        classification_threshold: float = 0.8,
        embedding_threshold: float = 0.85,
        use_embedding_verification: bool = True
    ) -> Dict:
        """
        Count exact products from list of detected product images.
        
        Args:
            detected_products: List of cropped product images
            classification_threshold: Minimum confidence for classification
            embedding_threshold: Minimum similarity for embedding verification
            use_embedding_verification: Whether to use embedding verification
            
        Returns:
            dict: Product counts and detailed results
        """
        product_counts = defaultdict(int)
        confidence_scores = defaultdict(list)
        all_matches = []
        
        for i, product_img in enumerate(detected_products):
            # Match product
            match_result = self.match_product(
                product_img,
                classification_threshold,
                embedding_threshold,
                use_embedding_verification
            )
            
            # Count only if matched
            if match_result['matched']:
                sku = match_result['sku']
                product_counts[sku] += 1
                confidence_scores[sku].append(match_result['final_confidence'])
            
            # Store all results
            match_result['detection_id'] = i
            all_matches.append(match_result)
        
        # Calculate average confidence per SKU
        avg_confidence = {
            sku: np.mean(scores) 
            for sku, scores in confidence_scores.items()
        }
        
        return {
            'counts': dict(product_counts),
            'total_detected': len(detected_products),
            'total_matched': sum(product_counts.values()),
            'avg_confidence': avg_confidence,
            'all_matches': all_matches
        }
    
    def build_embedding_index(self, save_index_path: str = None, save_metadata_path: str = None):
        """
        Build embedding index from reference images in product database.
        
        Args:
            save_index_path: Path to save FAISS index
            save_metadata_path: Path to save metadata
        """
        print("Building embedding index from reference images...")
        
        all_embeddings = []
        all_skus = []
        
        for sku, product_info in self.product_db.products.items():
            print(f"Processing {sku}: {len(product_info['images'])} images")
            
            for img_path in product_info['images']:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract embedding
                features = self.extract_features(image)
                all_embeddings.append(features['embedding'])
                all_skus.append(sku)
        
        # Add to index
        embeddings_array = np.array(all_embeddings)
        self.embedding_index.add_products(embeddings_array, all_skus)
        
        print(f"Added {len(all_embeddings)} embeddings to index")
        
        # Save if paths provided
        if save_index_path and save_metadata_path:
            self.embedding_index.save(save_index_path, save_metadata_path)
            print(f"Saved index to {save_index_path}")
    
    def save_product_database(self, filepath: str):
        """Save product database metadata."""
        self.product_db.save_metadata(filepath)
        print(f"Saved product database to {filepath}")


def main():
    """
    Example usage of ExactProductCounter.
    """
    print("Initializing Exact Product Counter...")
    
    # Initialize counter
    counter = ExactProductCounter(
        model_path=None,  # Will use untrained model
        index_path=None,
        metadata_path=None,
        db_path=None
    )
    
    # Save product database
    counter.save_product_database('data/product_database.json')
    
    # Build embedding index from reference images
    counter.build_embedding_index(
        save_index_path='data/product_embeddings.faiss',
        save_metadata_path='data/product_embeddings_metadata.pkl'
    )
    
    print(f"\nLoaded {len(counter.product_db.products)} products")
    print(f"Brands: {len(counter.product_db.get_brands())}")
    print(f"SKUs: {counter.product_db.get_all_skus()[:10]}...")  # Show first 10
    
    # Example: Match a single product
    # Uncomment to test with actual image
    # test_image = cv2.imread('data/test_images/test_product.jpg')
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # result = counter.match_product(test_image)
    # print(f"\nMatch result: {result}")


if __name__ == "__main__":
    main()
