"""
Training Script for Exact Product Matcher

This script trains the ExactProductMatcher model using reference images.
It uses self-supervised learning on the reference images to learn product embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json
from tqdm import tqdm
import random

from exact_product_matcher import ExactProductMatcher, ProductDatabase


class ProductDataset(Dataset):
    """
    Dataset for training product matcher.
    Loads images from reference_images directory.
    """
    
    def __init__(self, reference_dir: str = 'data/reference_images', augment: bool = True):
        self.reference_dir = Path(reference_dir)
        self.augment = augment
        
        # Load product database
        self.product_db = ProductDatabase(reference_dir)
        self.product_db.load_from_directory()
        
        # Create mappings
        self.sku_to_id = self.product_db.get_sku_to_id_mapping()
        self.brand_to_id = self.product_db.get_brand_to_id_mapping()
        
        # Build image list
        self.samples = []
        for sku, product_info in self.product_db.products.items():
            product_id = self.sku_to_id[sku]
            brand = product_info['brand']
            brand_id = self.brand_to_id[brand]
            
            for img_path in product_info['images']:
                self.samples.append({
                    'image_path': img_path,
                    'sku': sku,
                    'product_id': product_id,
                    'brand': brand,
                    'brand_id': brand_id
                })
        
        print(f"Dataset: {len(self.samples)} images, {len(self.product_db.products)} products, {len(self.brand_to_id)} brands")
        
        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            # Return random image if load fails
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        image = self.transform(image)
        
        return {
            'image': image,
            'product_id': sample['product_id'],
            'brand_id': sample['brand_id'],
            'sku': sample['sku']
        }


class TripletLoss(nn.Module):
    """
    Triplet loss for embedding learning.
    Pulls same products closer, pushes different products apart.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class Trainer:
    """
    Trainer for ExactProductMatcher model.
    """
    
    def __init__(
        self,
        model: ExactProductMatcher,
        train_dataset: ProductDataset,
        val_dataset: ProductDataset = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        save_dir: str = 'models'
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct_brand = 0
        correct_product = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            brand_labels = batch['brand_id'].to(self.device)
            product_labels = batch['product_id'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Classification losses
            brand_loss = self.classification_loss(outputs['brand'], brand_labels)
            product_loss = self.classification_loss(outputs['product'], product_labels)
            
            # Combined loss (weighted)
            loss = 0.3 * brand_loss + 0.7 * product_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Brand accuracy
            _, brand_pred = torch.max(outputs['brand'], 1)
            correct_brand += (brand_pred == brand_labels).sum().item()
            
            # Product accuracy
            _, product_pred = torch.max(outputs['product'], 1)
            correct_product += (product_pred == product_labels).sum().item()
            
            total += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'brand_acc': f'{100 * correct_brand / total:.2f}%',
                'prod_acc': f'{100 * correct_product / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        brand_acc = 100 * correct_brand / total
        product_acc = 100 * correct_product / total
        
        return avg_loss, brand_acc, product_acc
    
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return 0, 0, 0
        
        self.model.eval()
        
        total_loss = 0
        correct_brand = 0
        correct_product = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                brand_labels = batch['brand_id'].to(self.device)
                product_labels = batch['product_id'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Losses
                brand_loss = self.classification_loss(outputs['brand'], brand_labels)
                product_loss = self.classification_loss(outputs['product'], product_labels)
                loss = 0.3 * brand_loss + 0.7 * product_loss
                
                total_loss += loss.item()
                
                # Accuracies
                _, brand_pred = torch.max(outputs['brand'], 1)
                correct_brand += (brand_pred == brand_labels).sum().item()
                
                _, product_pred = torch.max(outputs['product'], 1)
                correct_product += (product_pred == product_labels).sum().item()
                
                total += images.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        brand_acc = 100 * correct_brand / total
        product_acc = 100 * correct_product / total
        
        return avg_loss, brand_acc, product_acc
    
    def train(self):
        """Full training loop."""
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 70)
        
        best_val_loss = float('inf')
        best_product_acc = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_brand_acc, train_product_acc = self.train_epoch()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Brand Accuracy: {train_brand_acc:.2f}%")
            print(f"Train Product Accuracy: {train_product_acc:.2f}%")
            
            # Validate
            if self.val_loader:
                val_loss, val_brand_acc, val_product_acc = self.validate()
                
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Brand Accuracy: {val_brand_acc:.2f}%")
                print(f"Val Product Accuracy: {val_product_acc:.2f}%")
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_product_acc > best_product_acc:
                    best_product_acc = val_product_acc
                    self.save_checkpoint('best_model.pth', epoch, val_loss, val_product_acc)
                    print(f"✓ Saved best model (Product Acc: {val_product_acc:.2f}%)")
            else:
                # No validation set, use training accuracy
                if train_product_acc > best_product_acc:
                    best_product_acc = train_product_acc
                    self.save_checkpoint('best_model.pth', epoch, train_loss, train_product_acc)
                    print(f"✓ Saved best model (Product Acc: {train_product_acc:.2f}%)")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, train_loss, train_product_acc)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best Product Accuracy: {best_product_acc:.2f}%")
        print("=" * 70)
    
    def save_checkpoint(self, filename: str, epoch: int, loss: float, accuracy: float):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }, checkpoint_path)


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'reference_dir': 'data/reference_images',
        'batch_size': 16,  # Smaller batch size for limited data
        'num_epochs': 100,  # More epochs for small dataset
        'learning_rate': 0.0001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models',
        'train_split': 0.8  # 80% train, 20% validation
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = ProductDataset(
        reference_dir=config['reference_dir'],
        augment=True
    )
    
    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(config['train_split'] * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset (without augmentation)
    val_dataset_no_aug = ProductDataset(
        reference_dir=config['reference_dir'],
        augment=False
    )
    val_indices = val_dataset.indices
    val_subset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_subset)}")
    
    # Initialize model
    num_brands = len(full_dataset.brand_to_id)
    num_products = len(full_dataset.sku_to_id)
    
    print(f"\nModel configuration:")
    print(f"  Brands: {num_brands}")
    print(f"  Products: {num_products}")
    
    model = ExactProductMatcher(
        num_brands=num_brands,
        num_exact_products=num_products,
        embedding_dim=256
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_subset,
        device=config['device'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir']
    )
    
    # Train
    trainer.train()
    
    # Save final model
    final_path = Path(config['save_dir']) / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Saved final model to {final_path}")
    
    # Save configuration
    config_path = Path(config['save_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved configuration to {config_path}")


if __name__ == "__main__":
    main()
