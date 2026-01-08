import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

class ProductEmbeddingModel(nn.Module):
    """
    Embedding model for product similarity matching.
    """
    def __init__(
        self,
        backbone_name: str = 'efficientnet_v2_s', 
        embedding_dim: int = 1024,
        freeze_backbone: bool = True, 
        unfreeze_last_n_layers: int = 0
        ):
        super().__init__()
        # Load pre-trained backbone
        if backbone_name == 'efficientnet_v2_s':
            backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            backbone.classifier = nn.Identity() # Remove classifier
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.backbone = backbone
        self.backbone_name = backbone_name
        
        # Setup backbone freezing
        self._setup_freezing(freeze_backbone, unfreeze_last_n_layers)
        
        # Embedding Head
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 1024),       # Step 1: Expand dimensions
            nn.ReLU(),                          # Step 2: Non-linearity
            nn.Dropout(0.3),                    # Step 3: Regularization
            nn.Linear(1024, embedding_dim),     # Step 4: Project to final dimension
            nn.LayerNorm(embedding_dim)         # Step 5: Normalize
            )
        
    def _setup_freezing(self, freeze_backbone, unfreeze_last_n_layers):
        """Setup which backbone layers to freeze/unfreeze"""
        if not freeze_backbone:
            # Don't freeze anything
            return
        
        if unfreeze_last_n_layers > 0:
            # Freeze all layers first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layers
            if self.backbone_name == 'efficientnet_v2_s':
                # EfficientNet V2 S structure: features -> avgpool -> classifier (removed)
                # features contains multiple blocks
                blocks = list(self.backbone.features.children())
                total_blocks = len(blocks)
                
                layers_to_unfreeze = min(unfreeze_last_n_layers, total_blocks)
                from_layer = total_blocks - layers_to_unfreeze
                last_layer = total_blocks
                
                for i in range(from_layer, last_layer):
                    for param in blocks[i].parameters():
                        param.requires_grad = True                
        else:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        # L2 normalize embeddings
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    