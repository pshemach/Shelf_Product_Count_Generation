import torch
import torch.nn as nn
import random
from typing import Tuple

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        anchor: embedding of anchor image
        positive: embedding of positive (same product) image
        negative: embedding of negative (different product) image
        """
        distance_positive = nn.functional.pairwise_distance(anchor, positive)
        distance_negative = nn.functional.pairwise_distance(anchor, negative)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        
        return loss.mean()

class TripletSampler:
    """Samples triplets (anchor, positive, negative) from dataset"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.product_to_indices = {}
    
        for idx, product_id in enumerate(self.dataset.product_ids):
            if product_id not in self.product_to_indices:
                self.product_to_indices[product_id] = []
            self.product_to_indices[product_id].append(idx)
            
    def sample_triplet(self) -> Tuple[int, int, int]:
        """Sample anchor, positive, negative indices"""
        # Random anchor product
        anchor_product = random.choice(list(self.product_to_indices.keys()))
        anchor_idx, positive_idx = random.sample(
            self.product_to_indices[anchor_product], 2
        )
        
        # Random negative product (different from anchor)
        negative_product = random.choice(
            [p for p in self.product_to_indices.keys() if p != anchor_product]
            )
        
        negative_idx = random.choice(self.product_to_indices[negative_product])
        
        return anchor_idx, positive_idx, negative_idx