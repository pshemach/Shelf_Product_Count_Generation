import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.product_dataset import ProductDataset
from src.embedding.triplet_loss import TripletSampler, TripletLoss
import torch

dataset = ProductDataset()
sampler = TripletSampler(dataset=dataset)
anchor_idx, positive_idx, negative_idx = sampler.sample_triplet()
print(anchor_idx, positive_idx, negative_idx)
criterion = TripletLoss()
loss = criterion(torch.tensor([1,2.5,3]), torch.tensor([1.5,2.4,2.9]), torch.tensor([140,5,7]))
print(loss)