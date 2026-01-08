import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple

class ProductDataset(Dataset):
    """
    Dataset for product embedding training.
    Loads images from reference_images directory organized by product ID folders.
    """
    def __init__(
        self,
        reference_dir: str ='data/reference_images',
        image_size: int = 224,
        augment: bool = True
        ):
        super().__init__()
        self.reference_dir = Path(reference_dir)
        self.image_size = image_size
        
        # Load all images with their product IDs
        self.images = []
        self.product_ids = []
        self.product_to_images = {}
        
        product_folders = sorted([d for d in self.reference_dir.iterdir() if d.is_dir()])
    
        for product_folder in product_folders[:2]:
            product_id = product_folder.name
            
            image_files = sorted(product_folder.glob('*.jpg'))
            
            self.product_to_images[product_id] = []
            
            for image_path in image_files:
                self.images.append(str(image_path))
                self.product_ids.append(product_id)
                self.product_to_images[product_id].append(len(self.images)-1)
                
            # Create product ID to integer mapping
            unique_products = sorted(set(self.product_ids))