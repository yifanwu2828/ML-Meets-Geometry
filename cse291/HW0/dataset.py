import os

import numpy as np
from torch.utils.data import Dataset

class simDataset(Dataset):
    def __init__(self, data_dir, transform=None, train = True) -> None:
        super().__init__()
        self.data = np.load(data_dir)
        self.images = self.data['images']
        self.masks = self.data['edges']
        self.transform = transform
        assert self.images.shape[0] == self.masks.shape[0]
        
        
        self.percent = 0.8 if train else 0.2
        self.dataset_size = int(self.images.shape[0] * self.percent)
        self.images = self.images[:self.dataset_size]
        self.masks = self.masks[:self.dataset_size]
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = np.asarray(self.masks[idx], dtype=np.float32)
        mask[mask == 255] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask
        
        