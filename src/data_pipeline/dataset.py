import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class FlexibleGraspDataset(Dataset):
    def __init__(self, size=100, img_size=(256, 256), transforms=None):
        """
        Mock dataset that generates synthetic FPC/Wafer images and target heatmaps.
        
        Args:
            size (int): Number of mock samples to generate.
            img_size (tuple): (height, width) of the images.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.size = size
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate a mock image (e.g., green/brown background for FPC, gray for Wafer)
        img = np.random.randint(50, 200, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Add a mock "circuit" structure (lines)
        cv2.line(img, (50, 50), (200, 200), (0, 0, 0), 3)
        cv2.line(img, (50, 200), (200, 50), (0, 0, 0), 3)

        # Generate a mock safe grasp heatmap (1 channel)
        # Hot regions (1.0) around the edges (safe), cold regions (0.0) in the center (circuits/fragile)
        heatmap = np.zeros(self.img_size, dtype=np.float32)
        # Safe edge zones
        heatmap[:, :20] = 1.0  # Left edge
        heatmap[:, -20:] = 1.0 # Right edge
        heatmap[:20, :] = 1.0  # Top edge
        heatmap[-20:, :] = 1.0 # Bottom edge
        
        # Add smooth transition (Gaussian blur)
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

        if self.transforms:
            # Albumentations expects HWC images and HW masks
            augmented = self.transforms(image=img, mask=heatmap)
            img = augmented['image']
            heatmap = augmented['mask']
        else:
            # Fallback to simple tensor conversion if no transforms
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            heatmap = torch.from_numpy(heatmap).float()

        # Ensure heatmap has a channel dimension (C, H, W)
        heatmap = heatmap.unsqueeze(0)

        # We also return a "stress mask" which indicates critical IC areas that shouldn't be touched
        # This will be used by our custom loss function. 
        # For mock data, let's say the center 100x100 is highly critical.
        stress_mask = torch.zeros((1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        cx, cy = self.img_size[0]//2, self.img_size[1]//2
        stress_mask[:, cx-50:cx+50, cy-50:cy+50] = 1.0

        return {
            'image': img,
            'heatmap': heatmap,
            'stress_mask': stress_mask
        }
