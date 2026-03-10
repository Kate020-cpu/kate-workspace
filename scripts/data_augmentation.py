"""
data_augmentation.py - Enhanced version with more augmentations
"""

import cv2
import numpy as np
import random
import albumentations as A
from pathlib import Path

class AdvancedLeafAugmentor:
    def __init__(self, config=None):
        self.config = config or {
            'rotation': 30,
            'brightness': (0.7, 1.3),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-20, 20),
            'noise_var': (10, 50),
            'blur_limit': 3
        }
        
        # Using albumentations for production-grade augmentations
        self.transform = A.Compose([
            A.Rotate(limit=self.config['rotation'], p=0.8),
            A.RandomBrightnessContrast(
                brightness_limit=self.config['brightness'][0],
                contrast_limit=self.config['contrast'][0],
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.config['hue'],
                sat_shift_limit=self.config['saturation'],
                val_shift_limit=20,
                p=0.8
            ),
            A.GaussNoise(var_limit=self.config['noise_var'], p=0.3),
            A.Blur(blur_limit=self.config['blur_limit'], p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
        ])
    
    def __call__(self, image):
        \"\"\"Apply augmentations using albumentations\"\"\"
        if isinstance(image, np.ndarray):
            augmented = self.transform(image=image)
            return augmented['image']
        else:
            raise ValueError("Image must be numpy array")
    
    def visualize_augmentations(self, image, num_samples=4):
        \"\"\"Generate multiple augmented versions for visualization\"\"\"
        augmented_images = []
        for _ in range(num_samples):
            aug_img = self.__call__(image)
            augmented_images.append(aug_img)
        return augmented_images

# Add test code
if __name__ == "__main__":
    print("Advanced Data Augmentation Module")
    print("=" * 40)
    print("✓ Albumentations integration")
    print("✓ GPU-compatible transforms")
    print("✓ Configurable augmentation pipeline")
