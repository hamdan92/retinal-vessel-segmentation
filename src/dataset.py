import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image
import cv2

class VesselSegmentationDataset(Dataset):
    """
    Dataset class for retinal vessel segmentation data loading and augmentation
    """
    def __init__(self, 
                 csv_file, 
                 image_dir,
                 mask_dir,
                 fov_dir=None,
                 transform=None,
                 phase='train',
                 preprocess=True,
                 img_size=(512, 512)):
        """
        Args:
            csv_file (str): Path to csv file with image filenames
            image_dir (str): Directory with preprocessed images
            mask_dir (str): Directory with ground truth masks
            fov_dir (str, optional): Directory with FOV masks
            transform (albumentations.Compose, optional): Albumentations transforms for augmentation
            phase (str): 'train', 'val', or 'test'
            preprocess (bool): Whether to use preprocessed images or raw ones
            img_size (tuple): Size to resize images to if not preprocessed
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fov_dir = fov_dir
        self.transform = transform
        self.phase = phase
        self.preprocess = preprocess
        self.img_size = img_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_name = str(img_name).strip()
        
        # Load image - either preprocessed or raw
        if self.preprocess:
            # Load preprocessed numpy array
            img_path = os.path.join(self.image_dir, 'images', f"{img_name}.npy")
            try:
                image = np.load(img_path)
            except FileNotFoundError:
                # Try with _preprocessed suffix
                img_path = os.path.join(self.image_dir, 'images', f"{img_name}_preprocessed.npy")
                image = np.load(img_path)
        else:
            # Load raw image and preprocess it
            file_ext = '.tif' if self.phase == 'train' else '.tif'
            img_path = os.path.join(self.image_dir, f"{img_name}{file_ext}")
            image = np.array(Image.open(img_path).convert('RGB'))
            # Extract green channel (provides better vessel contrast)
            image = image[:, :, 1]
            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Resize
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            
        # Load mask
        mask_file_ext = '.gif'
        if self.phase == 'train':
            mask_path = os.path.join(self.mask_dir, f"{img_name}_manual1{mask_file_ext}")
        else:
            mask_path = os.path.join(self.mask_dir, f"{img_name}_manual1{mask_file_ext}")
        
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
        except FileNotFoundError:
            # Try with different mask naming convention
            try:
                if self.phase == 'train':
                    mask_path = os.path.join(self.mask_dir, f"{img_name.split('_')[0]}_manual1{mask_file_ext}")
                else:
                    mask_path = os.path.join(self.mask_dir, f"{img_name.split('_')[0]}_manual1{mask_file_ext}")
                mask = np.array(Image.open(mask_path).convert('L'))
            except FileNotFoundError:
                # Try with npy file
                try:
                    mask_path = os.path.join(self.image_dir, 'masks', f"{img_name}.npy")
                    mask = np.load(mask_path)
                except FileNotFoundError:
                    mask_path = os.path.join(self.image_dir, 'masks', f"{img_name}_mask.npy")
                    mask = np.load(mask_path)
        
        # Resize mask if using raw images
        if not self.preprocess:
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Load FOV mask if provided
        if self.fov_dir is not None:
            if self.phase == 'train':
                fov_path = os.path.join(self.fov_dir, f"{img_name}_training_mask{mask_file_ext}")
            else:
                fov_path = os.path.join(self.fov_dir, f"{img_name}_test_mask{mask_file_ext}")
            
            try:
                fov = np.array(Image.open(fov_path).convert('L'))
            except FileNotFoundError:
                # Try different naming convention
                try:
                    if self.phase == 'train':
                        fov_path = os.path.join(self.fov_dir, f"{img_name.split('_')[0]}_training_mask{mask_file_ext}")
                    else:
                        fov_path = os.path.join(self.fov_dir, f"{img_name.split('_')[0]}_test_mask{mask_file_ext}")
                    fov = np.array(Image.open(fov_path).convert('L'))
                except FileNotFoundError:
                    # Use all ones as FOV if mask not found
                    fov = np.ones_like(mask)
                    
            if not self.preprocess:
                fov = cv2.resize(fov, self.img_size, interpolation=cv2.INTER_NEAREST)
            fov = fov.astype(np.float32) / 255.0
        else:
            fov = np.ones_like(mask)
        
        # Apply augmentations if in training mode
        if self.transform and self.phase == 'train':
            # Prepare for albumentation format
            if image.shape[0] == 1:  # If single channel
                image_for_aug = image[0]
            else:
                image_for_aug = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                
            # Apply augmentation
            augmented = self.transform(image=image_for_aug, mask=mask, fov=fov)
            
            image_aug = augmented['image']
            mask_aug = augmented['mask']
            fov_aug = augmented['fov']
            
            # Convert back to channel-first
            if len(image_aug.shape) == 2:  # If single channel
                image_aug = np.expand_dims(image_aug, axis=0)
            else:
                image_aug = np.transpose(image_aug, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            
            # Add channel dimension to mask and fov
            mask_aug = np.expand_dims(mask_aug, axis=0)
            fov_aug = np.expand_dims(fov_aug, axis=0)
            
            image = image_aug
            mask = mask_aug
            fov = fov_aug
        else:
            # Add channel dimension to mask and fov
            mask = np.expand_dims(mask, axis=0)
            fov = np.expand_dims(fov, axis=0)
        
        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        fov_tensor = torch.from_numpy(fov)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'fov': fov_tensor,
            'image_name': img_name
        } 