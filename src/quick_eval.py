import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import albumentations as A
from skimage import filters, measure, morphology

# A simple U-Net model
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)  # Add dropout to prevent overfitting
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 = 256 + 256 (skip)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 + 128 (skip)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1_features = self.enc1(x)
        x = self.pool1(enc1_features)
        
        enc2_features = self.enc2(x)
        x = self.pool2(enc2_features)
        
        enc3_features = self.enc3(x)
        x = self.pool3(enc3_features)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, enc3_features], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_features], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_features], dim=1)
        x = self.dec1(x)
        
        # Final output
        x = self.final(x)
        
        return x

# Simple dataset class for loading test images
class SimpleRetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        
        # Get file names
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Extract green channel (provides better vessel contrast)
        image = image[:, :, 1]
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Load mask - get from image name
        mask_name = f"{img_name.split('_')[0]}_manual1.gif"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Check if file exists
        if not os.path.exists(mask_path):
            print(f"Warning: Mask {mask_path} not found")
            mask = np.zeros(self.img_size, dtype=np.float32)
        else:
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 0).astype(np.float32)  # Binarize the mask
        
        # Apply augmentations if available
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'image_name': img_name
        }

def get_train_transforms(img_size=(256, 256)):
    """
    Enhanced training transforms for better vessel segmentation
    """
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=1.5, sigma=50, p=0.5),  # Increased for better vessel deformation
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),  # Increased contrast
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),  # Add CLAHE for better vessel contrast
        A.GaussNoise(p=0.3),
    ])

def get_val_transforms(img_size=(256, 256)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
    ])

# Dice loss for better segmentation performance
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, preds, targets):
        # Flatten predictions and targets
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.4, tversky_weight=0.4, bce_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=0.5, beta=0.7)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targets):
        # Apply sigmoid to get probabilities for Dice and Tversky
        probs = torch.sigmoid(preds)
        
        # Calculate individual losses
        dice = self.dice_loss(probs, targets)
        tversky = self.tversky_loss(probs, targets)
        bce = self.bce_loss(preds, targets)
        
        # Combine losses with weights
        return self.dice_weight * dice + self.tversky_weight * tversky + self.bce_weight * bce

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Controls weight of false positives
        self.beta = beta    # Controls weight of false negatives (higher = more recall)
        self.smooth = smooth
        
    def forward(self, preds, targets):
        # Flatten prediction and target tensors
        batch_size = preds.size(0)
        preds = preds.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Calculate Tversky index
        tp = torch.sum(preds * targets, dim=1)
        fp = torch.sum(preds * (1 - targets), dim=1) * self.alpha
        fn = torch.sum((1 - preds) * targets, dim=1) * self.beta
        
        tversky = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        
        # Return Tversky loss
        return 1 - torch.mean(tversky)

def process_prediction(pred, mask, min_size=30):
    """
    Enhanced post-processing pipeline optimized for thin vessels
    """
    # Convert to numpy if tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Apply adaptive thresholding
    # Use a lower threshold for thin vessels (Otsu * 0.85)
    otsu_threshold = filters.threshold_otsu(pred)
    binary_mask = (pred > (otsu_threshold * 0.85)).astype(np.uint8)
    
    # Remove small objects
    labeled_mask = measure.label(binary_mask)
    cleaned_mask = morphology.remove_small_objects(labeled_mask, min_size=min_size)
    cleaned_mask = (cleaned_mask > 0).astype(np.uint8)
    
    # Apply morphological refinement
    # Create different kernels for better preservation of vessel structures
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    
    # Apply closing to fill gaps in vessels
    closed_h = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, line_kernel_h)
    closed_v = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, line_kernel_v)
    closed_disk = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, disk_kernel)
    
    # Combine results from different morphological operations
    refined_mask = np.maximum(np.maximum(closed_h, closed_v), closed_disk)
    
    # Apply opening to remove small artifacts
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, disk_kernel)
    
    return refined_mask

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set parameters
    img_size = (512, 512)  # Increased resolution
    
    # Create transforms
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = SimpleRetinalDataset(
        image_dir="data/DRIVE/training/images",
        mask_dir="data/DRIVE/training/1st_manual",
        img_size=img_size,
        transform=train_transform
    )
    
    test_dataset = SimpleRetinalDataset(
        image_dir="data/DRIVE/test/images",
        mask_dir="data/DRIVE/test/1st_manual",
        img_size=img_size,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model with deeper architecture
    model = SimpleUNet(in_channels=1, out_channels=1)
    model.to(device)
    
    # Create loss function and optimizer with our optimized settings
    criterion = CombinedLoss(dice_weight=0.4, tversky_weight=0.4, bce_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Perform training 
    model.train()
    print("Starting training...")
    
    num_epochs = 20  # Increased number of epochs
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        # Calculate average training loss
        avg_train_loss = epoch_loss / batch_count
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / val_count
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model weights
            torch.save(model.state_dict(), 'best_model.pth')
            
        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Evaluate on test set
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    print("\nEvaluating on test set with enhanced post-processing...")
    
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        # Collect all predictions and masks
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Make predictions
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
    
    # Process with enhanced postprocessing
    processed_preds = []
    for i in range(len(all_preds)):
        for j in range(all_preds[i].shape[0]):
            pred = all_preds[i][j, 0]  # Get single prediction
            mask = all_masks[i][j, 0]  # Get corresponding mask
            
            # Apply enhanced post-processing
            processed_pred = process_prediction(pred, mask, min_size=30)
            processed_preds.append(processed_pred.flatten())
            all_masks[i][j, 0] = mask.flatten()
    
    # Flatten all masks for evaluation
    flat_masks = []
    for masks_batch in all_masks:
        for j in range(masks_batch.shape[0]):
            flat_masks.append(masks_batch[j, 0].flatten())
    
    # Convert to numpy arrays
    processed_preds_flat = np.array(processed_preds)
    flat_masks = np.array(flat_masks)
    
    # Calculate metrics
    f1 = f1_score(flat_masks, processed_preds_flat)
    precision = precision_score(flat_masks, processed_preds_flat)
    recall = recall_score(flat_masks, processed_preds_flat)
    
    print("\nEnhanced Test Set Metrics:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    main() 