import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import torch


def get_training_augmentation(
    rotate_limit=35,
    elastic_sigma=50,
    elastic_alpha=1,
    brightness_limit=0.2,
    contrast_limit=0.2,
    blur_prob=0.2,
    gamma_limit=(1.0, 1.5)
):
    """
    Create training augmentation pipeline using Albumentations
    
    Args:
        rotate_limit: Maximum rotation angle in degrees
        elastic_sigma: Sigma for elastic transformation
        elastic_alpha: Alpha for elastic transformation
        brightness_limit: Maximum brightness adjustment factor
        contrast_limit: Maximum contrast adjustment factor
        blur_prob: Probability of applying Gaussian blur
        gamma_limit: Range for gamma adjustment
        
    Returns:
        Albumentations Compose object with transformations
    """
    # Create augmentation pipeline
    return A.Compose([
        # Spatial augmentations - applied to both image and mask
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=rotate_limit, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            p=0.6
        ),
        
        # Pixel-level augmentations - applied to image only
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=0.8),
        ], p=0.7),
        
        A.GaussianBlur(blur_limit=3, p=blur_prob),
    ], p=1.0)


def get_test_augmentation():
    """
    Create test time augmentation (TTA) pipeline with 8-way augmentations
    
    Returns:
        List of Albumentations Compose objects, each with a different transformation
    """
    # No augmentation for default prediction
    default = A.Compose([])
    
    # Horizontal flip
    h_flip = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])
    
    # Vertical flip
    v_flip = A.Compose([
        A.VerticalFlip(p=1.0),
    ])
    
    # Horizontal + Vertical flip
    hv_flip = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
    ])
    
    # 90 degree rotation
    rot90 = A.Compose([
        A.Rotate(limit=(90, 90), p=1.0),
    ])
    
    # 90 degree rotation + horizontal flip
    rot90h = A.Compose([
        A.Rotate(limit=(90, 90), p=1.0),
        A.HorizontalFlip(p=1.0),
    ])
    
    # 180 degree rotation
    rot180 = A.Compose([
        A.Rotate(limit=(180, 180), p=1.0),
    ])
    
    # 270 degree rotation
    rot270 = A.Compose([
        A.Rotate(limit=(270, 270), p=1.0),
    ])
    
    # Return all 8 augmentations for more robust TTA
    return [default, h_flip, v_flip, hv_flip, rot90, rot90h, rot180, rot270]


def apply_augmentation(image, mask, augmentation):
    """
    Apply augmentation to image and mask
    
    Args:
        image: Input image as numpy array [H, W, C] or [H, W]
        mask: Binary mask as numpy array [H, W]
        augmentation: Albumentations Compose object
        
    Returns:
        Augmented image and mask
    """
    # Ensure image and mask have the right shapes and types
    if len(image.shape) == 2:
        # Add channel dimension if grayscale
        image = np.expand_dims(image, axis=-1)
    
    # Make sure mask is 2D
    mask_2d = mask.squeeze()
    
    # Apply augmentation
    augmented = augmentation(image=image, mask=mask_2d)
    
    # Get augmented image and mask
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    # Return image as is (with or without channel dim)
    if len(image.shape) == 2:
        aug_image = aug_image.squeeze()
    
    return aug_image, aug_mask


def mixup_masks(mask1, mask2, alpha=0.3):
    """
    Apply mixup to two masks with alpha blending
    
    Args:
        mask1: First mask as numpy array
        mask2: Second mask as numpy array
        alpha: Alpha blending factor
        
    Returns:
        Mixed mask
    """
    # Apply mixup
    mixed_mask = alpha * mask1 + (1 - alpha) * mask2
    
    # Thresholding to get binary mask
    return (mixed_mask > 0.5).astype(np.float32)


def apply_cutmix(mask1, mask2, patch_size=64):
    """
    Apply CutMix to two masks by swapping a random patch
    
    Args:
        mask1: First mask as numpy array
        mask2: Second mask as numpy array
        patch_size: Size of the patch to swap
        
    Returns:
        Mixed mask
    """
    h, w = mask1.shape
    
    # Generate random coordinates for top-left corner of patch
    top = np.random.randint(0, h - patch_size)
    left = np.random.randint(0, w - patch_size)
    
    # Create mixed mask
    mixed_mask = mask1.copy()
    
    # Replace patch in mask1 with patch from mask2
    mixed_mask[top:top+patch_size, left:left+patch_size] = mask2[top:top+patch_size, left:left+patch_size]
    
    return mixed_mask


def apply_mask_augmentation(batch_masks, prob=0.3):
    """
    Apply MixUp or CutMix to a batch of masks
    
    Args:
        batch_masks: Batch of masks as numpy array [B, H, W]
        prob: Probability of applying MixUp or CutMix
        
    Returns:
        Augmented masks
    """
    batch_size = len(batch_masks)
    augmented_masks = batch_masks.copy()
    
    # Apply MixUp or CutMix with probability prob
    for i in range(batch_size):
        if np.random.random() < prob:
            # Randomly select another mask from the batch
            j = np.random.choice([k for k in range(batch_size) if k != i])
            
            # Randomly choose between MixUp and CutMix
            if np.random.random() < 0.5:
                # Apply MixUp
                alpha = np.random.beta(0.2, 0.2)  # Beta distribution for mixup parameter
                augmented_masks[i] = mixup_masks(batch_masks[i], batch_masks[j], alpha)
            else:
                # Apply CutMix
                patch_size = np.random.randint(32, 128)  # Random patch size
                augmented_masks[i] = apply_cutmix(batch_masks[i], batch_masks[j], patch_size)
    
    return augmented_masks


def visualize_augmentations(image_path, mask_path, num_samples=5, save_dir="data/visualizations"):
    """
    Visualize different augmentations applied to the same image
    
    Args:
        image_path: Path to image file (can be .npy or image file)
        mask_path: Path to mask file (can be .npy or image file)
        num_samples: Number of augmented samples to generate
        save_dir: Directory to save visualization
    """
    # Load image
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    else:
        image = np.array(Image.open(image_path))
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load mask
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel if RGB
        mask = (mask > 0).astype(np.float32)
    
    # Create augmentation
    augmentation = get_train_transforms()
    
    # Generate augmented samples
    augmented_samples = []
    
    # Always include the original
    augmented_samples.append((image, mask, "Original"))
    
    # Generate augmented samples
    for i in range(num_samples):
        # Apply augmentation
        aug_image, aug_mask = apply_augmentation(image, mask, augmentation)
        augmented_samples.append((aug_image, aug_mask, f"Augmented {i+1}"))
    
    # Create visualization
    num_rows = (len(augmented_samples) + 2) // 3
    plt.figure(figsize=(15, 5 * num_rows))
    
    for i, (img, msk, title) in enumerate(augmented_samples):
        # Create overlay: red channel = mask, green channel = image
        overlay = np.zeros((img.shape[0], img.shape[1], 3))
        overlay[:, :, 0] = msk  # Red = mask
        if len(img.shape) == 3:
            overlay[:, :, 1] = img[:, :, 0]  # Green = image
        else:
            overlay[:, :, 1] = img  # Green = image
        
        # Plot side by side
        plt.subplot(num_rows, 3, i+1)
        plt.imshow(overlay)
        plt.title(title)
        plt.axis('off')
    
    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "augmentation_samples.png"), dpi=150)
    print(f"Augmentation samples saved to {os.path.join(save_dir, 'augmentation_samples.png')}")


def create_augmented_dataset(
    input_dir,
    output_dir,
    num_augmentations=5,
    random_seed=42
):
    """
    Create an augmented dataset from preprocessed images
    
    Args:
        input_dir: Directory with preprocessed data (should have 'images' and 'masks' subdirs)
        output_dir: Directory to save augmented data
        num_augmentations: Number of augmentations to generate per image
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # Create augmentation pipeline
    augmentation = get_train_transforms()
    
    # Get list of preprocessed images
    image_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'masks')
    
    # Include only .npy files
    images = [f for f in os.listdir(image_dir) if f.endswith('_preprocessed.npy')]
    
    # Process each image
    for image_file in images:
        # Get image ID
        image_id = image_file.split('_')[0]
        
        # Load image and mask
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, f"{image_id}_mask.npy")
        
        image = np.load(image_path)
        mask = np.load(mask_path)
        
        # First, copy original to output dir
        np.save(os.path.join(output_dir, 'images', f"{image_id}_orig.npy"), image)
        np.save(os.path.join(output_dir, 'masks', f"{image_id}_orig.npy"), mask)
        
        # Create augmentations
        for i in range(num_augmentations):
            # Apply augmentation
            aug_image, aug_mask = apply_augmentation(image, mask, augmentation)
            
            # Save augmented data
            np.save(
                os.path.join(output_dir, 'images', f"{image_id}_aug_{i}.npy"),
                aug_image
            )
            np.save(
                os.path.join(output_dir, 'masks', f"{image_id}_aug_{i}.npy"),
                aug_mask
            )
    
    # Count files
    aug_images = len(os.listdir(os.path.join(output_dir, 'images')))
    aug_masks = len(os.listdir(os.path.join(output_dir, 'masks')))
    
    print(f"Augmentation complete!")
    print(f"Original dataset: {len(images)} images")
    print(f"Augmented dataset: {aug_images} images and {aug_masks} masks")
    print(f"Expansion factor: {aug_images / len(images):.1f}x")


def get_train_transforms(
    img_size=(512, 512),
    rotate_limit=35,
    elastic_sigma=50,
    elastic_alpha=1.5,  # Increased for more dramatic deformations
    brightness_limit=0.3,  # Increased for more variation
    contrast_limit=0.3,    # Increased for better vessel contrast
    gamma_limit=(1.0, 1.3),  # Fixed: all values must be >= 1
    prob=0.7,  # Increased probability of augmentations
    use_clahe=True,
    use_grid_distortion=True
):
    """
    Enhanced training augmentation pipeline optimized for thin vessel detection
    
    Args:
        img_size: Target image size (height, width)
        rotate_limit: Maximum rotation angle in degrees
        elastic_sigma: Sigma for elastic transformation
        elastic_alpha: Alpha for elastic transformation
        brightness_limit: Maximum brightness adjustment factor
        contrast_limit: Maximum contrast adjustment factor
        gamma_limit: Range for gamma adjustment
        prob: Probability of applying augmentations
        use_clahe: Whether to use CLAHE for contrast enhancement
        use_grid_distortion: Whether to use grid distortion
        
    Returns:
        Albumentations Compose object with transformations
    """
    # Create augmentation pipeline
    transforms_list = [
        # Resize first if needed
        A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_LINEAR),
        
        # Spatial augmentations - applied to both image and mask
        A.RandomRotate90(p=prob),
        A.Rotate(limit=rotate_limit, p=prob),
        A.HorizontalFlip(p=prob),
        A.VerticalFlip(p=prob),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=rotate_limit, p=prob*0.7),
        
        # Enhanced elastic transform for better deformation of vessels
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            alpha_affine=15,  # Add affine transformations
            p=prob
        ),
        
        # Pixel-level augmentations - applied to image only
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=0.8),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),  # Add CLAHE for better vessel contrast
        ], p=prob),
        
        # Blur/Noise augmentations
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(p=0.3),
            A.MedianBlur(blur_limit=3, p=1.0)
        ], p=0.3),
    ]
    
    # Add grid distortion for more complex deformations
    if use_grid_distortion:
        transforms_list.append(
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3)
        )
    
    # Add additional transformations if requested
    if use_clahe:
        transforms_list.append(
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
        )
    
    # Build the final composition
    return A.Compose(transforms_list)


def get_val_transforms(img_size=(512, 512)):
    """
    Create validation/test transforms (no augmentation, just resize if needed)
    
    Args:
        img_size: Target image size (height, width)
        
    Returns:
        Albumentations Compose object
    """
    # No augmentation for validation
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_LINEAR)
    ])


def apply_test_time_augmentation(model, image, device):
    """
    Apply 8-way test-time augmentation (TTA) using the model
    
    Args:
        model: PyTorch model
        image: Input image tensor [B, C, H, W]
        device: Device for computation
    
    Returns:
        Average of predictions from all augmentations
    """
    # Get augmentations
    transforms = get_test_augmentation()
    batch_size, channels, height, width = image.size()
    
    # Ensure model is in eval mode
    model.eval()
    
    # Store predictions
    preds = []
    
    # Original prediction
    with torch.no_grad():
        output = model(image)
        if isinstance(output, list):
            output = output[0]  # Use the main output for deep supervision models
        preds.append(torch.sigmoid(output))
    
    # Apply each augmentation
    for transform in transforms[1:]:  # Skip the first one (identity)
        aug_images = []
        
        # Apply augmentation to each image in batch
        for i in range(batch_size):
            img = image[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
            
            # Apply augmentation (transforms expect [H, W, C])
            augmented = transform(image=img)['image']
            
            # Convert back to tensor
            aug_img = torch.from_numpy(augmented).permute(2, 0, 1).to(device)
            aug_images.append(aug_img)
        
        # Stack augmented images
        aug_batch = torch.stack(aug_images)
        
        # Get prediction
        with torch.no_grad():
            aug_output = model(aug_batch)
            if isinstance(aug_output, list):
                aug_output = aug_output[0]
            
            # Apply inverse augmentation to prediction
            aug_pred = torch.sigmoid(aug_output)
            inv_aug_pred = inverse_augmentation(aug_pred, transform, batch_size, height, width, device)
            
            preds.append(inv_aug_pred)
    
    # Average predictions
    ensemble_pred = torch.stack(preds).mean(dim=0)
    
    return ensemble_pred.cpu().numpy()


def inverse_augmentation(pred, transform, batch_size, height, width, device):
    """
    Apply inverse augmentation to prediction
    
    Args:
        pred: Prediction tensor [B, C, H, W]
        transform: Augmentation transform
        batch_size: Batch size
        height, width: Original image dimensions
        device: Device for computation
    
    Returns:
        Inversely augmented prediction
    """
    inv_preds = []
    
    # Apply inverse augmentation to each prediction in batch
    for i in range(batch_size):
        p = pred[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        
        # Apply inverse augmentation based on transform type
        if isinstance(transform, A.Compose):
            transforms = transform.transforms
            
            for t in reversed(transforms):
                if isinstance(t, A.HorizontalFlip):
                    p = np.flip(p, axis=1)
                elif isinstance(t, A.VerticalFlip):
                    p = np.flip(p, axis=0)
                elif isinstance(t, A.Rotate) or isinstance(t, A.RandomRotate90):
                    if hasattr(t, 'limit') and t.limit[0] == 90 and t.limit[1] == 90:
                        p = np.rot90(p, k=3)  # Rotate 270 degrees (inverse of 90)
                    elif hasattr(t, 'limit') and t.limit[0] == 180 and t.limit[1] == 180:
                        p = np.rot90(p, k=2)  # Rotate 180 degrees (inverse of 180)
                    elif hasattr(t, 'limit') and t.limit[0] == 270 and t.limit[1] == 270:
                        p = np.rot90(p, k=1)  # Rotate 90 degrees (inverse of 270)
                    else:
                        # For random rotations, just resize to original size
                        p = cv2.resize(p, (width, height))
        
        # Convert back to tensor
        inv_pred = torch.from_numpy(p).permute(2, 0, 1).to(device)
        inv_preds.append(inv_pred)
    
    # Stack inversely augmented predictions
    return torch.stack(inv_preds)


if __name__ == "__main__":
    # Visualize different augmentations on a sample image
    image_path = "data/preprocessed/train/images/21_preprocessed.npy"
    mask_path = "data/preprocessed/train/masks/21_mask.npy"
    
    visualize_augmentations(image_path, mask_path, num_samples=8)
    
    # Create augmented dataset from training data
    create_augmented_dataset(
        input_dir="data/preprocessed/train",
        output_dir="data/augmented",
        num_augmentations=5,  # Create 5 augmentations per image
        random_seed=42
    ) 