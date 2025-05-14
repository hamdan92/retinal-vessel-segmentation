import cv2
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def crop_to_fov(image, fov_mask):
    """
    Crop the image to the field of view using the FOV mask
    
    Args:
        image: Input image (RGB or grayscale)
        fov_mask: Binary FOV mask
        
    Returns:
        Cropped image (with areas outside FOV set to 0)
    """
    # Ensure FOV mask is binary
    fov_binary = (fov_mask > 0).astype(np.uint8)
    
    # Apply FOV mask
    if len(image.shape) == 3:  # RGB image
        # Expand mask to 3 channels
        fov_3ch = np.stack([fov_binary] * 3, axis=2)
        return image * fov_3ch
    else:  # Grayscale image
        return image * fov_binary


def extract_green_channel(image):
    """
    Extract the green channel from an RGB image
    
    Args:
        image: RGB image
        
    Returns:
        Green channel as grayscale image
    """
    # Check if image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image[:, :, 1]
    
    # If already grayscale, return as is
    return image


def convert_to_lab(image):
    """
    Convert RGB image to LAB color space and extract 'a' channel
    
    Args:
        image: RGB image
        
    Returns:
        'a' channel from LAB color space
    """
    # Check if image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Extract 'a' channel
        return lab_image[:, :, 1]
    
    # If already grayscale, return as is
    return image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE enhanced image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    return clahe.apply(image)


def resize_image(image, target_size=(512, 512), keep_aspect_ratio=False):
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size as tuple (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    # Get current dimensions
    h, w = image.shape[:2]
    
    if keep_aspect_ratio:
        # Calculate scaling factor
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        if len(image.shape) == 3:  # RGB image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:  # Grayscale image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create empty target image
        if len(image.shape) == 3:  # RGB
            result = np.zeros((target_size[1], target_size[0], image.shape[2]), dtype=image.dtype)
        else:  # Grayscale
            result = np.zeros((target_size[1], target_size[0]), dtype=image.dtype)
        
        # Calculate position to paste resized image
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        
        # Paste resized image
        if len(image.shape) == 3:  # RGB
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
        else:  # Grayscale
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    else:
        # Simple resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image, min_val=0.0, max_val=1.0):
    """
    Normalize image to [min_val, max_val] range
    
    Args:
        image: Input image
        min_val: Minimum value after normalization
        max_val: Maximum value after normalization
        
    Returns:
        Normalized image
    """
    # Convert to float
    image_float = image.astype(np.float32)
    
    # Get min and max values
    img_min = np.min(image_float)
    img_max = np.max(image_float)
    
    # Avoid division by zero
    if img_max > img_min:
        # Normalize to [0, 1]
        normalized = (image_float - img_min) / (img_max - img_min)
        
        # Scale to [min_val, max_val]
        if min_val != 0.0 or max_val != 1.0:
            normalized = normalized * (max_val - min_val) + min_val
    else:
        normalized = np.zeros_like(image_float)
    
    return normalized


def preprocess_image(image_path, fov_path=None, target_size=(512, 512), 
                     use_green_channel=True, apply_clahe_flag=True,
                     keep_aspect_ratio=False, keep_original_size=False):
    """
    Full preprocessing pipeline for retinal images
    
    Args:
        image_path: Path to input image
        fov_path: Path to FOV mask (optional)
        target_size: Target size for resizing
        use_green_channel: If True, extract green channel; otherwise use 'a' channel from LAB
        apply_clahe_flag: Whether to apply CLAHE
        keep_aspect_ratio: Whether to maintain aspect ratio when resizing
        keep_original_size: If True, don't resize the image
        
    Returns:
        Preprocessed image normalized to [0,1]
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Crop to FOV if mask is provided
    if fov_path is not None:
        if fov_path.endswith('.gif'):
            # Use PIL for GIF files
            fov_pil = Image.open(fov_path)
            fov_mask = np.array(fov_pil)
        else:
            fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
        
        image = crop_to_fov(image, fov_mask)
    
    # Extract channel with best vessel contrast
    if use_green_channel:
        # Extract green channel
        processed = extract_green_channel(image)
    else:
        # Use 'a' channel from LAB
        processed = convert_to_lab(image)
    
    # Apply CLAHE
    if apply_clahe_flag:
        processed = apply_clahe(processed)
    
    # Resize if needed
    if not keep_original_size:
        processed = resize_image(processed, target_size, keep_aspect_ratio)
    
    # Normalize to [0,1]
    processed = normalize_image(processed)
    
    return processed


def visualize_preprocessing_steps(image_path, fov_path, mask_path=None):
    """
    Visualize all preprocessing steps for a single image
    
    Args:
        image_path: Path to input image
        fov_path: Path to FOV mask
        mask_path: Path to vessel mask (optional)
    """
    # Create output directory
    os.makedirs("data/visualizations", exist_ok=True)
    
    # Load image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Load FOV mask
    if fov_path.endswith('.gif'):
        fov_pil = Image.open(fov_path)
        fov_mask = np.array(fov_pil)
    else:
        fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
    
    # Load vessel mask if provided
    if mask_path is not None:
        if mask_path.endswith('.gif'):
            mask_pil = Image.open(mask_path)
            vessel_mask = np.array(mask_pil)
        else:
            vessel_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        vessel_mask = None
    
    # Step 1: Crop to FOV
    cropped = crop_to_fov(original, fov_mask)
    
    # Step 2a: Extract green channel
    green_channel = extract_green_channel(cropped)
    
    # Step 2b: Extract 'a' channel from LAB
    a_channel = convert_to_lab(cropped)
    
    # Step 3a: Apply CLAHE to green channel
    green_clahe = apply_clahe(green_channel)
    
    # Step 3b: Apply CLAHE to 'a' channel
    a_clahe = apply_clahe(a_channel)
    
    # Step 4: Resize
    resized = resize_image(green_clahe, target_size=(512, 512))
    
    # Step 5: Normalize
    normalized = normalize_image(resized)
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(original)
    plt.title("Original RGB Image")
    
    # FOV mask
    plt.subplot(3, 3, 2)
    plt.imshow(fov_mask, cmap='gray')
    plt.title("FOV Mask")
    
    # Cropped image
    plt.subplot(3, 3, 3)
    plt.imshow(cropped)
    plt.title("Cropped to FOV")
    
    # Green channel
    plt.subplot(3, 3, 4)
    plt.imshow(green_channel, cmap='gray')
    plt.title("Green Channel")
    
    # 'a' channel
    plt.subplot(3, 3, 5)
    plt.imshow(a_channel, cmap='gray')
    plt.title("'a' Channel (LAB)")
    
    # Green + CLAHE
    plt.subplot(3, 3, 6)
    plt.imshow(green_clahe, cmap='gray')
    plt.title("Green + CLAHE")
    
    # 'a' + CLAHE
    plt.subplot(3, 3, 7)
    plt.imshow(a_clahe, cmap='gray')
    plt.title("'a' + CLAHE")
    
    # Resized
    plt.subplot(3, 3, 8)
    plt.imshow(resized, cmap='gray')
    plt.title("Resized (512×512)")
    
    # Vessel mask or normalized
    if vessel_mask is not None:
        plt.subplot(3, 3, 9)
        plt.imshow(vessel_mask, cmap='gray')
        plt.title("Vessel Ground Truth")
    else:
        plt.subplot(3, 3, 9)
        plt.imshow(normalized, cmap='gray')
        plt.title("Normalized [0,1]")
    
    plt.tight_layout()
    plt.savefig(f"data/visualizations/preprocessing_steps.png", dpi=150)
    
    print("Visualization saved to 'data/visualizations/preprocessing_steps.png'")


if __name__ == "__main__":
    # Example usage
    image_path = "DRIVE/training/images/21_training.tif"
    fov_path = "DRIVE/training/mask/21_training_mask.gif"
    mask_path = "DRIVE/training/1st_manual/21_manual1.gif"
    
    # Visualize preprocessing steps
    visualize_preprocessing_steps(image_path, fov_path, mask_path)
    
    # Process image with different settings
    processed = preprocess_image(
        image_path=image_path,
        fov_path=fov_path,
        target_size=(512, 512),
        use_green_channel=True,
        apply_clahe_flag=True
    )
    
    # Save processed image visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(processed, cmap='gray')
    plt.title("Final Preprocessed Image")
    plt.axis('off')
    plt.savefig("data/visualizations/final_preprocessed.png", dpi=150)
    print("Final preprocessed image saved to 'data/visualizations/final_preprocessed.png'") 