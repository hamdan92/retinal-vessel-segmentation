import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def inspect_image_and_mask():
    """
    Load and display a sample image with its corresponding mask and FOV mask
    """
    # Path to a sample training image (using original paths)
    img_path = "DRIVE/training/images/21_training.tif"
    mask_path = "DRIVE/training/1st_manual/21_manual1.gif"
    fov_path = "DRIVE/training/mask/21_training_mask.gif"
    
    # Read images
    img = cv2.imread(img_path)
    
    # Use PIL for GIF files
    mask_pil = Image.open(mask_path)
    mask = np.array(mask_pil)
    
    fov_pil = Image.open(fov_path)
    fov = np.array(fov_pil)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract green channel
    green_channel = img[:, :, 1]
    
    # Print image information
    print(f"Image shape: {img.shape}")
    print(f"Image type: {img.dtype}")
    print(f"Image min/max values: {img.min()}/{img.max()}")
    print(f"Green channel shape: {green_channel.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask type: {mask.dtype}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"FOV mask shape: {fov.shape}")
    print(f"FOV mask unique values: {np.unique(fov)}")
    
    # Create a side-by-side visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original RGB Image")
    
    plt.subplot(2, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Vessel Mask")
    
    plt.subplot(2, 3, 3)
    plt.imshow(fov, cmap='gray')
    plt.title("FOV Mask")
    
    plt.subplot(2, 3, 4)
    plt.imshow(green_channel, cmap='gray')
    plt.title("Green Channel")
    
    # Apply CLAHE to green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_channel.astype(np.uint8))
    
    plt.subplot(2, 3, 5)
    plt.imshow(green_clahe, cmap='gray')
    plt.title("Green Channel + CLAHE")
    
    # Create overlay of vessels on green channel
    overlay = np.zeros_like(img)
    overlay[:, :, 0] = np.where(mask > 0, 255, 0)  # Red for vessels
    overlay[:, :, 1] = green_channel
    overlay[:, :, 2] = 0
    
    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title("Vessel Overlay on Green Channel")
    
    plt.tight_layout()
    
    # Save the visualization to file
    os.makedirs("data/visualizations", exist_ok=True)
    plt.savefig("data/visualizations/sample_image_inspection.png", dpi=150)
    
    print("Visualization saved to 'data/visualizations/sample_image_inspection.png'")

if __name__ == "__main__":
    inspect_image_and_mask() 