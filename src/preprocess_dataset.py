import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocessing import preprocess_image
import cv2


def preprocess_dataset(data_csv, output_dir, resize=(512, 512), use_green=True):
    """
    Preprocess all images in the dataset and save them to the output directory
    
    Args:
        data_csv: Path to CSV file with dataset information
        output_dir: Directory to save preprocessed images
        resize: Target size for resizing
        use_green: Whether to use green channel (True) or 'a' channel from LAB (False)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # Load dataset information
    df = pd.read_csv(data_csv)
    
    # Process each image
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
        # Get file paths
        image_path = row['image_path']
        mask_path = row['mask_path']
        fov_path = row['fov_path']
        image_id = row['id']
        
        # Preprocess image
        processed_image = preprocess_image(
            image_path=image_path,
            fov_path=fov_path,
            target_size=resize,
            use_green_channel=use_green,
            apply_clahe_flag=True,
            keep_aspect_ratio=False
        )
        
        # Save preprocessed image (as numpy array for later use)
        np.save(
            os.path.join(output_dir, 'images', f"{image_id}_preprocessed.npy"),
            processed_image
        )
        
        # Also save as PNG for visual inspection
        plt.imsave(
            os.path.join(output_dir, 'images', f"{image_id}_preprocessed.png"),
            processed_image,
            cmap='gray'
        )
        
        # Process mask (just resize and convert to binary)
        # Use OpenCV for masks
        from PIL import Image
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil)
        
        # Convert to binary (0 or 1)
        binary_mask = (mask > 0).astype(np.float32)
        
        # Resize mask to match the processed image
        resized_mask = cv2.resize(
            binary_mask, 
            resize, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Save preprocessed mask
        np.save(
            os.path.join(output_dir, 'masks', f"{image_id}_mask.npy"),
            resized_mask
        )
        
        # Also save as PNG for visual inspection
        plt.imsave(
            os.path.join(output_dir, 'masks', f"{image_id}_mask.png"),
            resized_mask,
            cmap='gray'
        )
    
    print(f"Preprocessing complete. Saved to {output_dir}")


if __name__ == "__main__":
    # Preprocess training set
    preprocess_dataset(
        data_csv='data/splits/train_official.csv',
        output_dir='data/preprocessed/train',
        resize=(512, 512),
        use_green=True
    )
    
    # Preprocess test set
    preprocess_dataset(
        data_csv='data/splits/test_official.csv',
        output_dir='data/preprocessed/test',
        resize=(512, 512),
        use_green=True
    )
    
    # Print summary
    train_count = len(os.listdir('data/preprocessed/train/images'))
    test_count = len(os.listdir('data/preprocessed/test/images'))
    print(f"Preprocessed {train_count // 2} training images and {test_count // 2} test images")
    print("(Each image is saved as both .npy and .png)")
    
    # Display sample
    print("\nDisplaying a sample preprocessed image and mask...")
    sample_img = np.load('data/preprocessed/train/images/21_preprocessed.npy')
    sample_mask = np.load('data/preprocessed/train/masks/21_mask.npy')
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(sample_img, cmap='gray')
    plt.title("Preprocessed Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(sample_mask, cmap='gray')
    plt.title("Vessel Mask")
    
    plt.subplot(1, 3, 3)
    # Overlay: red vessels on green channel
    overlay = np.zeros((sample_img.shape[0], sample_img.shape[1], 3))
    overlay[:, :, 0] = sample_mask  # Red for vessels
    overlay[:, :, 1] = sample_img   # Green for background
    plt.imshow(overlay)
    plt.title("Vessel Overlay")
    
    plt.tight_layout()
    plt.savefig("data/visualizations/sample_preprocessed_dataset.png", dpi=150)
    print("Sample visualization saved to 'data/visualizations/sample_preprocessed_dataset.png'") 