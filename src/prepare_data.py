import os
import shutil
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def create_symlinks():
    """
    Create symbolic links to the original DRIVE dataset files to maintain 
    the original data structure but in our project organization
    """
    # Define source and destination directories
    drive_src = 'DRIVE'
    drive_dest = 'data/DRIVE'
    
    # Create directories if they don't exist
    os.makedirs(drive_dest, exist_ok=True)
    
    # Create symbolic links for the training and test directories
    for split in ['training', 'test']:
        os.makedirs(os.path.join(drive_dest, split), exist_ok=True)
        
        # Copy each subdirectory
        for subdir in ['images', '1st_manual', 'mask']:
            src_dir = os.path.join(drive_src, split, subdir)
            dest_dir = os.path.join(drive_dest, split, subdir)
            
            # Create the destination directory
            os.makedirs(dest_dir, exist_ok=True)
            
            # Create symbolic links for each file
            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                dest_file = os.path.join(dest_dir, file)
                
                # Create a symbolic link
                if not os.path.exists(dest_file):
                    os.symlink(os.path.abspath(src_file), dest_file)
    
    print("Symbolic links created successfully!")

def create_official_split_csv():
    """
    Create CSV files for the official 20/20 split
    """
    train_df = []
    test_df = []
    
    # Process training set
    train_image_dir = 'data/DRIVE/training/images'
    train_mask_dir = 'data/DRIVE/training/1st_manual'
    train_fov_dir = 'data/DRIVE/training/mask'
    
    for img_file in os.listdir(train_image_dir):
        if not img_file.endswith('.tif'):
            continue
            
        img_id = img_file.split('_')[0]
        img_path = os.path.join(train_image_dir, img_file)
        
        # Find corresponding mask and FOV files
        mask_file = f"{img_id}_manual1.gif"
        mask_path = os.path.join(train_mask_dir, mask_file)
        
        fov_file = f"{img_id}_training_mask.gif"
        fov_path = os.path.join(train_fov_dir, fov_file)
        
        if os.path.exists(mask_path) and os.path.exists(fov_path):
            train_df.append({
                'id': img_id,
                'image_path': img_path,
                'mask_path': mask_path,
                'fov_path': fov_path,
                'split': 'train'
            })
    
    # Process test set
    test_image_dir = 'data/DRIVE/test/images'
    test_mask_dir = 'data/DRIVE/test/1st_manual'
    test_fov_dir = 'data/DRIVE/test/mask'
    
    for img_file in os.listdir(test_image_dir):
        if not img_file.endswith('.tif'):
            continue
            
        img_id = img_file.split('_')[0]
        img_path = os.path.join(test_image_dir, img_file)
        
        # Find corresponding mask and FOV files
        mask_file = f"{img_id}_manual1.gif"
        mask_path = os.path.join(test_mask_dir, mask_file)
        
        fov_file = f"{img_id}_test_mask.gif"
        fov_path = os.path.join(test_fov_dir, fov_file)
        
        if os.path.exists(mask_path) and os.path.exists(fov_path):
            test_df.append({
                'id': img_id,
                'image_path': img_path,
                'mask_path': mask_path,
                'fov_path': fov_path,
                'split': 'test'
            })
    
    # Create dataframes and save to CSV
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    
    # Save official split
    os.makedirs('data/splits', exist_ok=True)
    train_df.to_csv('data/splits/train_official.csv', index=False)
    test_df.to_csv('data/splits/test_official.csv', index=False)
    
    # Combine for overall dataset
    all_df = pd.concat([train_df, test_df])
    all_df.to_csv('data/splits/all_data.csv', index=False)
    
    print(f"Official split created with {len(train_df)} training and {len(test_df)} test images.")
    return train_df, test_df

def create_cv_folds(train_df, n_folds=5, seed=42):
    """
    Create n-fold CV splits from the training set
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Get training data and shuffle
    train_data = train_df.copy()
    
    # Define KFold cross validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Create directory for CV splits if it doesn't exist
    os.makedirs('data/splits/cv', exist_ok=True)
    
    # Generate folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        # Split into train and validation for this fold
        fold_train = train_data.iloc[train_idx].copy()
        fold_val = train_data.iloc[val_idx].copy()
        
        # Add fold number
        fold_train['fold'] = fold
        fold_val['fold'] = fold
        
        # Set split type
        fold_train['split'] = 'train'
        fold_val['split'] = 'val'
        
        # Save to CSV
        fold_train.to_csv(f'data/splits/cv/fold_{fold}_train.csv', index=False)
        fold_val.to_csv(f'data/splits/cv/fold_{fold}_val.csv', index=False)
        
        # Also create a combined CSV for convenience
        fold_combined = pd.concat([fold_train, fold_val])
        fold_combined.to_csv(f'data/splits/cv/fold_{fold}.csv', index=False)
        
        print(f"Fold {fold}: {len(fold_train)} training, {len(fold_val)} validation images")

if __name__ == "__main__":
    print("Preparing DRIVE dataset...")
    
    # Create symbolic links to original data
    create_symlinks()
    
    # Create official train/test split CSVs
    train_df, test_df = create_official_split_csv()
    
    # Create 5-fold CV splits from training data
    create_cv_folds(train_df, n_folds=5, seed=42)
    
    print("Data preparation complete!") 