import os
import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from datetime import datetime

# Import our modules
from model_wrapper import SwinResNet
from dataset import VesselSegmentationDataset
from trainer import train_model
from augmentation import get_train_transforms, get_val_transforms

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function using Hydra for configuration
    
    Args:
        cfg: Hydra configuration
    """
    # Set random seed for reproducibility
    pl.seed_everything(cfg.training.seed)
    
    # Create directories if they don't exist
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    # Create model
    model = SwinResNet(
        input_channels=cfg.model.in_channels,
        num_classes=cfg.model.out_channels,
        output_size=cfg.model.img_size,
        use_swin=False,  # Use custom encoder for flexibility with input size
        encoder_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained
    )
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get augmentations
    train_transforms = get_train_transforms(
        img_size=cfg.data.img_size,
        rotate_limit=cfg.augmentation.rotate_limit,
        elastic_sigma=cfg.augmentation.elastic_sigma,
        elastic_alpha=cfg.augmentation.elastic_alpha,
        brightness_limit=cfg.augmentation.brightness_limit,
        contrast_limit=cfg.augmentation.contrast_limit,
        gamma_limit=cfg.augmentation.gamma_limit,
        prob=cfg.augmentation.prob
    )
    
    val_transforms = get_val_transforms(img_size=cfg.data.img_size)
    
    # Create datasets
    if cfg.data.use_cross_validation:
        # Cross-validation training
        current_fold = cfg.data.current_fold
        train_csv = os.path.join(cfg.paths.csv_dir, f"fold_{current_fold}_train.csv")
        val_csv = os.path.join(cfg.paths.csv_dir, f"fold_{current_fold}_val.csv")
        
        train_dataset = VesselSegmentationDataset(
            csv_file=train_csv,
            image_dir=cfg.paths.preprocessed_image_dir,
            mask_dir=cfg.paths.mask_dir,
            fov_dir=cfg.paths.fov_dir,
            transform=train_transforms,
            phase='train',
            preprocess=cfg.data.use_preprocessed,
            img_size=cfg.data.img_size
        )
        
        val_dataset = VesselSegmentationDataset(
            csv_file=val_csv,
            image_dir=cfg.paths.preprocessed_image_dir,
            mask_dir=cfg.paths.mask_dir,
            fov_dir=cfg.paths.fov_dir,
            transform=val_transforms,
            phase='val',
            preprocess=cfg.data.use_preprocessed,
            img_size=cfg.data.img_size
        )
        
        # Update checkpoint path for the current fold
        cfg.paths.checkpoint_dir = os.path.join(cfg.paths.checkpoint_dir, f"fold_{current_fold}")
        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    else:
        # Official train/test split
        train_csv = os.path.join(cfg.paths.csv_dir, "train.csv")
        val_csv = os.path.join(cfg.paths.csv_dir, "test.csv")
        
        train_dataset = VesselSegmentationDataset(
            csv_file=train_csv,
            image_dir=cfg.paths.preprocessed_image_dir,
            mask_dir=cfg.paths.mask_dir,
            fov_dir=cfg.paths.fov_dir,
            transform=train_transforms,
            phase='train',
            preprocess=cfg.data.use_preprocessed,
            img_size=cfg.data.img_size
        )
        
        val_dataset = VesselSegmentationDataset(
            csv_file=val_csv,
            image_dir="data/DRIVE/test/images",  # Use test images for validation
            mask_dir="data/DRIVE/test/1st_manual",  # Use test masks for validation
            fov_dir="data/DRIVE/test/mask",  # Use test FOV masks for validation
            transform=val_transforms,
            phase='test',  # Set phase to test
            preprocess=cfg.data.use_preprocessed,
            img_size=cfg.data.img_size
        )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create a timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{cfg.experiment.name}_{timestamp}"
    
    # Update wandb project name
    wandb_project = f"{cfg.experiment.wandb_project}"
    if cfg.data.use_cross_validation:
        wandb_project = f"{wandb_project}_fold{current_fold}"
    
    # Training parameters from config
    loss_params = {
        'dice_weight': cfg.loss.dice_weight,
        'tversky_weight': cfg.loss.tversky_weight,
        'topology_weight': cfg.loss.topology_weight,
        'tversky_alpha': cfg.loss.tversky_alpha,
        'tversky_beta': cfg.loss.tversky_beta
    }
    
    # Train model
    model_module = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        cosine_t_max=cfg.training.cosine_t_max,
        freeze_backbone_epochs=cfg.training.freeze_backbone_epochs,
        max_epochs=cfg.training.max_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        ckpt_path=cfg.paths.checkpoint_dir,
        log_images=cfg.experiment.log_images,
        wandb_project=wandb_project,
        use_wandb=cfg.experiment.get('use_wandb', True),
        use_amp=cfg.training.use_amp,
        seed=cfg.training.seed,
        loss_params=loss_params
    )
    
    print(f"Training completed. Model checkpoints saved to {cfg.paths.checkpoint_dir}")

if __name__ == "__main__":
    main() 