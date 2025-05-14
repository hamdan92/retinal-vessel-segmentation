import os
import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from tqdm import tqdm
import wandb

# Import our modules
from model_wrapper import SwinResNet
from dataset import VesselSegmentationDataset
from augmentation import get_val_transforms
from postprocessing import process_batch_predictions, apply_test_time_augmentation
from trainer import VesselSegmentationModule

def visualize_results(images, masks, preds, processed_preds, image_names, output_dir, num_samples=5):
    """
    Visualize and save segmentation results
    
    Args:
        images: Batch of input images
        masks: Batch of ground truth masks
        preds: Batch of raw predictions
        processed_preds: Batch of post-processed predictions
        image_names: Names of the images
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create output directory if it doesn't exist
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Limit number of samples
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get data for current sample
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:  # Single channel
            img = img[0]
        else:  # RGB - use green channel
            img = img[1]
        
        mask = masks[i, 0].cpu().numpy()
        pred = preds[i, 0].cpu().numpy()
        proc_pred = processed_preds[i]
        
        # Plot original image
        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title(f"Original Image: {image_names[i]}")
        axs[0, 0].axis('off')
        
        # Plot ground truth mask
        axs[0, 1].imshow(mask, cmap='gray')
        axs[0, 1].set_title("Ground Truth Mask")
        axs[0, 1].axis('off')
        
        # Plot raw prediction
        axs[1, 0].imshow(pred, cmap='gray')
        axs[1, 0].set_title("Raw Prediction")
        axs[1, 0].axis('off')
        
        # Plot post-processed prediction
        axs[1, 1].imshow(proc_pred, cmap='gray')
        axs[1, 1].set_title("Post-processed Prediction")
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{image_names[i]}_results.png"), dpi=150)
        plt.close(fig)

def plot_roc_curve(y_true, y_pred, output_dir):
    """
    Plot and save ROC curve
    
    Args:
        y_true: Ground truth labels (flattened)
        y_pred: Predicted probabilities (flattened)
        output_dir: Directory to save plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    
    return roc_auc

def plot_pr_curve(y_true, y_pred, output_dir):
    """
    Plot and save precision-recall curve
    
    Args:
        y_true: Ground truth labels (flattened)
        y_pred: Predicted probabilities (flattened)
        output_dir: Directory to save plot
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
    plt.close()
    
    return average_precision

def evaluate_with_tta(model, dataloader, device, fov_masks=None, postprocess_params=None):
    """
    Evaluate model with test-time augmentation (TTA) for improved performance
    
    Args:
        model: PyTorch model
        dataloader: PyTorch dataloader
        device: Device to use for inference
        fov_masks: Optional field of view masks
        postprocess_params: Parameters for postprocessing
        
    Returns:
        Dictionary of evaluation metrics
    """
    import torch
    from tqdm import tqdm
    import numpy as np
    from postprocessing import apply_test_time_augmentation, process_prediction
    from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAUROC
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    metrics = {
        'f1': BinaryF1Score().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'auroc': BinaryAUROC().to(device)
    }
    
    # Default postprocessing parameters
    if postprocess_params is None:
        postprocess_params = {
            'min_size': 30,
            'apply_morphology': True,
            'kernel_size': 3,
            'enhance_vessels': True,
            'threshold_sensitivity': 0.45
        }
    
    # Store predictions and targets for computing overall metrics
    all_preds = []
    all_masks = []
    all_fovs = []
    
    # Process each batch
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating with TTA"):
            # Extract data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Get FOV masks if available
            fov_mask = None
            if 'fov' in batch:
                fov_mask = batch['fov']
                all_fovs.append(fov_mask.cpu().numpy())
            
            # Apply test-time augmentation
            ensemble_pred = apply_test_time_augmentation(model, images, device)
            
            # Store predictions and targets
            all_preds.append(ensemble_pred)
            all_masks.append(masks.cpu().numpy())
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Concatenate FOV masks if available
    if len(all_fovs) > 0:
        all_fovs = np.concatenate(all_fovs, axis=0)
    else:
        all_fovs = None
    
    # Apply postprocessing to predictions
    processed_preds = process_batch_predictions(
        all_preds, 
        fov_masks=all_fovs, 
        **postprocess_params
    )
    
    # Convert to tensors for metric calculation
    processed_preds_tensor = torch.from_numpy(processed_preds).float().to(device)
    masks_tensor = torch.from_numpy(all_masks).float().to(device)
    
    # Calculate metrics
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(processed_preds_tensor, masks_tensor).item()
    
    return results

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main evaluation function
    
    Args:
        cfg: Hydra configuration
    """
    # Set random seed for reproducibility
    pl.seed_everything(cfg.training.seed)
    
    # Create output directory if it doesn't exist
    output_dir = cfg.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model checkpoint
    checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, "last.ckpt")
    
    # Check for best checkpoint
    best_checkpoint = os.path.join(cfg.paths.checkpoint_dir, "best.ckpt")
    if os.path.exists(best_checkpoint):
        checkpoint_path = best_checkpoint
    else:
        # Check for checkpoint with highest validation dice
        checkpoint_files = [f for f in os.listdir(cfg.paths.checkpoint_dir) 
                          if f.endswith(".ckpt") and f != "last.ckpt"]
        if checkpoint_files:
            # Extract validation dice from filename
            dice_scores = [float(f.split("-")[-1].split(".ckpt")[0]) for f in checkpoint_files]
            best_idx = np.argmax(dice_scores)
            checkpoint_path = os.path.join(cfg.paths.checkpoint_dir, checkpoint_files[best_idx])
    
    print(f"Loading model checkpoint: {checkpoint_path}")
    
    # Create model
    model = SwinResNet(
        input_channels=cfg.model.in_channels,
        num_classes=cfg.model.out_channels,
        output_size=cfg.model.img_size,
        use_swin=False,  # No need for pretrained weights at inference
        encoder_name=cfg.model.backbone,
        pretrained=False
    )
    
    # Load model weights
    model_module = VesselSegmentationModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu"
    )
    model_module.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_module.to(device)
    
    # Create test dataset
    test_transforms = get_val_transforms(img_size=cfg.data.img_size)
    
    # Get test CSV file
    test_csv = os.path.join(cfg.paths.csv_dir, "test.csv")
    
    test_dataset = VesselSegmentationDataset(
        csv_file=test_csv,
        image_dir=cfg.paths.preprocessed_image_dir,
        mask_dir=cfg.paths.mask_dir,
        fov_dir=cfg.paths.fov_dir,
        transform=test_transforms,
        phase='test',
        preprocess=cfg.data.use_preprocessed,
        img_size=cfg.data.img_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize metrics
    metrics = {
        'dice': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'accuracy': []
    }
    
    all_preds = []
    all_masks = []
    all_images = []
    all_image_names = []
    all_processed_preds = []
    
    # Run evaluation
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            fov_masks = batch['fov'].to(device)
            image_names = batch['image_name']
            
            # Apply test-time augmentation
            if cfg.experiment.get('use_tta', False):
                preds = apply_test_time_augmentation(model_module.model, images, device)
            else:
                # Forward pass
                outputs = model_module.model(images)
                
                # Handle deep supervision
                if isinstance(outputs, list):
                    outputs = outputs[0]
                    
                # Apply sigmoid
                preds = torch.sigmoid(outputs)
            
            # Apply FOV mask
            if fov_masks is not None:
                preds_masked = preds * fov_masks
                masks_masked = masks * fov_masks
            else:
                preds_masked = preds
                masks_masked = masks
            
            # Post-process predictions
            processed_preds = process_batch_predictions(
                preds_masked.cpu().numpy(),
                fov_masks=fov_masks.cpu().numpy(),
                min_size=cfg.postprocessing.min_object_size,
                apply_morphology=cfg.postprocessing.apply_morphology,
                kernel_size=cfg.postprocessing.kernel_size,
                skeletonize_result=cfg.postprocessing.skeletonize
            )
            
            # Convert to binary for metrics (threshold=0.5)
            preds_binary = (preds_masked > 0.5).float()
            
            # Calculate metrics for batch
            # Dice/F1
            intersection = (preds_binary * masks_masked).sum(dim=[1, 2, 3])
            union = preds_binary.sum(dim=[1, 2, 3]) + masks_masked.sum(dim=[1, 2, 3])
            dice = (2. * intersection) / (union + 1e-8)
            
            # Precision
            precision = intersection / (preds_binary.sum(dim=[1, 2, 3]) + 1e-8)
            
            # Recall/Sensitivity
            recall = intersection / (masks_masked.sum(dim=[1, 2, 3]) + 1e-8)
            
            # Specificity
            true_negatives = ((1 - preds_binary) * (1 - masks_masked)).sum(dim=[1, 2, 3])
            false_positives = (preds_binary * (1 - masks_masked)).sum(dim=[1, 2, 3])
            specificity = true_negatives / (true_negatives + false_positives + 1e-8)
            
            # Accuracy
            total_pixels = preds_binary.shape[1] * preds_binary.shape[2] * preds_binary.shape[3]
            correct = ((preds_binary == masks_masked).sum(dim=[1, 2, 3])).float()
            accuracy = correct / total_pixels
            
            # Add batch metrics to overall metrics
            metrics['dice'].extend(dice.cpu().numpy())
            metrics['precision'].extend(precision.cpu().numpy())
            metrics['recall'].extend(recall.cpu().numpy())
            metrics['specificity'].extend(specificity.cpu().numpy())
            metrics['accuracy'].extend(accuracy.cpu().numpy())
            
            # Store predictions and masks for ROC and PR curves
            all_preds.append(preds_masked.cpu())
            all_masks.append(masks_masked.cpu())
            all_images.append(images.cpu())
            all_image_names.extend(image_names)
            all_processed_preds.append(torch.from_numpy(processed_preds).float())
    
    # Calculate mean metrics
    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    # Print metrics
    print("\nTest Metrics:")
    for metric, value in mean_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'image_name': all_image_names,
        'dice': metrics['dice'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'specificity': metrics['specificity'],
        'accuracy': metrics['accuracy']
    })
    metrics_df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)
    
    # Save mean metrics
    with open(os.path.join(output_dir, "mean_metrics.txt"), "w") as f:
        for metric, value in mean_metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f} ± {std_metrics[metric]:.4f}\n")
    
    # Concatenate all predictions and masks
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_processed_preds = torch.cat(all_processed_preds, dim=0)
    
    # Flatten for ROC and PR curves
    flat_preds = all_preds.view(-1).numpy()
    flat_masks = all_masks.view(-1).numpy()
    
    # Calculate and plot ROC curve
    roc_auc = plot_roc_curve(flat_masks, flat_preds, output_dir)
    
    # Calculate and plot PR curve
    average_precision = plot_pr_curve(flat_masks, flat_preds, output_dir)
    
    # Visualize sample results
    visualize_results(all_images, all_masks, all_preds, all_processed_preds, 
                      all_image_names, output_dir)
    
    # Log to wandb if enabled
    if cfg.experiment.get('log_to_wandb', False):
        wandb.init(project=cfg.experiment.wandb_project, name="evaluation")
        
        # Log metrics
        wandb.log({
            "test/dice": mean_metrics['dice'],
            "test/precision": mean_metrics['precision'],
            "test/recall": mean_metrics['recall'],
            "test/specificity": mean_metrics['specificity'],
            "test/accuracy": mean_metrics['accuracy'],
            "test/roc_auc": roc_auc,
            "test/average_precision": average_precision
        })
        
        # Log sample images
        sample_idx = np.random.choice(len(all_images), min(5, len(all_images)), replace=False)
        for i in sample_idx:
            img = all_images[i].numpy()[0]  # Assuming single channel
            mask = all_masks[i].numpy()[0]
            pred = all_preds[i].numpy()[0]
            proc_pred = all_processed_preds[i].numpy()
            
            # Create figure
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs[0, 0].imshow(img, cmap='gray')
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(mask, cmap='gray')
            axs[0, 1].set_title("Ground Truth")
            axs[0, 1].axis('off')
            
            axs[1, 0].imshow(pred, cmap='gray')
            axs[1, 0].set_title("Raw Prediction")
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(proc_pred, cmap='gray')
            axs[1, 1].set_title("Post-processed")
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f"sample_{i}": wandb.Image(fig)})
            plt.close(fig)
        
        wandb.finish()
    
    print(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 