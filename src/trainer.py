import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAUROC
import wandb
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import matplotlib.pyplot as plt
from losses import CombinedLoss
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from optimizers import Lion, AdamP, Lookahead

class VesselSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for training retinal vessel segmentation models
    """
    def __init__(self, 
                 model,
                 lr=2e-4,
                 weight_decay=1e-4,
                 dice_weight=0.3,
                 tversky_weight=0.4,
                 topology_weight=0.3,
                 tversky_alpha=0.3,
                 tversky_beta=0.7,
                 optimizer_type='lion',  # 'lion', 'adamp', or 'adamw'
                 scheduler_type='one_cycle',  # 'one_cycle' or 'cosine'
                 one_cycle_pct_start=0.3,
                 one_cycle_div_factor=25,
                 one_cycle_max_lr=None,  # If None, use lr*3
                 cosine_t_max=40,
                 use_lookahead=True,
                 freeze_backbone_epochs=10,
                 accumulate_grad_batches=1,
                 log_images=True):
        """
        Args:
            model: Model to be trained
            lr: Initial learning rate
            weight_decay: Weight decay for optimizer
            dice_weight: Weight for Dice loss component
            tversky_weight: Weight for Tversky loss component
            topology_weight: Weight for topology-aware loss component
            tversky_alpha: Alpha parameter for Tversky loss
            tversky_beta: Beta parameter for Tversky loss
            optimizer_type: Type of optimizer to use ('lion', 'adamp', or 'adamw')
            scheduler_type: Type of learning rate scheduler ('one_cycle' or 'cosine')
            one_cycle_pct_start: Percentage of cycle spent increasing learning rate
            one_cycle_div_factor: Initial learning rate divisor
            one_cycle_max_lr: Maximum learning rate (defaults to 3*lr)
            cosine_t_max: Number of epochs for cosine annealing cycle
            use_lookahead: Whether to wrap optimizer with Lookahead
            freeze_backbone_epochs: Number of epochs to keep backbone frozen
            accumulate_grad_batches: Number of batches to accumulate gradients over
            log_images: Whether to log images to wandb
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.one_cycle_pct_start = one_cycle_pct_start
        self.one_cycle_div_factor = one_cycle_div_factor
        self.one_cycle_max_lr = one_cycle_max_lr if one_cycle_max_lr is not None else 3*lr
        self.cosine_t_max = cosine_t_max
        self.use_lookahead = use_lookahead
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_images = log_images
        
        # Initialize loss function
        self.loss_fn = CombinedLoss(
            dice_weight=dice_weight,
            tversky_weight=tversky_weight,
            topology_weight=topology_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta
        )
        
        # Initialize metrics
        self.dice_metric = BinaryF1Score(threshold=0.5)
        self.precision_metric = BinaryPrecision(threshold=0.5)
        self.recall_metric = BinaryRecall(threshold=0.5)
        self.auroc_metric = BinaryAUROC()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Freeze backbone if needed
        if self.freeze_backbone_epochs > 0:
            self._freeze_backbone()
            
        # For gradient accumulation
        self.automatic_optimization = False
    
    def _freeze_backbone(self):
        """Freeze Swin Transformer backbone"""
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
                
    def _unfreeze_backbone(self):
        """Unfreeze Swin Transformer backbone"""
        for param in self.model.parameters():
            param.requires_grad = True
            
    def on_epoch_start(self):
        """Check if backbone should be unfrozen"""
        if self.current_epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
            self._unfreeze_backbone()
            self.log('unfrozen_backbone', 1.0)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay and learning rate scheduler"""
        # Create optimizer based on type
        if self.optimizer_type == 'lion':
            base_optimizer = Lion(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.99)
            )
        elif self.optimizer_type == 'adamp':
            base_optimizer = AdamP(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:  # Default to AdamW
            base_optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            
        # Wrap with Lookahead if requested (only for automatic optimization)
        # Don't use Lookahead with manual optimization due to compatibility issues
        if self.use_lookahead and self.automatic_optimization is False:
            print("Warning: Lookahead is disabled with manual optimization due to compatibility issues")
            optimizer = base_optimizer
        elif self.use_lookahead:
            optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        else:
            optimizer = base_optimizer
            
        # Set up learning rate scheduler
        if self.scheduler_type == 'one_cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.one_cycle_max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.one_cycle_pct_start,
                div_factor=self.one_cycle_div_factor,
                final_div_factor=1e4,
                three_phase=False
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        else:  # Cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.cosine_t_max,
                eta_min=1e-6
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with MixUp/CutMix augmentation for masks and gradient accumulation"""
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        # Get data
        images = batch['image']
        masks = batch['mask']
        fov_masks = batch['fov'] 
        
        # Apply MixUp/CutMix augmentation to masks with 30% probability (during training only)
        if self.current_epoch > 5 and np.random.random() < 0.3:  # Start after 5 epochs
            from augmentation import apply_mask_augmentation
            
            # Convert masks to numpy for augmentation
            masks_np = masks.cpu().numpy()
            
            # Apply mask augmentation
            aug_masks_np = apply_mask_augmentation(masks_np, prob=0.3)
            
            # Convert back to tensor
            masks = torch.from_numpy(aug_masks_np).to(masks.device)
            
            # Apply FOV mask again if available
            if fov_masks is not None:
                masks = masks * fov_masks
        
        # Forward pass
        outputs = self(images)
        
        # For deep supervision, outputs is a list
        if isinstance(outputs, list):
            main_output = outputs[0]  # Main output (full scale)
            
            # Calculate loss for the main output
            main_loss = self.loss_fn(main_output, masks)
            
            # Calculate losses for auxiliary outputs (lower scales)
            aux_losses = []
            for i, aux_output in enumerate(outputs[1:]):
                # Resize mask to match auxiliary output size
                scale_factor = aux_output.shape[2] / masks.shape[2]
                if scale_factor < 1.0:
                    aux_mask = nn.functional.interpolate(
                        masks, 
                        scale_factor=scale_factor, 
                        mode='nearest'
                    )
                    aux_loss = self.loss_fn(aux_output, aux_mask)
                    aux_losses.append(aux_loss)
            
            # Combine main loss with auxiliary losses
            if aux_losses:
                aux_weight = 0.4  # Weight for auxiliary losses
                total_loss = main_loss + aux_weight * sum(aux_losses) / len(aux_losses)
            else:
                total_loss = main_loss
            
            # Use the main output for metrics
            preds = torch.sigmoid(main_output)
        else:
            # No deep supervision, just single output
            total_loss = self.loss_fn(outputs, masks)
            preds = torch.sigmoid(outputs)
        
        # Manual optimization with gradient accumulation
        # Divide loss by accumulation steps to maintain same gradients
        total_loss = total_loss / self.accumulate_grad_batches
        
        # Calculate metrics
        dice = self.dice_metric(preds, masks.int())
        precision = self.precision_metric(preds, masks.int())
        recall = self.recall_metric(preds, masks.int())
        
        # Apply FOV mask to predictions for proper vessel evaluation
        if fov_masks is not None:
            preds_fov = preds * fov_masks
            masks_fov = masks * fov_masks
            
            # Recalculate metrics with FOV masking
            dice_fov = self.dice_metric(preds_fov, masks_fov.int())
            precision_fov = self.precision_metric(preds_fov, masks_fov.int())
            recall_fov = self.recall_metric(preds_fov, masks_fov.int())
            
            # Log metrics with FOV masking
            self.log('train/dice_fov', dice_fov, prog_bar=True)
            self.log('train/precision_fov', precision_fov)
            self.log('train/recall_fov', recall_fov)
        
        # Log metrics
        self.log('train/loss', total_loss * self.accumulate_grad_batches)  # Log the full loss
        self.log('train/dice', dice, prog_bar=True)
        self.log('train/precision', precision)
        self.log('train/recall', recall)
        
        # Backward pass
        self.manual_backward(total_loss)
        
        # Update weights every accumulate_grad_batches
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            # Manual gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            opt.step()
            opt.zero_grad()
            
            # Update learning rate
            if self.scheduler_type == 'one_cycle':
                sch.step()
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images = batch['image']
        masks = batch['mask']
        fov_masks = batch['fov']
        image_names = batch['image_name']
        
        # Forward pass
        outputs = self(images)
        
        # Handle deep supervision outputs
        if isinstance(outputs, list):
            outputs = outputs[0]  # Use main output for validation
        
        # Calculate loss
        loss = self.loss_fn(outputs, masks)
        
        # Get predictions
        preds = torch.sigmoid(outputs)
        
        # Apply FOV mask if available
        if fov_masks is not None:
            preds_fov = preds * fov_masks
            masks_fov = masks * fov_masks
        else:
            preds_fov = preds
            masks_fov = masks
        
        # Calculate metrics
        dice = self.dice_metric(preds_fov, masks_fov.int())
        precision = self.precision_metric(preds_fov, masks_fov.int())
        recall = self.recall_metric(preds_fov, masks_fov.int())
        auroc = self.auroc_metric(preds_fov.reshape(-1), masks_fov.reshape(-1).int())
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/dice', dice, prog_bar=True, sync_dist=True)
        self.log('val/precision', precision, sync_dist=True)
        self.log('val/recall', recall, prog_bar=True, sync_dist=True)
        self.log('val/auroc', auroc, sync_dist=True)
        
        # Log images (first batch only)
        if batch_idx == 0 and self.log_images:
            self._log_images(images, masks, preds, fov_masks, image_names)
        
        return {
            'val_loss': loss,
            'val_dice': dice,
            'val_precision': precision,
            'val_recall': recall,
            'val_auroc': auroc
        }
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        # Same as validation step
        return self.validation_step(batch, batch_idx)
    
    def _log_images(self, images, masks, preds, fov_masks, image_names, num_images=4):
        """Log images to wandb"""
        if not self.logger or not isinstance(self.logger, WandbLogger):
            return
        
        num_images = min(num_images, len(images))
        fig, axs = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
        
        for i in range(num_images):
            # Original image (green channel)
            img = images[i].cpu().numpy()
            if img.shape[0] == 1:  # Single channel
                axs[i, 0].imshow(img[0], cmap='gray')
            else:  # RGB - take green channel
                axs[i, 0].imshow(img[1], cmap='gray')
            axs[i, 0].set_title(f"Image: {image_names[i]}")
            axs[i, 0].axis('off')
            
            # Ground truth mask
            mask = masks[i, 0].cpu().numpy()
            axs[i, 1].imshow(mask, cmap='gray')
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis('off')
            
            # Prediction
            pred = preds[i, 0].cpu().detach().numpy()
            axs[i, 2].imshow(pred, cmap='gray')
            axs[i, 2].set_title(f"Prediction")
            axs[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Log to wandb
        self.logger.experiment.log({
            f"val_images_epoch_{self.current_epoch}": wandb.Image(fig)
        })
        
        plt.close(fig)

def train_model(model, 
                train_dataloader, 
                val_dataloader,
                test_dataloader=None,
                lr=2e-4,
                weight_decay=1e-4,
                dice_weight=0.3,
                tversky_weight=0.4,
                topology_weight=0.3,
                tversky_alpha=0.3,
                tversky_beta=0.7,
                optimizer_type='lion',
                scheduler_type='one_cycle',
                one_cycle_pct_start=0.3,
                one_cycle_div_factor=25,
                one_cycle_max_lr=None,
                cosine_t_max=40,
                use_lookahead=True,
                freeze_backbone_epochs=10,
                max_epochs=400,
                early_stopping_patience=30,
                use_swa=True,
                swa_start_epoch=300,
                swa_freq=5,
                swa_lr=1e-4,
                accumulate_grad_batches=4,
                ckpt_path='checkpoints',
                log_images=True,
                wandb_project="drive-vessel-f1-85",
                use_wandb=True,
                use_amp=True,
                seed=42,
                loss_params=None):
    """
    Train a vessel segmentation model
    
    Args:
        model: The model to train
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        test_dataloader: Test dataloader (optional)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        dice_weight: Weight for Dice loss component
        tversky_weight: Weight for Tversky loss component
        topology_weight: Weight for topology-aware loss component
        tversky_alpha: Alpha parameter for Tversky loss
        tversky_beta: Beta parameter for Tversky loss
        optimizer_type: Type of optimizer to use ('lion', 'adamp', or 'adamw')
        scheduler_type: Type of learning rate scheduler ('one_cycle' or 'cosine')
        one_cycle_pct_start: Percentage of cycle spent increasing learning rate
        one_cycle_div_factor: Initial learning rate divisor
        one_cycle_max_lr: Maximum learning rate (defaults to 3*lr)
        cosine_t_max: Number of epochs for cosine annealing cycle
        use_lookahead: Whether to wrap optimizer with Lookahead
        freeze_backbone_epochs: Number of epochs to keep backbone frozen
        max_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        use_swa: Whether to use Stochastic Weight Averaging
        swa_start_epoch: Epoch to start SWA
        swa_freq: Frequency of SWA updates (in epochs)
        swa_lr: Learning rate for SWA
        accumulate_grad_batches: Number of batches to accumulate gradients over
        ckpt_path: Path to save checkpoints
        log_images: Whether to log images to wandb
        wandb_project: WandB project name
        use_wandb: Whether to use wandb for logging
        use_amp: Whether to use automatic mixed precision
        seed: Random seed for reproducibility
        loss_params: Dictionary with loss function parameters
    
    Returns:
        Trained Lightning module
    """
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    # Default loss parameters
    default_loss_params = {
        'dice_weight': 0.3,
        'tversky_weight': 0.4,
        'topology_weight': 0.3,
        'tversky_alpha': 0.3,
        'tversky_beta': 0.7
    }
    
    # Use provided loss parameters or defaults
    if loss_params is None:
        loss_params = default_loss_params
    
    # Create lightning module
    model_module = VesselSegmentationModule(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        dice_weight=loss_params.get('dice_weight', default_loss_params['dice_weight']),
        tversky_weight=loss_params.get('tversky_weight', default_loss_params['tversky_weight']),
        topology_weight=loss_params.get('topology_weight', default_loss_params['topology_weight']),
        tversky_alpha=loss_params.get('tversky_alpha', default_loss_params['tversky_alpha']),
        tversky_beta=loss_params.get('tversky_beta', default_loss_params['tversky_beta']),
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type, 
        one_cycle_pct_start=one_cycle_pct_start,
        one_cycle_div_factor=one_cycle_div_factor,
        one_cycle_max_lr=one_cycle_max_lr,
        cosine_t_max=cosine_t_max,
        use_lookahead=use_lookahead,
        freeze_backbone_epochs=freeze_backbone_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        log_images=log_images
    )
    
    # Create logger
    if use_wandb:
        logger = WandbLogger(project=wandb_project, log_model=True)
    else:
        logger = None
    
    # Create callbacks
    callbacks = [
        # Model checkpoint based on validation dice score
        ModelCheckpoint(
            dirpath=ckpt_path,
            filename='{epoch}-{val/dice:.4f}',
            monitor='val/dice',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        # Early stopping based on validation dice score
        EarlyStopping(
            monitor='val/dice',
            patience=early_stopping_patience,
            mode='max',
            verbose=True
        )
    ]
    
    # Add Stochastic Weight Averaging if enabled
    if use_swa:
        # Add SWA callback
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=swa_start_epoch,
                swa_lrs=swa_lr,
                annealing_epochs=5
            )
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Use GPU if available
        devices='auto',
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed' if use_amp else '32',
        accumulate_grad_batches=1,  # We handle this manually now
        deterministic=False,  # For better performance
        log_every_n_steps=1  # Log every step for small datasets
    )
    
    # Train model
    trainer.fit(model_module, train_dataloader, val_dataloader)
    
    # Test model if test dataloader is provided
    if test_dataloader:
        trainer.test(model_module, test_dataloader, ckpt_path='best')
    
    return model_module 