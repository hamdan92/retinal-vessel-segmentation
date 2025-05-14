import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten prediction and target tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced binary segmentation
    Optimized for thin vessel detection by emphasizing recall
    """
    def __init__(self, alpha=0.5, beta=0.7, gamma=0.75, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Controls weight of false positives
        self.beta = beta    # Controls weight of false negatives (higher = more weight to FN)
        self.gamma = gamma  # Controls focusing effect (lower = more focus on hard examples)
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Apply sigmoid if the input is logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
            
        # Flatten prediction and target tensors
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1) * self.alpha
        fn = ((1 - pred) * target).sum(dim=1) * self.beta
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        # Focal Tversky Loss
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function as specified in the plan:
    TotalLoss = 0.4 ⋅ DiceLoss + 0.4 ⋅ FocalTversky(α=0.5,β=0.7) + 0.2 ⋅ BCE
    Optimized for thin vessel detection
    """
    def __init__(self, dice_weight=0.4, tversky_weight=0.4, bce_weight=0.2, 
                 tversky_alpha=0.5, tversky_beta=0.7, tversky_gamma=0.75, 
                 smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        
        self.dice = DiceLoss(smooth=smooth)
        self.tversky = FocalTverskyLoss(
            alpha=tversky_alpha, 
            beta=tversky_beta, 
            gamma=tversky_gamma, 
            smooth=smooth
        )
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model prediction (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) - binary mask
        """
        # For Dice and Tversky, apply sigmoid to get probabilities
        probs = torch.sigmoid(pred)
        
        dice_loss = self.dice(probs, target)
        tversky_loss = self.tversky(pred, target)  # Will handle sigmoid internally
        
        # For BCE, use logits directly
        bce_loss = self.bce(pred, target)
        
        # Combine losses with weights
        total_loss = (
            self.dice_weight * dice_loss + 
            self.tversky_weight * tversky_loss + 
            self.bce_weight * bce_loss
        )
        
        return total_loss 