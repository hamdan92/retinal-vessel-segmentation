import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Controls weight of false positives (lower to focus more on thin vessels)
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

class TopologyAwareLoss(nn.Module):
    """
    Topology-aware loss component that penalizes disconnected vessels 
    and encourages structural continuity in the prediction
    """
    def __init__(self, kernel_size=5, sigma=1.0, penalty_weight=1.0):
        super(TopologyAwareLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.penalty_weight = penalty_weight
        
        # Create Gaussian kernel for edge detection
        self.register_buffer('kernel', self._create_gaussian_kernel())
        
    def _create_gaussian_kernel(self):
        # Create a Gaussian kernel for edge detection
        x = torch.arange(-(self.kernel_size//2), self.kernel_size//2 + 1)
        y = torch.arange(-(self.kernel_size//2), self.kernel_size//2 + 1)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        
        # Shape the kernel for conv2d: (1, 1, kernel_size, kernel_size)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)
    
    def _extract_edges(self, x):
        # Pad the input to maintain spatial dimensions
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        # Apply Gaussian filter
        smoothed = F.conv2d(x_padded, self.kernel)
        
        # Calculate gradients (Sobel-like)
        grad_x = smoothed[:, :, 1:, :-1] - smoothed[:, :, 1:, 1:]
        grad_y = smoothed[:, :, :-1, 1:] - smoothed[:, :, 1:, 1:]
        
        # Calculate gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return grad_mag
    
    def forward(self, pred, target):
        # Apply sigmoid if the input is logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Extract edges from prediction and target
        pred_edges = self._extract_edges(pred)
        target_edges = self._extract_edges(target)
        
        # Calculate topology loss - penalize differences in edge structure
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        # Penalize broken connections - focus on target edges that are missed by prediction
        missed_connections = target_edges * (1 - pred_edges)
        broken_penalty = (missed_connections**2).mean() * self.penalty_weight
        
        return edge_loss + broken_penalty

class ClassBalancedBCELoss(nn.Module):
    """
    Class-balanced BCE loss with weights determined by class frequency 
    in each batch to enhance thin vessel detection
    """
    def __init__(self, beta=0.99, smooth=1e-5):
        super(ClassBalancedBCELoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Calculate class frequencies
        pos_pixels = target.sum(dim=[1, 2, 3], keepdim=True)
        neg_pixels = target.shape[1] * target.shape[2] * target.shape[3] - pos_pixels
        total_pixels = pos_pixels + neg_pixels
        
        # Calculate class weights (higher weight for positive/vessel class)
        pos_weight = (1 - self.beta) / (1 - torch.pow(self.beta, pos_pixels + self.smooth))
        neg_weight = (1 - self.beta) / (1 - torch.pow(self.beta, neg_pixels + self.smooth))
        
        # Normalize weights
        pos_weight = pos_weight / (pos_weight + neg_weight)
        neg_weight = neg_weight / (pos_weight + neg_weight)
        
        # Create weight map
        weights = target * pos_weight + (1 - target) * neg_weight
        
        # Apply weighted BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * weights).mean()
        
        return weighted_bce

class CombinedLoss(nn.Module):
    """
    Enhanced combined loss function as specified in the plan:
    TotalLoss = dice_weight ⋅ DiceLoss + tversky_weight ⋅ FocalTversky + topology_weight ⋅ TopologyAwareLoss
    Optimized for thin vessel detection with focus on connectivity
    """
    def __init__(self, dice_weight=0.5, tversky_weight=0.3, topology_weight=0.2, 
                 tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=0.75, 
                 smooth=1.0, use_cb_bce=True):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.topology_weight = topology_weight
        self.use_cb_bce = use_cb_bce
        
        self.dice = DiceLoss(smooth=smooth)
        self.tversky = FocalTverskyLoss(
            alpha=tversky_alpha, 
            beta=tversky_beta, 
            gamma=tversky_gamma, 
            smooth=smooth
        )
        self.topology = TopologyAwareLoss(kernel_size=5, sigma=1.0, penalty_weight=1.0)
        
        # Class-balanced BCE for better handling of class imbalance
        if use_cb_bce:
            self.bce = ClassBalancedBCELoss(beta=0.99)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model prediction (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) - binary mask
        """
        # For Dice and Tversky, apply sigmoid to get probabilities
        probs = torch.sigmoid(pred)
        
        # Apply each loss component
        dice_loss = self.dice(probs, target)
        tversky_loss = self.tversky(pred, target)  # Will handle sigmoid internally
        topology_loss = self.topology(pred, target)  # Will handle sigmoid internally
        
        # Also calculate BCE loss
        bce_loss = self.bce(pred, target)
        
        # Add a small regularization that prevents predicting all pixels as 1
        # This helps avoid the perfect recall / terrible precision problem
        all_positive_penalty = torch.mean(probs) * 0.01
        
        # Combine losses with weights
        total_loss = (
            self.dice_weight * dice_loss + 
            self.tversky_weight * tversky_loss + 
            self.topology_weight * topology_loss +
            0.1 * bce_loss +  # Small BCE component
            all_positive_penalty  # Penalty for predicting too many positives
        )
        
        # For debugging
        losses = {
            'total': total_loss,
            'dice': dice_loss,
            'tversky': tversky_loss,
            'topology': topology_loss,
            'bce': bce_loss,
            'all_positive_penalty': all_positive_penalty
        }
        
        return total_loss 