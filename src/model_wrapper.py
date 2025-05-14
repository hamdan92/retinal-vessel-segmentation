from model import SwinResNetPlus, get_swin_res_net_plus, DynamicThreshold
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinResNetEnhanced(SwinResNetPlus):
    """
    Enhanced SwinResNetPlus with optimizations for thin vessel detection
    
    Improvements:
    1. Improved boundary detection with edge-aware convolutional blocks
    2. Label smoothing for regularization
    3. Enhanced multi-scale supervision
    4. Dynamic threshold learning
    """
    def __init__(self, 
                 input_channels=1,
                 num_classes=1,
                 use_swin=False,
                 encoder_name="swin_tiny_patch4_window7_224",
                 pretrained=True,
                 output_size=(512, 512),
                 edge_detection=True,
                 deep_supervision_weights=[0.7, 0.2, 0.1],
                 label_smoothing=0.05,
                 use_dynamic_threshold=True):
        # Initialize parent class
        super(SwinResNetEnhanced, self).__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            use_swin=use_swin,
            encoder_name=encoder_name,
            pretrained=pretrained,
            output_size=output_size
        )
        
        self.edge_detection = edge_detection
        self.deep_supervision_weights = deep_supervision_weights
        self.label_smoothing = label_smoothing
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # Add boundary detection branch if enabled
        if edge_detection:
            self.edge_detection_branch = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1)
            )
            
        # Add dynamic threshold layer if enabled
        if use_dynamic_threshold:
            self.dynamic_threshold = DynamicThreshold(initial_threshold=0.5, channels=num_classes)
    
    def forward(self, x):
        # Get features from parent class
        if self.training:
            main_output, aux_outputs = super().forward(x)
            
            # Add edge detection branch if enabled
            if self.edge_detection:
                # Derive edge features from input with a lightweight projection if channel mismatch
                if not hasattr(self, "edge_proj") or self.edge_proj.in_channels != x.shape[1]:
                    self.edge_proj = nn.Conv2d(x.shape[1], 32, kernel_size=1, bias=False).to(x.device)
                projected = self.edge_proj(x)
                edge_features = self.decoder_blocks[-1].attention(projected)
                
                edge_output = self.edge_detection_branch(edge_features)
                
                # Resize to match target size if necessary
                target_size = tuple(int(s) for s in self.output_size)
                edge_output = F.interpolate(
                    edge_output,
                    size=target_size,
                    mode='bilinear',
                    align_corners=True
                )
                
                # Return main, edge, and auxiliary outputs
                return [main_output, edge_output] + aux_outputs
            else:
                return main_output, aux_outputs
        else:
            output = super().forward(x)
            
            # Apply dynamic threshold if enabled during inference
            if self.use_dynamic_threshold and not self.training:
                # Sigmoid is required here as the output is still logits
                probs = torch.sigmoid(output)
                return self.dynamic_threshold(probs)
            
            return output
    
    def apply_label_smoothing(self, targets):
        """
        Apply label smoothing to target masks
        
        Args:
            targets: Target masks [B, 1, H, W]
        
        Returns:
            Smoothed targets
        """
        if self.label_smoothing > 0:
            # Convert binary targets to soft targets
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        return targets

# Create a wrapper class for compatibility
class SwinResNet(SwinResNetEnhanced):
    """
    Wrapper class for enhanced SwinResNetPlus to maintain compatibility with existing code
    """
    pass

# Also create a wrapper function for convenience
def get_swin_res_net(*args, **kwargs):
    """
    Wrapper for get_swin_res_net_plus to maintain compatibility
    """
    model = get_swin_res_net_plus(*args, **kwargs)
    
    # Convert to enhanced version if it's a SwinResNetPlus
    if isinstance(model, SwinResNetPlus) and not isinstance(model, SwinResNetEnhanced):
        # Create enhanced model with same parameters
        enhanced_model = SwinResNetEnhanced(
            input_channels=model.encoder.conv1[0].in_channels,
            num_classes=model.final_conv[-1].out_channels,
            use_swin=hasattr(model, 'encoder') and hasattr(model.encoder, 'features'),
            encoder_name=kwargs.get('encoder_name', "swin_tiny_patch4_window7_224"),
            pretrained=kwargs.get('pretrained', True),
            output_size=model.output_size,
            edge_detection=kwargs.get('edge_detection', True),
            deep_supervision_weights=kwargs.get('deep_supervision_weights', [0.7, 0.2, 0.1]),
            label_smoothing=kwargs.get('label_smoothing', 0.05),
            use_dynamic_threshold=kwargs.get('use_dynamic_threshold', True)
        )
        
        # Copy weights from original model to enhanced model
        enhanced_model.load_state_dict(model.state_dict(), strict=False)
        
        return enhanced_model
    
    return model 