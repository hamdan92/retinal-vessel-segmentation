import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict, Tuple, Optional


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention module as per the paper:
    'Coordinate Attention for Efficient Mobile Network Design'
    Modified to handle arbitrary feature map sizes
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        # Height-wise attention
        x_h = self.pool_h(x)  # Shape: [N, C, H, 1]
        # Width-wise attention
        x_w = self.pool_w(x)  # Shape: [N, C, 1, W]
        x_w = x_w.transpose(2, 3)  # Shape: [N, C, W, 1]
        
        # Process height-wise and width-wise features separately
        # Process height attention
        y_h = self.conv1(x_h)
        y_h = self.bn1(y_h)
        y_h = self.act(y_h)
        a_h = torch.sigmoid(self.conv_h(y_h))
        
        # Process width attention
        y_w = self.conv1(x_w)
        y_w = self.bn1(y_w)
        y_w = self.act(y_w)
        a_w = torch.sigmoid(self.conv_w(y_w))
        a_w = a_w.transpose(2, 3)  # Transpose back
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out


class GatedSkipConnection(nn.Module):
    """
    Gated Skip Connection for better feature selection in U-Net
    """
    def __init__(self, in_channels, out_channels):
        super(GatedSkipConnection, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        gate = self.gate(x)
        x = self.conv(x)
        return x * gate


class Res2NetBlock(nn.Module):
    """
    Res2Net block with multi-scale feature extraction
    """
    def __init__(self, in_channels, out_channels, scale=4):
        super(Res2NetBlock, self).__init__()
        width = out_channels // scale
        self.scale = scale
        
        # Input convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multiple branch processing
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True)
            ) for _ in range(scale-1)
        ])
        
        # Output convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        # Initial projection
        x = self.conv1(x)
        
        # Split along channels
        spx = torch.split(x, x.shape[1] // self.scale, 1)
        xs = list(spx)
        
        # Multi-scale processing
        for i, conv in enumerate(self.convs):
            if i == 0:
                xs[i+1] = conv(xs[i])
            else:
                xs[i+1] = conv(xs[i] + xs[i+1])
        
        # Concatenate all scales
        x = torch.cat(xs, dim=1)
        x = self.conv2(x)
        
        # Add residual connection
        x = x + residual
        
        return x


class TwoPathInteractiveModule(nn.Module):
    """
    Two-path interactive fusion module combining Transformer and Res2Net
    """
    def __init__(self, in_channels, out_channels):
        super(TwoPathInteractiveModule, self).__init__()
        
        # Transformer path processing
        self.transformer_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CoordinateAttention(out_channels, reduction=16)
        )
        
        # Res2Net path
        self.res2net_path = Res2NetBlock(in_channels, out_channels, scale=4)
        self.align_conv = None  # Will be initialised at runtime if channel mismatch occurs
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, transformer_features, conv_features=None):
        # If conv_features is not provided, use transformer_features
        if conv_features is None:
            conv_features = transformer_features
        
        # Process transformer path
        t_path = self.transformer_path(transformer_features)
        
        # Align channels if necessary
        if conv_features.shape[1] != transformer_features.shape[1]:
            # Lazily create a 1×1 conv to match channels
            if self.align_conv is None or self.align_conv.in_channels != conv_features.shape[1]:
                self.align_conv = nn.Conv2d(conv_features.shape[1], transformer_features.shape[1], kernel_size=1, bias=False).to(conv_features.device)
            conv_features = self.align_conv(conv_features)
        # Process Res2Net path on the (possibly) projected features
        r_path = self.res2net_path(conv_features)
        
        # Ensure spatial resolution match before concatenation
        if r_path.shape[2:] != t_path.shape[2:]:
            r_path = F.interpolate(r_path, size=t_path.shape[2:], mode='bilinear', align_corners=True)
        
        # Combine both paths
        combined = torch.cat([t_path, r_path], dim=1)
        output = self.fusion_conv(combined)
        
        return output


class DecoderBlock(nn.Module):
    """
    Decoder block with gated skip connection and coordinate attention
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # Gated skip connection
        self.skip_connection = GatedSkipConnection(skip_channels, out_channels)
        
        # Upsampling and convolution
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Coordinate attention
        self.attention = CoordinateAttention(out_channels, reduction=16)
        
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Process skip connection
        processed_skip = self.skip_connection(skip)
        
        # Ensure spatial size match
        if processed_skip.shape[2:] != x.shape[2:]:
            processed_skip = F.interpolate(processed_skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate
        x = torch.cat([x, processed_skip], dim=1)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply attention
        x = self.attention(x)
        
        return x


class CustomResNetEncoder(nn.Module):
    """
    Custom ResNet-based encoder to replace Swin Transformer for flexibility with input sizes
    """
    def __init__(self, in_channels=1):
        super(CustomResNetEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Res2Net blocks with increasing channels and downsampling
        self.layer1 = Res2NetBlock(64, 96, scale=4)  # 1/4 scale
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer2 = Res2NetBlock(96, 192, scale=4)  # 1/8 scale
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer3 = Res2NetBlock(192, 384, scale=4)  # 1/16 scale
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer4 = Res2NetBlock(384, 768, scale=4)  # 1/32 scale
        
    def forward(self, x):
        # Initial processing
        x = self.conv1(x)  # 1/2 scale
        x = self.maxpool(x)  # 1/4 scale
        
        # Layer 1
        f1 = self.layer1(x)  # 1/4 scale
        x = self.down1(f1)
        
        # Layer 2
        f2 = self.layer2(x)  # 1/8 scale
        x = self.down2(f2)
        
        # Layer 3
        f3 = self.layer3(x)  # 1/16 scale
        x = self.down3(f3)
        
        # Layer 4
        f4 = self.layer4(x)  # 1/32 scale
        
        # Return features from all scales
        return [f1, f2, f3, f4]


class SwinResNetPlus(nn.Module):
    """
    Swin-Res-Net+ model with encoder, two-path interactive fusion,
    U-Net decoder with gated skip connections, and deep supervision
    """
    def __init__(
        self, 
        input_channels=1,
        num_classes=1,
        use_swin=False,
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        output_size=(512, 512)
    ):
        super(SwinResNetPlus, self).__init__()
        self.output_size = output_size
        
        # Feature dimensions will depend on the encoder
        if use_swin:
            # Try to use Swin Transformer (may have input size limitations)
            try:
                self.encoder = timm.create_model(
                    encoder_name,
                    pretrained=pretrained,
                    features_only=True,
                    in_chans=input_channels
                )
                feature_dims = self.encoder.feature_info.channels()
            except Exception as e:
                print(f"Warning: Could not load Swin Transformer encoder: {e}")
                print("Falling back to custom ResNet encoder")
                use_swin = False
        
        if not use_swin:
            # Use custom encoder for flexibility with input sizes
            self.encoder = CustomResNetEncoder(in_channels=input_channels)
            feature_dims = [96, 192, 384, 768]  # Match the custom encoder's output channels
            
        # Two-path interactive fusion modules
        self.fusion_blocks = nn.ModuleList([
            TwoPathInteractiveModule(feature_dims[3], 512),
            TwoPathInteractiveModule(feature_dims[2], 256),
            TwoPathInteractiveModule(feature_dims[1], 128),
            TwoPathInteractiveModule(feature_dims[0], 64)
        ])
        
        # Decoder blocks with corrected skip channel sizes to match fused feature maps
        fused_channels = [512, 256, 128, 64]
        skip_channels = [256, 128, 64, input_channels]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(fused_channels[i], skip_channels[i], fused_channels[i+1] if i < 3 else 32)
            for i in range(4)
        ])
        
        # Auxiliary output heads for deep supervision
        self.aux_head1 = nn.Conv2d(256, num_classes, kernel_size=1)  # 1/4 scale
        self.aux_head2 = nn.Conv2d(128, num_classes, kernel_size=1)  # 1/8 scale
        
        # Final output head
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Store input for potential skip connection
        input_size = x.shape[2:]
        input_tensor = x
        
        # Get encoder features
        features = self.encoder(x)
        
        # Apply fusion blocks to encoder features
        fused_features = []
        for i, feature in enumerate(reversed(features)):
            if i == 0:
                # First fusion block uses only transformer features
                fused = self.fusion_blocks[i](feature)
            else:
                # Subsequent fusion blocks use previous fused features and current transformer features
                fused = self.fusion_blocks[i](feature, fused_features[-1])
            fused_features.append(fused)
        
        # Apply decoder blocks
        x = fused_features[0]
        aux_outputs = []
        
        for i, decoder in enumerate(self.decoder_blocks):
            if i < 3:  # For the first 3 decoder blocks
                # Use fused features for skip connections
                x = decoder(x, fused_features[i+1] if i < len(fused_features)-1 else input_tensor)
                
                # Store auxiliary outputs for deep supervision
                if i == 0:  # 1/4 scale
                    aux1 = self.aux_head1(x)
                    # Resize to match target size
                    aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
                    aux_outputs.append(aux1)
                elif i == 1:  # 1/8 scale
                    aux2 = self.aux_head2(x)
                    # Resize to match target size
                    aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
                    aux_outputs.append(aux2)
            else:
                # Last decoder block uses input tensor for skip
                x = decoder(x, input_tensor)
        
        # Apply final convolution
        logits = self.final_conv(x)
        
        # Resize to match target size if necessary
        if self.output_size is not None and (x.shape[2] != self.output_size[0] or x.shape[3] != self.output_size[1]):
            target_size = tuple(int(s) for s in self.output_size)
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=True)
        
        # Return main output and auxiliary outputs for deep supervision
        if self.training:
            return logits, aux_outputs
        else:
            return logits


class DiceFocalLoss(nn.Module):
    """
    Combined Dice and Focal loss for binary segmentation
    """
    def __init__(self, dice_weight=0.4, focal_weight=0.4, bce_weight=0.2, alpha=0.5, beta=0.7):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.alpha = alpha  # Controls weight of false positives
        self.beta = beta    # Controls weight of false negatives (higher = more weight to FN)
        self.smooth = 1e-6
        
    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten predictions and targets
        batch_size = logits.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Calculate Focal-Tversky loss
        # More weight to false negatives for better vessel detection
        tp = torch.sum(probs * targets, dim=1)
        fp = torch.sum(probs * (1 - targets), dim=1) * self.alpha
        fn = torch.sum((1 - probs) * targets, dim=1) * self.beta
        
        tversky_coef = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        focal_tversky = torch.pow(1 - tversky_coef, 0.75)  # gamma=0.75 for focal effect
        focal_loss = torch.mean(focal_tversky)
        
        # Calculate Dice loss
        dice_coef = (2 * tp + self.smooth) / (torch.sum(probs, dim=1) + torch.sum(targets, dim=1) + self.smooth)
        dice_loss = 1 - torch.mean(dice_coef)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.view_as(logits), reduction='mean')
        
        # Combine losses
        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss + self.bce_weight * bce_loss
        
        return combined_loss


def get_swin_res_net_plus(
    input_channels=1,
    num_classes=1,
    pretrained=True,
    output_size=(512, 512)
):
    """
    Helper function to create a Swin-Res-Net+ model
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        pretrained: Whether to use pretrained Swin Transformer encoder
        output_size: Output size of the model
        
    Returns:
        SwinResNetPlus model
    """
    # First try with Swin, if it fails, fall back to custom encoder
    try:
        model = SwinResNetPlus(
            input_channels=input_channels,
            num_classes=num_classes,
            use_swin=True,
            encoder_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            output_size=output_size
        )
        # Test with dummy input
        dummy = torch.zeros(1, input_channels, output_size[0], output_size[1])
        with torch.no_grad():
            model.eval()
            model(dummy)
        return model
    except Exception as e:
        print(f"Falling back to custom encoder due to: {e}")
        return SwinResNetPlus(
            input_channels=input_channels,
            num_classes=num_classes,
            use_swin=False,
            pretrained=False,
            output_size=output_size
        )


if __name__ == "__main__":
    # Test model creation and forward pass
    model = get_swin_res_net_plus(input_channels=1, num_classes=1)
    
    # Print model summary
    print(f"Model created: Swin-Res-Net+")
    
    # Test input (batch_size, channels, height, width)
    x = torch.randn(2, 1, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        model.eval()
        output = model(x)
    
    # Print output shape
    if isinstance(output, tuple):
        print(f"Output shape (eval mode): {output[0].shape}")
    else:
        print(f"Output shape (eval mode): {output.shape}")
    
    # Test in training mode with auxiliary outputs
    model.train()
    output, aux_outputs = model(x)
    
    # Print output shapes
    print(f"Main output shape (train mode): {output.shape}")
    for i, aux in enumerate(aux_outputs):
        print(f"Auxiliary output {i+1} shape: {aux.shape}")
    
    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Test loss function
    loss_fn = DiceFocalLoss()
    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    loss = loss_fn(output, targets)
    print(f"Loss value: {loss.item()}") 