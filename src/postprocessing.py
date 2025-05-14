import numpy as np
import cv2
from skimage import filters
from skimage import morphology
from skimage import measure

def apply_otsu_threshold(prob_map):
    """
    Apply Otsu's thresholding to probability map
    
    Args:
        prob_map: Probability map (numpy array)
        
    Returns:
        Binary mask
    """
    # Convert to uint8 if float
    if prob_map.dtype == np.float32 or prob_map.dtype == np.float64:
        prob_map_uint8 = (prob_map * 255).astype(np.uint8)
    else:
        prob_map_uint8 = prob_map
        
    # Apply Otsu's thresholding
    threshold = filters.threshold_otsu(prob_map_uint8)
    binary_mask = (prob_map_uint8 > threshold).astype(np.uint8)
    
    return binary_mask

def remove_small_objects(binary_mask, min_size=50):
    """
    Remove small connected components from binary mask
    
    Args:
        binary_mask: Binary mask
        min_size: Minimum size of connected components to keep
        
    Returns:
        Cleaned binary mask
    """
    # Label connected components
    labeled_mask = measure.label(binary_mask)
    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(labeled_mask, min_size=min_size)
    # Convert back to binary
    cleaned_mask = (cleaned_mask > 0).astype(np.uint8)
    
    return cleaned_mask

def skeletonize(binary_mask):
    """
    Skeletonize binary mask for vessel width analysis
    
    Args:
        binary_mask: Binary mask
        
    Returns:
        Skeletonized mask
    """
    # Skeletonize the binary mask
    skeleton = morphology.skeletonize(binary_mask > 0)
    # Convert to uint8
    skeleton = skeleton.astype(np.uint8)
    
    return skeleton

def apply_morphological_operations(binary_mask, kernel_size=3):
    """
    Apply morphological operations to refine the vessels
    
    Args:
        binary_mask: Binary mask
        kernel_size: Size of the structuring element
        
    Returns:
        Refined binary mask
    """
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply closing to fill small holes
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply opening to remove small noise
    refined_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    return refined_mask

def adaptive_vessel_threshold(prob_map, fov_mask=None, sensitivity=0.45):
    """
    Apply adaptive thresholding optimized for vessel segmentation
    
    Args:
        prob_map: Probability map (numpy array)
        fov_mask: Field of view mask (optional)
        sensitivity: Controls threshold sensitivity (lower = more vessels)
        
    Returns:
        Binary mask
    """
    # Apply FOV mask if provided
    if fov_mask is not None:
        prob_map = prob_map * fov_mask
        
    # Convert to uint8 if float
    if prob_map.dtype == np.float32 or prob_map.dtype == np.float64:
        prob_map_uint8 = (prob_map * 255).astype(np.uint8)
    else:
        prob_map_uint8 = prob_map
        
    # First try Otsu's thresholding
    otsu_threshold = filters.threshold_otsu(prob_map_uint8)
    
    # Use a lower threshold to capture thin vessels
    # The sensitivity parameter controls how much lower than Otsu
    adjusted_threshold = otsu_threshold * sensitivity
    
    # Create binary mask
    binary_mask = (prob_map_uint8 > adjusted_threshold).astype(np.uint8)
    
    return binary_mask

def vessel_enhancement_frangi(prob_map, scale_range=(1, 5), scale_step=1):
    """
    Apply Frangi vessel enhancement filter to improve vessel detection
    
    Args:
        prob_map: Probability map (numpy array)
        scale_range: Range of scales for multiscale Frangi filtering
        scale_step: Step size for scale sampling
        
    Returns:
        Enhanced probability map
    """
    from skimage.filters import frangi
    
    # Convert to float if not already
    if prob_map.dtype != np.float32 and prob_map.dtype != np.float64:
        prob_map = prob_map.astype(np.float32) / 255.0
        
    # Apply Frangi filter for vessel enhancement
    enhanced = frangi(
        prob_map, 
        scale_range=scale_range, 
        scale_step=scale_step,
        black_ridges=False  # Vessels are bright in our case
    )
    
    # Normalize to [0, 1]
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-7)
    
    # Combine with original probabilities
    combined = 0.7 * prob_map + 0.3 * enhanced
    
    return combined

def refine_vessels(binary_mask, kernel_size=3):
    """
    Advanced vessel refinement to better preserve thin vessels
    
    Args:
        binary_mask: Binary vessel mask
        kernel_size: Size of the structuring element
        
    Returns:
        Refined vessel mask
    """
    # Create kernels for morphological operations
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size*2+1, kernel_size//2))
    line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size//2, kernel_size*2+1))
    
    # Apply closing with different kernels to better capture vessels in all orientations
    closed_h = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, line_kernel_h)
    closed_v = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, line_kernel_v)
    closed_disk = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, disk_kernel)
    
    # Combine the results (union)
    combined = np.maximum(np.maximum(closed_h, closed_v), closed_disk)
    
    # Remove small isolated pixels
    refined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, disk_kernel)
    
    return refined

def apply_crf(probs, img, theta_pos=10, theta_rgb=15, theta_bil=10, w1=5, w2=3, max_iter=10):
    """
    Apply Conditional Random Field (CRF) refinement to predicted probability map
    
    Args:
        probs: Predicted probability map [H, W] in range [0, 1]
        img: Original image [H, W, 3] or [H, W]
        theta_pos: Positional kernel bandwidth
        theta_rgb: RGB kernel bandwidth
        theta_bil: Bilateral kernel bandwidth
        w1: Weight of appearance kernel
        w2: Weight of smoothness kernel
        max_iter: Maximum CRF iterations
        
    Returns:
        Refined binary mask [H, W]
    """
    import numpy as np
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    
    # Check if img is grayscale and convert to RGB if needed
    if len(img.shape) == 2:
        img_rgb = np.stack([img] * 3, axis=2)
    else:
        img_rgb = img
    
    # Ensure img_rgb is in range [0, 255] and uint8
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    
    # Create CRF model
    h, w = probs.shape
    
    # Convert probability map to softmax format needed by CRF [2, H, W]
    softmax = np.zeros((2, h, w), dtype=np.float32)
    softmax[0] = 1 - probs  # Background probability
    softmax[1] = probs  # Foreground probability
    
    # Create CRF model
    crf = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
    
    # Add unary potentials
    unary = unary_from_softmax(softmax)
    crf.setUnaryEnergy(unary)
    
    # Add pairwise potentials (position only - "smoothness kernel")
    pairwise_gaussian = create_pairwise_gaussian((theta_pos, theta_pos), (h, w))
    crf.addPairwiseEnergy(pairwise_gaussian, compat=w2)
    
    # Add pairwise potentials (position and RGB - "appearance kernel")
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(theta_bil, theta_bil),
        schan=(theta_rgb, theta_rgb, theta_rgb),
        img=img_rgb,
        chdim=2
    )
    crf.addPairwiseEnergy(pairwise_bilateral, compat=w1)
    
    # Perform inference
    q = crf.inference(max_iter)
    
    # Get MAP prediction
    map_prediction = np.argmax(q, axis=0).reshape((h, w))
    
    return map_prediction.astype(np.float32)

def process_prediction(pred_prob, fov_mask=None, min_size=30, apply_morphology=True, 
                       kernel_size=3, skeletonize_result=False, enhance_vessels=True,
                       threshold_sensitivity=0.45, apply_crf=False, original_image=None):
    """
    Post-process prediction probability map
    
    Args:
        pred_prob: Prediction probability map [H, W] in range [0, 1]
        fov_mask: Field of view mask [H, W] (optional)
        min_size: Minimum size for small object removal
        apply_morphology: Whether to apply morphological operations
        kernel_size: Kernel size for morphological operations
        skeletonize_result: Whether to skeletonize the result
        enhance_vessels: Whether to enhance vessel connectivity
        threshold_sensitivity: Threshold sensitivity (lower = more vessels)
        apply_crf: Whether to apply CRF refinement
        original_image: Original image for CRF refinement [H, W, 3] or [H, W]
        
    Returns:
        Post-processed binary mask [H, W]
    """
    import numpy as np
    import cv2
    from skimage import morphology
    
    # Apply field of view mask if provided
    if fov_mask is not None:
        pred_prob = pred_prob * fov_mask
    
    # Apply CRF refinement if requested
    if apply_crf and original_image is not None:
        try:
            from pydensecrf.densecrf import DenseCRF2D
            pred_prob = apply_crf(pred_prob, original_image)
        except ImportError:
            print("Warning: pydensecrf not installed. Skipping CRF refinement.")
    
    # Dynamic thresholding adapted to the vessel density
    if threshold_sensitivity < 1.0:
        # Use Otsu's method to find a base threshold
        pred_8bit = (pred_prob * 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(pred_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold = otsu_threshold / 255.0
        
        # Adjust threshold based on sensitivity parameter
        dynamic_threshold = max(0.2, otsu_threshold * threshold_sensitivity)
    else:
        # Use fixed threshold if sensitivity is >= 1.0
        dynamic_threshold = threshold_sensitivity
    
    # Apply threshold
    pred_binary = (pred_prob >= dynamic_threshold).astype(np.uint8)
    
    # Apply morphological operations if requested
    if apply_morphology:
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Close operation to connect nearby vessels
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small connected components
        if min_size > 0:
            pred_binary = morphology.remove_small_objects(pred_binary.astype(bool), min_size=min_size)
            pred_binary = pred_binary.astype(np.uint8)
        
        # Enhance vessel connectivity with thinning and dilation
        if enhance_vessels:
            # Skeletonize
            skeleton = morphology.skeletonize(pred_binary.astype(bool))
            # Dilate the skeleton
            skeleton_dilated = cv2.dilate(skeleton.astype(np.uint8), 
                                          np.ones((kernel_size-1, kernel_size-1), np.uint8))
            # Combine with original prediction
            pred_binary = np.maximum(pred_binary, skeleton_dilated)
    
    # Skeletonize if requested
    if skeletonize_result:
        pred_binary = morphology.skeletonize(pred_binary.astype(bool)).astype(np.uint8)
    
    return pred_binary.astype(np.float32)

def process_batch_predictions(preds, fov_masks=None, min_size=50, apply_morphology=True, 
                           kernel_size=3, skeletonize=False, enhance_vessels=True,
                           threshold_sensitivity=0.45, apply_crf=False, original_images=None):
    """
    Process a batch of predictions
    
    Args:
        preds: Batch of prediction probability maps [B, C, H, W] or [B, H, W]
        fov_masks: Batch of FOV masks [B, C, H, W] or [B, H, W] (optional)
        min_size: Minimum size for small object removal
        apply_morphology: Whether to apply morphological operations
        kernel_size: Kernel size for morphological operations
        skeletonize: Whether to skeletonize the result
        enhance_vessels: Whether to enhance vessel connectivity
        threshold_sensitivity: Controls threshold sensitivity (lower = more vessels)
        apply_crf: Whether to apply CRF refinement
        original_images: Batch of original images for CRF [B, C, H, W] or [B, H, W]
        
    Returns:
        Batch of post-processed binary masks [B, C, H, W] or [B, H, W]
    """
    import numpy as np
    
    # Ensure preds is a numpy array
    if not isinstance(preds, np.ndarray):
        preds = preds.numpy()
    
    # Get batch size
    batch_size = preds.shape[0]
    
    # Convert fov_masks to numpy array if provided
    if fov_masks is not None and not isinstance(fov_masks, np.ndarray):
        fov_masks = fov_masks.numpy()
    
    # Initialize array for processed predictions
    # Match the original shape
    if len(preds.shape) == 4:  # [B, C, H, W]
        processed_preds = np.zeros_like(preds)
    else:  # [B, H, W]
        processed_preds = np.zeros_like(preds)
    
    # Process each prediction in batch
    for i in range(batch_size):
        # Get current prediction and reshape if needed
        if len(preds.shape) == 4:  # [B, C, H, W]
            pred_prob = preds[i, 0]  # Take first channel
        else:  # [B, H, W]
            pred_prob = preds[i]
        
        # Get FOV mask if provided
        fov_mask = None
        if fov_masks is not None:
            if len(fov_masks.shape) == 4:  # [B, C, H, W]
                fov_mask = fov_masks[i, 0]  # Take first channel
            else:  # [B, H, W]
                fov_mask = fov_masks[i]
        
        # Get original image if provided
        orig_img = None
        if original_images is not None and apply_crf:
            if len(original_images.shape) == 4:  # [B, C, H, W]
                # For RGB, transpose from [C, H, W] to [H, W, C]
                if original_images.shape[1] == 3:
                    orig_img = np.transpose(original_images[i], (1, 2, 0))
                else:
                    orig_img = original_images[i, 0]  # Take first channel for grayscale
            else:  # [B, H, W]
                orig_img = original_images[i]
        
        # Apply post-processing
        processed_pred = process_prediction(
            pred_prob=pred_prob,
            fov_mask=fov_mask,
            min_size=min_size,
            apply_morphology=apply_morphology,
            kernel_size=kernel_size,
            skeletonize_result=skeletonize,
            enhance_vessels=enhance_vessels,
            threshold_sensitivity=threshold_sensitivity,
            apply_crf=apply_crf,
            original_image=orig_img
        )
        
        # Store processed prediction
        if len(preds.shape) == 4:  # [B, C, H, W]
            processed_preds[i, 0] = processed_pred
        else:  # [B, H, W]
            processed_preds[i] = processed_pred
    
    return processed_preds

def ensemble_predictions(pred_list, method='weighted_mean', threshold=0.5, weights=None):
    """
    Enhanced ensemble multiple predictions with weighted options
    
    Args:
        pred_list: List of prediction probability maps
        method: Ensembling method ('mean', 'weighted_mean', 'max', or 'voting')
        threshold: Threshold for voting method
        weights: List of weights for weighted_mean method
        
    Returns:
        Ensembled prediction
    """
    # Convert to numpy if needed
    numpy_preds = []
    for pred in pred_list:
        if hasattr(pred, 'cpu'):
            numpy_preds.append(pred.cpu().numpy())
        else:
            numpy_preds.append(pred)
    
    # Stack predictions
    stacked_preds = np.stack(numpy_preds)
    
    if method == 'mean':
        # Mean probability
        ensemble = np.mean(stacked_preds, axis=0)
    elif method == 'weighted_mean':
        # Weighted mean with provided weights
        if weights is None:
            # Default: give more weight to central prediction
            weights = np.ones(len(numpy_preds))
            if len(weights) > 2:
                weights[0] = 1.5  # More weight to original (non-augmented) prediction
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Apply weighted average
        ensemble = np.sum(stacked_preds * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
    elif method == 'max':
        # Maximum probability
        ensemble = np.max(stacked_preds, axis=0)
    elif method == 'voting':
        # Majority voting
        binary_preds = (stacked_preds > threshold).astype(np.uint8)
        votes = np.sum(binary_preds, axis=0)
        ensemble = (votes > (len(pred_list) // 2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble

def apply_test_time_augmentation(model, image, device, augmentations=None):
    """
    Apply test-time augmentation to an image and ensemble the results
    
    Args:
        model: PyTorch model
        image: Input image (torch tensor)
        device: Device to use for inference
        augmentations: List of augmentation functions
        
    Returns:
        Ensembled prediction
    """
    import torch
    import torch.nn.functional as F
    
    # Default augmentations: identity, horizontal flip, vertical flip, 90-degree rotation
    if augmentations is None:
        def identity(x): return x
        def hflip(x): return torch.flip(x, dims=[3])
        def vflip(x): return torch.flip(x, dims=[2])
        def rot90(x): return torch.rot90(x, k=1, dims=[2, 3])
        
        augmentations = [identity, hflip, vflip, rot90]
    
    # Apply model to original and augmented images
    pred_list = []
    
    with torch.no_grad():
        for aug_func in augmentations:
            # Apply augmentation
            aug_image = aug_func(image)
            
            # Forward pass
            output = model(aug_image.to(device))
            
            # Handle deep supervision output
            if isinstance(output, list):
                output = output[0]
            
            # Apply sigmoid
            pred = torch.sigmoid(output)
            
            # Reverse augmentation for prediction
            if aug_func == hflip:
                pred = torch.flip(pred, dims=[3])
            elif aug_func == vflip:
                pred = torch.flip(pred, dims=[2])
            elif aug_func == rot90:
                pred = torch.rot90(pred, k=-1, dims=[2, 3])
            
            pred_list.append(pred.cpu())
    
    # Ensemble predictions
    ensemble = ensemble_predictions(pred_list, method='mean')
    
    return ensemble 