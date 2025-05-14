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

def process_prediction(pred, fov_mask=None, min_size=30, apply_morphology=True, 
                      kernel_size=3, skeletonize_result=False, enhance_vessels=True,
                      threshold_sensitivity=0.45):
    """
    Apply full post-processing pipeline to prediction, optimized for thin vessels
    
    Args:
        pred: Prediction probability map (numpy array)
        fov_mask: Field of view mask (optional)
        min_size: Minimum size of connected components to keep
        apply_morphology: Whether to apply morphological operations
        kernel_size: Size of the structuring element for morphological operations
        skeletonize_result: Whether to skeletonize the result
        enhance_vessels: Whether to apply vessel enhancement
        threshold_sensitivity: Controls threshold sensitivity for vessel detection
        
    Returns:
        Post-processed binary mask
    """
    # Apply FOV mask if provided
    if fov_mask is not None:
        pred = pred * fov_mask
    
    # Apply vessel enhancement if requested
    if enhance_vessels:
        pred = vessel_enhancement_frangi(pred)
    
    # Apply adaptive thresholding
    binary_mask = adaptive_vessel_threshold(pred, fov_mask, threshold_sensitivity)
    
    # Remove small connected components - use smaller min_size to keep thin vessels
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
    
    # Apply advanced vessel refinement if requested
    if apply_morphology:
        refined_mask = refine_vessels(cleaned_mask, kernel_size=kernel_size)
    else:
        refined_mask = cleaned_mask
    
    # Skeletonize if requested
    if skeletonize_result:
        result = skeletonize(refined_mask)
    else:
        result = refined_mask
    
    return result

def process_batch_predictions(preds, fov_masks=None, min_size=50, apply_morphology=True, 
                              kernel_size=3, skeletonize_result=False):
    """
    Apply post-processing to a batch of predictions
    
    Args:
        preds: Batch of prediction probability maps (numpy array or torch tensor)
        fov_masks: Batch of field of view masks (optional)
        min_size: Minimum size of connected components to keep
        apply_morphology: Whether to apply morphological operations
        kernel_size: Size of the structuring element for morphological operations
        skeletonize_result: Whether to skeletonize the result
        
    Returns:
        Batch of post-processed binary masks
    """
    # Convert to numpy if torch tensor
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()
    
    if fov_masks is not None and hasattr(fov_masks, 'cpu'):
        fov_masks = fov_masks.cpu().numpy()
    
    batch_size = preds.shape[0]
    results = []
    
    for i in range(batch_size):
        # Get single prediction
        pred = preds[i].squeeze()
        
        # Get corresponding FOV mask if available
        fov_mask = None
        if fov_masks is not None:
            fov_mask = fov_masks[i].squeeze()
        
        # Process prediction
        result = process_prediction(
            pred, 
            fov_mask=fov_mask, 
            min_size=min_size, 
            apply_morphology=apply_morphology,
            kernel_size=kernel_size,
            skeletonize_result=skeletonize_result
        )
        
        results.append(result)
    
    # Stack results
    return np.stack(results)

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