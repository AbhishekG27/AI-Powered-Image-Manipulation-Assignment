# Advanced Shoe Image Replacement with Enhanced Background Integration
# For Google Colab environment

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import requests
from PIL import Image
from google.colab.patches import cv2_imshow
from io import BytesIO
import os
from scipy.ndimage import gaussian_filter

# Install necessary packages if not already installed
!pip install -q segment-anything

# Download the SAM model weights if needed
if not os.path.exists('sam_vit_h_4b8939.pth'):
    !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Import SAM after installation
from segment_anything import sam_model_registry, SamPredictor

# Function to upload images from local machine to Colab
def upload_images():
    from google.colab import files
    uploaded = files.upload()
    return uploaded

# Function to create a better shoe mask using SAM
def create_shoes_mask_sam(image_path, predictor):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in the predictor
    predictor.set_image(image_rgb)
    
    # For shoes, we typically want to focus on the center of the image
    h, w = image.shape[:2]
    center_point = np.array([[w//2, h//2]])
    
    # Get masks from the center point
    masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=np.array([1]),  # 1 means foreground
        multimask_output=True
    )
    
    # Choose the mask with the highest score
    mask_idx = np.argmax(scores)
    mask = masks[mask_idx]
    
    # Convert to uint8 format
    shoe_mask = mask.astype(np.uint8) * 255
    
    # Clean up the mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    shoe_mask = cv2.morphologyEx(shoe_mask, cv2.MORPH_CLOSE, kernel)
    
    return shoe_mask, image

# Function to create a mask for the new shoe
def create_new_shoe_mask(new_shoe_path, predictor):
    new_shoe = cv2.imread(new_shoe_path)
    if new_shoe is None:
        raise ValueError(f"Could not load new shoe image at {new_shoe_path}")
    
    # Convert BGR to RGB
    new_shoe_rgb = cv2.cvtColor(new_shoe, cv2.COLOR_BGR2RGB)
    
    # Set image in the predictor
    predictor.set_image(new_shoe_rgb)
    
    # For shoes, we typically want to focus on the center of the image
    h, w = new_shoe.shape[:2]
    center_point = np.array([[w//2, h//2]])
    
    # Get masks from the center point
    masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=np.array([1]),  # 1 means foreground
        multimask_output=True
    )
    
    # Choose the mask with the highest score
    mask_idx = np.argmax(scores)
    mask = masks[mask_idx]
    
    # Convert to uint8 format
    new_shoe_mask = mask.astype(np.uint8) * 255
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    new_shoe_mask = cv2.morphologyEx(new_shoe_mask, cv2.MORPH_CLOSE, kernel)
    
    return new_shoe_mask, new_shoe

# Function to extract the shape of a shoe from its mask
def extract_shoe_shape(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask")
    
    # Get the largest contour (which should be the shoe)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a bounding rect
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Create a more precise convex hull
    hull = cv2.convexHull(largest_contour)
    
    return {"contour": largest_contour, "hull": hull, "bbox": (x, y, w, h)}

# IMPROVED: Better alignment and sizing of shoes with enhanced positioning
def align_shoes_improved(original_shape, new_shoe, new_shoe_mask, original_mask, background):
    # Extract dimensions from original shoe
    orig_x, orig_y, orig_w, orig_h = original_shape["bbox"]
    
    # Calculate the area of the original mask to better understand the shoe size
    orig_area = cv2.countNonZero(original_mask)
    new_area = cv2.countNonZero(new_shoe_mask)
    
    # Calculate scale factors based on both bounding box and area
    scale_area = np.sqrt(orig_area / new_area)
    scale_bbox_w = orig_w / new_shoe.shape[1]
    scale_bbox_h = orig_h / new_shoe.shape[0]
    
    # Take the maximum scale factor to ensure the shoe fills the space properly
    # Using a factor of 1.15 to ensure complete coverage
    scale = max(scale_area, scale_bbox_w, scale_bbox_h) * 1.15
    
    # Calculate new dimensions
    new_width = int(new_shoe.shape[1] * scale)
    new_height = int(new_shoe.shape[0] * scale)
    
    # Resize the new shoe and its mask
    resized_shoe = cv2.resize(new_shoe, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(new_shoe_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Calculate center of mass of original shoe mask for better positioning
    M = cv2.moments(original_mask)
    if M["m00"] != 0:
        orig_center_x = int(M["m10"] / M["m00"])
        orig_center_y = int(M["m01"] / M["m00"])
    else:
        # Fallback to center of bounding box
        orig_center_x = orig_x + orig_w // 2
        orig_center_y = orig_y + orig_h // 2
    
    # Get the orientation of the original shoe (if it's sideways, upright, etc.)
    # Calculate the minimum area rectangle
    rect = cv2.minAreaRect(original_shape["contour"])
    
    # Calculate placement coordinates to center the new shoe on the original shoe's center
    place_x = orig_center_x - new_width // 2
    place_y = orig_center_y - new_height // 2
    
    # Adjust placement to ensure the shoe's bottom aligns with the original
    # Get the bottom y-coordinate of the original shoe
    orig_bottom = orig_y + orig_h
    
    # Calculate the difference between bottoms and adjust y position
    bottom_diff = orig_bottom - (place_y + new_height)
    place_y += bottom_diff
    
    # Fine-tune positioning based on the angle of the shoe
    # If the original shoe is angled, we could use that information to align better
    angle = rect[2]
    if abs(angle) > 30:
        # The shoe is likely sideways or at an angle
        # Adjust horizontal positioning more than vertical
        place_x += int(orig_w * 0.05)  # Small adjustment based on angle
    
    # Ensure the shoe doesn't go out of bounds
    place_x = max(0, min(place_x, background.shape[1] - new_width))
    place_y = max(0, min(place_y, background.shape[0] - new_height))
    
    return resized_shoe, resized_mask, (place_x, place_y, new_width, new_height)

# ENHANCED: Improved advanced blending function for realistic integration
def advanced_blending(background, original_mask, new_shoe, new_shoe_mask, placement):
    result = background.copy()
    place_x, place_y, new_width, new_height = placement
    
    # First, create a dilated version of the original mask to find the transition region
    kernel = np.ones((15, 15), np.uint8)
    dilated_original_mask = cv2.dilate(original_mask, kernel, iterations=1)
    transition_region = cv2.subtract(dilated_original_mask, original_mask)
    
    # Create a background-only image (with the original shoe removed)
    background_mask = cv2.bitwise_not(original_mask)
    background_only = background.copy()
    for c in range(3):
        background_only[:,:,c] = cv2.bitwise_and(background[:,:,c], background_mask)
    
    # Create a new shoe layer
    new_shoe_layer = np.zeros_like(background)
    
    # Place the new shoe on the layer with anti-aliased edges
    new_shoe_placed_mask = np.zeros_like(original_mask)
    
    # Place the new shoe with soft edges
    for y in range(place_y, min(place_y + new_height, background.shape[0])):
        for x in range(place_x, min(place_x + new_width, background.shape[1])):
            rel_y = y - place_y
            rel_x = x - place_x
            
            if rel_y >= new_shoe.shape[0] or rel_x >= new_shoe.shape[1]:
                continue
                
            if new_shoe_mask[rel_y, rel_x] > 0:
                new_shoe_layer[y, x] = new_shoe[rel_y, rel_x]
                new_shoe_placed_mask[y, x] = 255
    
    # Apply color correction to match the background lighting conditions
    # Extract color statistics from the original image near the shoe
    color_sample_mask = cv2.dilate(original_mask, kernel, iterations=2)
    color_sample_mask = cv2.subtract(color_sample_mask, original_mask)
    
    # Get color statistics from the background around the original shoe
    bg_pixels = []
    for y in range(background.shape[0]):
        for x in range(background.shape[1]):
            if color_sample_mask[y, x] > 0:
                bg_pixels.append(background[y, x])
    
    if bg_pixels:
        bg_pixels = np.array(bg_pixels)
        bg_mean = np.mean(bg_pixels, axis=0)
        bg_std = np.std(bg_pixels, axis=0)
        
        # Get color statistics from the new shoe
        shoe_pixels = []
        for y in range(place_y, min(place_y + new_height, background.shape[0])):
            for x in range(place_x, min(place_x + new_width, background.shape[1])):
                rel_y = y - place_y
                rel_x = x - place_x
                
                if rel_y >= new_shoe.shape[0] or rel_x >= new_shoe.shape[1]:
                    continue
                    
                if new_shoe_mask[rel_y, rel_x] > 0:
                    shoe_pixels.append(new_shoe[rel_y, rel_x])
                    
        if shoe_pixels:
            shoe_pixels = np.array(shoe_pixels)
            shoe_mean = np.mean(shoe_pixels, axis=0)
            shoe_std = np.std(shoe_pixels, axis=0)
            
            # Adjust colors of the new shoe to match the background lighting
            for y in range(place_y, min(place_y + new_height, background.shape[0])):
                for x in range(place_x, min(place_x + new_width, background.shape[1])):
                    rel_y = y - place_y
                    rel_x = x - place_x
                    
                    if rel_y >= new_shoe.shape[0] or rel_x >= new_shoe.shape[1]:
                        continue
                        
                    if new_shoe_mask[rel_y, rel_x] > 0:
                        for c in range(3):
                            # Normalize the pixel value
                            normalized = (new_shoe_layer[y, x, c] - shoe_mean[c]) / (shoe_std[c] + 1e-6)
                            # Match the background statistics
                            new_shoe_layer[y, x, c] = np.clip(normalized * bg_std[c] + bg_mean[c], 0, 255)
    
    # Apply advanced blending with large feathering for realistic edges
    feather_size = 9  # Increased for super smooth blending
    feathered_mask = cv2.GaussianBlur(new_shoe_placed_mask, (feather_size*2+1, feather_size*2+1), 0)
    
    # Create the final result with advanced alpha blending
    for y in range(background.shape[0]):
        for x in range(background.shape[1]):
            alpha = feathered_mask[y, x] / 255.0
            
            # Enhanced blending
            if alpha > 0:
                for c in range(3):
                    result[y, x, c] = int((1 - alpha) * background_only[y, x, c] + alpha * new_shoe_layer[y, x, c])
    
    return result, new_shoe_placed_mask

# ENHANCED: Comprehensive shadow and lighting adjustment
def enhance_lighting_and_shadows(original_image, original_mask, result, final_mask):
    # Create a copy of the result for enhancing
    enhanced = result.copy()
    
    # 1. Extract shadow and lighting information from original image
    # Convert to LAB color space to separate lighting information
    original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    
    # Extract the L channel (lightness)
    original_l = original_lab[:,:,0]
    
    # Create a dilated mask to cover shadow areas around the shoe
    kernel = np.ones((25,25), np.uint8)
    shadow_region_mask = cv2.dilate(original_mask, kernel, iterations=2)
    shadow_region_mask = cv2.subtract(shadow_region_mask, original_mask)
    
    # Find the shadow areas (darker regions near the shoe)
    bg_pixels = original_l[shadow_region_mask > 0]
    if len(bg_pixels) > 0:
        bg_mean = np.mean(bg_pixels)
        shadow_threshold = bg_mean * 0.85  # Threshold for identifying shadows
        
        # Create shadow mask where pixels are darker than the threshold
        shadow_mask = np.zeros_like(original_l)
        shadow_mask[(original_l < shadow_threshold) & (shadow_region_mask > 0)] = 255
        
        # Refine the shadow mask
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
        
        # 2. Apply shadow to the result image where appropriate
        shadow_intensity = 0.7  # How dark the shadow should be
        
        # Convert result to LAB to modify lightness
        for y in range(enhanced.shape[0]):
            for x in range(enhanced.shape[1]):
                # Only apply shadows where:
                # 1. There is shadow in original image
                # 2. We're outside the new shoe
                if shadow_mask[y, x] > 0 and final_mask[y, x] == 0:
                    shadow_strength = shadow_mask[y, x] / 255.0 * 0.5
                    result_lab[y, x, 0] = np.clip(result_lab[y, x, 0] * (1 - shadow_strength * (1 - shadow_intensity)), 0, 255)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    # 3. Add reflection/highlight effects where appropriate
    # Create a highlight mask based on the original image
    highlight_mask = np.zeros_like(original_l)
    if len(bg_pixels) > 0:
        highlight_threshold = bg_mean * 1.15
        highlight_mask[(original_l > highlight_threshold) & (shadow_region_mask > 0)] = 255
        highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
        
        # Apply subtle highlights
        for y in range(enhanced.shape[0]):
            for x in range(enhanced.shape[1]):
                if highlight_mask[y, x] > 0 and final_mask[y, x] == 0:
                    highlight_strength = highlight_mask[y, x] / 255.0 * 0.3
                    for c in range(3):
                        enhanced[y, x, c] = np.clip(enhanced[y, x, c] * (1 + highlight_strength * 0.2), 0, 255)
    
    # 4. Apply environment color reflection to the shoe edges
    # This simulates how the shoe reflects the surrounding colors
    edge_kernel = np.ones((7, 7), np.uint8)
    edge_mask = cv2.dilate(final_mask, edge_kernel, iterations=1)
    edge_mask = cv2.subtract(edge_mask, final_mask)
    
    # Get average color of the background near the shoe
    bg_color = np.mean(original_image[shadow_region_mask > 0], axis=0) if np.any(shadow_region_mask > 0) else np.array([0, 0, 0])
    
    # Apply subtle color tinting to the edges of the shoe
    for y in range(enhanced.shape[0]):
        for x in range(enhanced.shape[1]):
            if edge_mask[y, x] > 0:
                edge_strength = 0.15  # Subtle effect
                for c in range(3):
                    enhanced[y, x, c] = np.clip(enhanced[y, x, c] * (1 - edge_strength) + bg_color[c] * edge_strength, 0, 255)
    
    # 5. Add a subtle ground shadow if the bottom of the shoe touches the ground
    # Detect the bottom edge of the shoe
    bottom_mask = np.zeros_like(final_mask)
    for x in range(final_mask.shape[1]):
        # Find the lowest white pixel in each column
        col = final_mask[:, x]
        white_pixels = np.where(col > 0)[0]
        if len(white_pixels) > 0:
            bottom_y = np.max(white_pixels)
            # Mark a few pixels below as the ground contact point
            for y in range(bottom_y, min(bottom_y + 20, final_mask.shape[0])):
                if y < final_mask.shape[0] and final_mask[y, x] == 0:  # Don't overwrite the shoe
                    bottom_mask[y, x] = 255
    
    # Blur the ground shadow for a soft effect
    ground_shadow = cv2.GaussianBlur(bottom_mask, (21, 7), 0)  # More horizontal blur
    
    # Apply the ground shadow
    shadow_factor = 0.7  # How dark the shadow should be
    for y in range(enhanced.shape[0]):
        for x in range(enhanced.shape[1]):
            if ground_shadow[y, x] > 0 and final_mask[y, x] == 0:
                shadow_strength = ground_shadow[y, x] / 255.0 * 0.4  # Control shadow intensity
                for c in range(3):
                    enhanced[y, x, c] = np.clip(enhanced[y, x, c] * (1 - shadow_strength * (1 - shadow_factor)), 0, 255)
    
    return enhanced

# Add final polish with high-quality vignetting and tone adjustment
def final_polish(image):
    # 1. Apply subtle vignette
    rows, cols = image.shape[:2]
    
    # Create a radial gradient for vignette
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    vignette = 255 * kernel / np.linalg.norm(kernel)
    
    # Apply the vignette with a subtle effect
    polished = image.copy()
    for i in range(3):
        polished[:,:,i] = polished[:,:,i] * 0.85 + polished[:,:,i] * vignette * 0.15 / 255
    
    # 2. Apply color grading with tone mapping
    # Convert to HSV for easier color adjustments
    hsv = cv2.cvtColor(polished, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Increase vibrance slightly
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.1, 0, 255)
    
    # Enhance contrast in the value channel
    v = hsv[:,:,2]
    v_mean = np.mean(v)
    v = np.clip((v - v_mean) * 1.05 + v_mean, 0, 255)
    hsv[:,:,2] = v
    
    # Convert back to BGR
    polished = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 3. Apply subtle denoising for a cleaner look
    polished = cv2.fastNlMeansDenoisingColored(polished, None, 3, 3, 7, 21)
    
    # 4. Apply very subtle sharpening for details
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]]) * 0.5 + np.array([[0, 0, 0],
                                                        [0, 1, 0],
                                                        [0, 0, 0]]) * 0.5
    polished = cv2.filter2D(polished, -1, kernel)
    
    return polished

# Main function to perform the shoe replacement with all enhancements
def replace_shoe_in_image_enhanced(background_path, new_shoe_path, output_path):
    # Load the SAM model
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    
    # Get masks and images
    print("Creating mask for original shoe...")
    original_mask, background = create_shoes_mask_sam(background_path, predictor)
    
    print("Creating mask for new shoe...")
    new_shoe_mask, new_shoe = create_new_shoe_mask(new_shoe_path, predictor)
    
    # Extract shape information from the masks
    print("Analyzing shoe shapes...")
    original_shape = extract_shoe_shape(original_mask)
    
    # Use the improved alignment function
    print("Aligning shoes with improved positioning...")
    aligned_shoe, aligned_mask, placement = align_shoes_improved(
        original_shape, new_shoe, new_shoe_mask, original_mask, background
    )
    
    # Replace the original shoe with the new shoe using advanced blending
    print("Replacing shoe with advanced blending...")
    basic_result, final_mask = advanced_blending(background, original_mask, aligned_shoe, aligned_mask, placement)
    
    # Enhance lighting and shadows
    print("Enhancing lighting and shadows...")
    lighting_enhanced = enhance_lighting_and_shadows(background, original_mask, basic_result, final_mask)
    
    # Apply final polish
    print("Applying final polish...")
    final_result = final_polish(lighting_enhanced)
    
    # Save the result
    print("Saving final image...")
    cv2.imwrite(output_path, final_result)
    
    # Save intermediate results for analysis
    stages_path = output_path.replace('.jpg', '_stages.jpg')
    
    # Create a visualization of the various stages
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Original Shoe Mask')
    plt.imshow(original_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('New Shoe')
    plt.imshow(cv2.cvtColor(new_shoe, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Basic Replacement')
    plt.imshow(cv2.cvtColor(basic_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title('Lighting Enhanced')
    plt.imshow(cv2.cvtColor(lighting_enhanced, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title('Final Result')
    plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(stages_path)
    plt.close()
    
    return final_result

# Google Colab usage example
if __name__ == "__main__":
    print("Please upload your background image (with original shoe):")
    background_files = upload_images()
    background_path = next(iter(background_files))
    
    print("\nPlease upload your new shoe image:")
    new_shoe_files = upload_images()
    new_shoe_path = next(iter(new_shoe_files))
    
    output_path = "shoe_replacement_final.jpg"
    
    try:
        # Perform the shoe replacement with all enhancements
        final_image = replace_shoe_in_image_enhanced(background_path, new_shoe_path, output_path)
        
        print(f"\nComplete! The results are saved as:")
        print(f"- Final enhanced result: {output_path}")
        print(f"- Process stages: {output_path.replace('.jpg', '_stages.jpg')}")
        print("\nYou can download the results using:")
        print("from google.colab import files")
        print(f"files.download('{output_path}')")
        print(f"files.download('{output_path.replace('.jpg', '_stages.jpg')}')")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()