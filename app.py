import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
import io
import torch
from scipy.ndimage import gaussian_filter

# Set page configuration
st.set_page_config(
    page_title="Shoe Editor Pro",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for a more professional appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #333;
    }
    .editor-panel {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .color-preview {
        width: 30px;
        height: 30px;
        display: inline-block;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'new_shoe_image' not in st.session_state:
    st.session_state.new_shoe_image = None
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'shoe_mask' not in st.session_state:
    st.session_state.shoe_mask = None
if 'bg_mask' not in st.session_state:
    st.session_state.bg_mask = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Replace Shoe"

# Load SAM model if available
# Load SAM model if available
@st.cache_resource
def load_sam_model():
    try:
        # Check if segment-anything is installed
        import segment_anything
        from segment_anything import sam_model_registry, SamPredictor
        
        # Load the model with the specified weights
        model_path = r"C:\Users\abhis\OneDrive\Documents\AI manipulation\sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        predictor = SamPredictor(sam)
        return predictor
    except ImportError:
        st.error("Segment Anything Model (SAM) not installed. Install with: pip install segment-anything")
        return None
    except FileNotFoundError:
        st.error(f"SAM model weights not found at the specified path. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading SAM model: {str(e)}")
        return None

# Function to create a mask using SAM (simplified for demonstration)
# Function to create a mask using SAM
def create_mask_sam(image, predictor):
    # If we have the SAM predictor, use it
    if predictor is not None:
        try:
            # Set the image for the predictor
            predictor.set_image(image)
            
            # For automatic mask generation, we'll use the center point as a prompt
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Generate masks using point prompts
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])  # 1 indicates a foreground point
            
            # Predict masks
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Select the mask with the highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx].astype(np.uint8) * 255
            
            return mask
        except Exception as e:
            st.error(f"Error using SAM: {str(e)}")
            # Fall back to the simple method below
    
    # Fallback method (same as before)
    if len(image.shape) == 3 and image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define a color range for common shoe colors (adjust as needed)
        lower_bound = np.array([0, 20, 20])
        upper_bound = np.array([180, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find the largest contour which should be the shoe
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Create a clean mask with just the largest contour
            clean_mask = np.zeros_like(mask)
            cv2.drawContours(clean_mask, [largest_contour], 0, 255, -1)
            return clean_mask
    
    # Fallback to a simple center mask if all else fails
    h, w = image.shape[:2]
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    # Create an elliptical mask in the center
    cv2.ellipse(center_mask, (center_x, center_y), (w//3, h//2), 0, 0, 360, 255, -1)
    return center_mask

# Function to change shoe color
# Function to change shoe color
def change_shoe_color(image, mask, result_image, hue_shift=0, saturation_scale=1.0, value_scale=1.0):
    # Convert to HSV for easier color manipulation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Apply mask to identify shoe pixels
    mask_3d = np.stack([mask/255]*3, axis=2)
    
    # Apply hue shift (0-179 in OpenCV)
    hsv_image[:,:,0] = (hsv_image[:,:,0] + hue_shift) % 180
    
    # Apply saturation scaling (0-255 in OpenCV)
    hsv_image[:,:,1] = np.clip(hsv_image[:,:,1] * saturation_scale, 0, 255)
    
    # Apply value/brightness scaling (0-255 in OpenCV)
    hsv_image[:,:,2] = np.clip(hsv_image[:,:,2] * value_scale, 0, 255)
    
    # Convert back to RGB
    modified_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Create a result by blending the modified shoe with the CURRENT result image's background
    # This is the key change - use result_image instead of image for the background
    result = result_image.copy()
    result = result * (1 - mask_3d) + modified_image * mask_3d
    
    return result.astype(np.uint8)

# Function to apply preset color schemes
# Function to apply preset color schemes
def apply_color_preset(image, mask, result_image, preset_name):
    presets = {
        "Red": {"hue": 0, "saturation": 1.5, "value": 1.0},
        "Blue": {"hue": 120, "saturation": 1.5, "value": 1.0},
        "Green": {"hue": 60, "saturation": 1.5, "value": 1.0},
        "Yellow": {"hue": 30, "saturation": 1.5, "value": 1.0},
        "Purple": {"hue": 150, "saturation": 1.5, "value": 1.0},
        "Black": {"hue": 0, "saturation": 0.0, "value": 0.5},
        "White": {"hue": 0, "saturation": 0.0, "value": 1.5}
    }
    
    if preset_name in presets:
        preset = presets[preset_name]
        return change_shoe_color(
            image, 
            mask,
            result_image,  # Pass the current result image
            hue_shift=preset["hue"], 
            saturation_scale=preset["saturation"], 
            value_scale=preset["value"]
        )
    return result_image  # Return result_image instead of image

# Function to change background
def change_background(image, shoe_mask, new_bg):
    # Resize background to match image dimensions
    new_bg_resized = cv2.resize(new_bg, (image.shape[1], image.shape[0]))
    
    # Create inverse mask for background
    bg_mask = 255 - shoe_mask
    bg_mask_3d = np.stack([bg_mask/255]*3, axis=2)
    shoe_mask_3d = np.stack([shoe_mask/255]*3, axis=2)
    
    # Blend the images
    result = image * shoe_mask_3d + new_bg_resized * bg_mask_3d
    
    return result.astype(np.uint8)

# Function to apply textures to shoe
def apply_texture(image, mask, texture_image):
    # Resize texture to match image dimensions
    texture_resized = cv2.resize(texture_image, (image.shape[1], image.shape[0]))
    
    # Create 3D mask
    mask_3d = np.stack([mask/255]*3, axis=2)
    
    # Apply texture using multiplicative blending (preserves shadows and highlights)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image_3d = np.stack([gray_image]*3, axis=2) / 255.0
    
    # Blend texture with original image's luminance
    textured_shoe = (texture_resized.astype(np.float32) * gray_image_3d).astype(np.uint8)
    
    # Create a result by blending the textured shoe with the original background
    result = image.copy()
    result = result * (1 - mask_3d) + textured_shoe * mask_3d
    
    return result.astype(np.uint8)

# Function to add shadows or highlights
def add_lighting_effect(image, mask, effect_type="shadow", intensity=0.5):
    result = image.copy()
    mask_3d = np.stack([mask/255]*3, axis=2)
    
    if effect_type == "shadow":
        # Create a darkened version of the image
        darkened = (image.astype(np.float32) * (1 - intensity)).astype(np.uint8)
        
        # Create shadow by dilating the mask and subtracting original mask
        kernel = np.ones((15, 15), np.uint8)
        shadow_region = cv2.dilate(mask, kernel, iterations=1)
        shadow_region = cv2.subtract(shadow_region, mask)
        shadow_mask_3d = np.stack([shadow_region/255]*3, axis=2) * intensity
        
        # Apply shadow to result
        result = result * (1 - shadow_mask_3d) + darkened * shadow_mask_3d
        
    elif effect_type == "highlight":
        # Create a brightened version of the image
        brightened = np.clip(image.astype(np.float32) * (1 + intensity), 0, 255).astype(np.uint8)
        
        # Apply highlight to shoe edges
        kernel = np.ones((5, 5), np.uint8)
        highlight_region = cv2.dilate(mask, kernel, iterations=1)
        highlight_region = cv2.subtract(highlight_region, cv2.erode(mask, kernel, iterations=1))
        highlight_mask_3d = np.stack([highlight_region/255]*3, axis=2) * intensity
        
        # Apply highlight to result
        result = result * (1 - highlight_mask_3d) + brightened * highlight_mask_3d
    
    return result.astype(np.uint8)

# Function to add filters (like Instagram)
def apply_filter(image, filter_name):
    result = image.copy()
    
    if filter_name == "Vintage":
        # Sepia-like effect
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        result = cv2.transform(result, sepia_kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    elif filter_name == "Cool":
        # Blue tint
        result[:,:,0] = np.clip(result[:,:,0] * 0.8, 0, 255).astype(np.uint8)  # Reduce red
        result[:,:,2] = np.clip(result[:,:,2] * 1.2, 0, 255).astype(np.uint8)  # Enhance blue
        
    elif filter_name == "Warm":
        # Warm tint
        result[:,:,0] = np.clip(result[:,:,0] * 1.2, 0, 255).astype(np.uint8)  # Enhance red
        result[:,:,2] = np.clip(result[:,:,2] * 0.8, 0, 255).astype(np.uint8)  # Reduce blue
        
    elif filter_name == "High Contrast":
        # Increase contrast
        result = result.astype(np.float32)
        mean = np.mean(result, axis=(0, 1))
        result = (result - mean) * 1.5 + mean
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

# Main interface
def main():
    # Header
    st.title("👟 Shoe Editor Pro")
    st.markdown("Edit your shoes with professional tools - change colors, backgrounds, and more!")
    
    # Sidebar for main controls
    with st.sidebar:
        st.header("Main Controls")
        
        # Mode selection
        mode = st.radio(
            "Select Mode", 
            ["Replace Shoe", "Edit Existing Shoe"],
            index=0 if st.session_state.current_mode == "Replace Shoe" else 1
        )
        st.session_state.current_mode = mode
        
        # File uploaders
        st.subheader("Upload Images")
        background_file = st.file_uploader("Upload Background Image (with original shoe)", type=["jpg", "jpeg", "png"])
        
        if st.session_state.current_mode == "Replace Shoe":
            new_shoe_file = st.file_uploader("Upload New Shoe Image", type=["jpg", "jpeg", "png"])
        
        # Process button
        process_button = st.button("Process Images")
        
        # Reset button
        if st.button("Reset All"):
            st.session_state.original_image = None
            st.session_state.new_shoe_image = None
            st.session_state.result_image = None
            st.session_state.shoe_mask = None
            st.session_state.bg_mask = None
            st.session_state.processed = False
            st.experimental_rerun()
        
        # Download result
        if st.session_state.result_image is not None:
            result_pil = Image.fromarray(st.session_state.result_image)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="Download Result",
                data=buf.getvalue(),
                file_name="shoe_edit_result.png",
                mime="image/png"
            )
    
    # Process images when button is clicked
    if background_file is not None:
        # Load background image
        background_pil = Image.open(background_file).convert('RGB')
        background_np = np.array(background_pil)
        st.session_state.original_image = background_np
        
        # Load new shoe if in replace mode
        if st.session_state.current_mode == "Replace Shoe" and new_shoe_file is not None:
            new_shoe_pil = Image.open(new_shoe_file).convert('RGB')
            new_shoe_np = np.array(new_shoe_pil)
            st.session_state.new_shoe_image = new_shoe_np
        
        # Process images when button is clicked
        if process_button:
            with st.spinner("Processing images..."):
                # Load SAM model (placeholder for demonstration)
                sam_model = load_sam_model()
                
                # Create mask for original shoe
                if st.session_state.original_image is not None:
                    shoe_mask = create_mask_sam(st.session_state.original_image, sam_model)
                    st.session_state.shoe_mask = shoe_mask
                    st.session_state.bg_mask = 255 - shoe_mask
                    
                    # Set result image to original initially
                    st.session_state.result_image = st.session_state.original_image.copy()
                    st.session_state.processed = True
    
    # Main content area with two columns
    if st.session_state.processed:
        col1, col2 = st.columns([1, 1])
        
        # Left column: Original and processed images
        with col1:
            st.subheader("Images")
            
            # Show original image
            st.markdown("**Original Image**")
            st.image(st.session_state.original_image, use_column_width=True)
            
            # Show current result
            st.markdown("**Current Result**")
            st.image(st.session_state.result_image, use_column_width=True)
            
            # Show masks for debugging (can be removed in production)
            with st.expander("Show Masks (Debug)"):
                st.image(st.session_state.shoe_mask, caption="Shoe Mask", use_column_width=True)
        
        # Right column: Editing tools
        with col2:
            st.subheader("Editing Tools")
            
            # Create tabs for different editing options
            tabs = st.tabs(["Color", "Background", "Effects", "Textures", "Filters"])
            
            # Color editing tab
            with tabs[0]:
                st.markdown("### Color Editor")
                
                # Color presets
                st.markdown("#### Quick Color Presets")
                color_cols = st.columns(4)
                preset_colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Black", "White"]
                
                # Create color swatch buttons
# Create color swatch buttons
                for i, color in enumerate(preset_colors):
                    col_idx = i % 4
                    with color_cols[col_idx]:
                        if st.button(color, key=f"preset_{color}"):
                            if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                                st.session_state.result_image = apply_color_preset(
                                    st.session_state.original_image, 
                                    st.session_state.shoe_mask,
                                    st.session_state.result_image,  # Pass the current result image
                                    color
                                )
                
                # Custom color controls
                st.markdown("#### Custom Color Adjustment")
                hue_shift = st.slider("Hue", 0, 180, 0)
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0)
                brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
                
                if st.button("Apply Color Changes"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        st.session_state.result_image = change_shoe_color(
                            st.session_state.original_image,
                            st.session_state.shoe_mask,
                            st.session_state.result_image,  # Pass the current result image
                            hue_shift,
                            saturation,
                            brightness
                        )
            
            # Background editing tab
            with tabs[1]:
                st.markdown("### Background Editor")
                
                # Background color option
                st.markdown("#### Solid Color Background")
                bg_color = st.color_picker("Choose background color", "#f0f0f0")
                
                if st.button("Apply Solid Background"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        # Create solid color background
                        r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
                        solid_bg = np.ones_like(st.session_state.original_image) * np.array([r, g, b])
                        
                        # Apply background change
                        st.session_state.result_image = change_background(
                            st.session_state.original_image,
                            st.session_state.shoe_mask,
                            solid_bg.astype(np.uint8)
                        )
                
                # Gradient background option
                st.markdown("#### Gradient Background")
                grad_color1 = st.color_picker("Top color", "#87CEEB")
                grad_color2 = st.color_picker("Bottom color", "#4682B4")
                
                if st.button("Apply Gradient Background"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        # Create gradient background
                        h, w = st.session_state.original_image.shape[:2]
                        gradient = np.zeros((h, w, 3), dtype=np.uint8)
                        
                        r1, g1, b1 = int(grad_color1[1:3], 16), int(grad_color1[3:5], 16), int(grad_color1[5:7], 16)
                        r2, g2, b2 = int(grad_color2[1:3], 16), int(grad_color2[3:5], 16), int(grad_color2[5:7], 16)
                        
                        # Create gradient
                        for y in range(h):
                            ratio = y / h
                            gradient[y, :, 0] = int(r1 * (1 - ratio) + r2 * ratio)
                            gradient[y, :, 1] = int(g1 * (1 - ratio) + g2 * ratio)
                            gradient[y, :, 2] = int(b1 * (1 - ratio) + b2 * ratio)
                        
                        # Apply background change
                        st.session_state.result_image = change_background(
                            st.session_state.original_image,
                            st.session_state.shoe_mask,
                            gradient
                        )
                
                # Custom background image
                st.markdown("#### Custom Background Image")
                bg_file = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
                
                if bg_file is not None and st.button("Apply Custom Background"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        # Load background image
                        bg_pil = Image.open(bg_file).convert('RGB')
                        bg_np = np.array(bg_pil)
                        
                        # Apply background change
                        st.session_state.result_image = change_background(
                            st.session_state.original_image,
                            st.session_state.shoe_mask,
                            bg_np
                        )
            
            # Effects tab
            with tabs[2]:
                st.markdown("### Effects")
                
                # Shadow effects
                st.markdown("#### Shadow Effects")
                shadow_intensity = st.slider("Shadow Intensity", 0.0, 1.0, 0.5)
                
                if st.button("Add Shadow"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        st.session_state.result_image = add_lighting_effect(
                            st.session_state.result_image,
                            st.session_state.shoe_mask,
                            "shadow",
                            shadow_intensity
                        )
                
                # Highlight effects
                st.markdown("#### Highlight Effects")
                highlight_intensity = st.slider("Highlight Intensity", 0.0, 1.0, 0.3)
                
                if st.button("Add Highlight"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        st.session_state.result_image = add_lighting_effect(
                            st.session_state.result_image,
                            st.session_state.shoe_mask,
                            "highlight",
                            highlight_intensity
                        )
                
                # Blur background
                st.markdown("#### Background Blur")
                blur_amount = st.slider("Blur Amount", 0, 30, 0)
                
                if st.button("Apply Background Blur"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        # Create a blurred version of the background
                        blurred_bg = cv2.GaussianBlur(st.session_state.result_image, (blur_amount*2+1, blur_amount*2+1), 0)
                        
                        # Apply blur to background only
                        shoe_mask_3d = np.stack([st.session_state.shoe_mask/255]*3, axis=2)
                        st.session_state.result_image = st.session_state.result_image * shoe_mask_3d + blurred_bg * (1 - shoe_mask_3d)
            
            # Textures tab
            with tabs[3]:
                st.markdown("### Textures")
                
                # Preset texture options
                texture_options = ["Leather", "Canvas", "Metal", "Plastic", "Suede"]
                selected_texture = st.selectbox("Select Texture", texture_options)
                
                if st.button("Apply Texture"):
                    if st.session_state.shoe_mask is not None and st.session_state.result_image is not None:
                        # Create a simple texture pattern (in a real app, you would load actual textures)
                        h, w = st.session_state.original_image.shape[:2]
                        texture = np.zeros((h, w, 3), dtype=np.uint8)
                        
                        if selected_texture == "Leather":
                            # Simulate leather texture with noise
                            noise = np.random.randint(150, 200, (h, w)).astype(np.uint8)
                            texture[:,:,0] = noise
                            texture[:,:,1] = (noise * 0.8).astype(np.uint8)
                            texture[:,:,2] = (noise * 0.6).astype(np.uint8)
                        
                        elif selected_texture == "Canvas":
                            # Simulate canvas with grid pattern
                            texture.fill(200)
                            for i in range(0, h, 4):
                                texture[i:i+1, :, :] = 180
                            for j in range(0, w, 4):
                                texture[:, j:j+1, :] = 180
                        
                        elif selected_texture == "Metal":
                            # Simulate metallic texture
                            base = np.ones((h, w)) * 200
                            for i in range(h):
                                base[i, :] = 150 + 100 * np.sin(i/10)
                            texture[:,:,0] = texture[:,:,1] = texture[:,:,2] = base.astype(np.uint8)
                        
                        elif selected_texture == "Plastic":
                            # Simulate plastic with smooth gradient
                            texture.fill(200)
                            for i in range(h):
                                brightness = 150 + 100 * np.sin(i/50)**2
                                texture[i,:,:] = brightness
                        
                        elif selected_texture == "Suede":
                            # Simulate suede with fine noise
                            base = np.random.randint(100, 130, (h, w))
                            # Apply slight blur to simulate suede
                            base = gaussian_filter(base, sigma=1)
                            texture[:,:,0] = (base * 1.2).clip(0, 255).astype(np.uint8)
                            texture[:,:,1] = base.astype(np.uint8)
                            texture[:,:,2] = (base * 0.8).clip(0, 255).astype(np.uint8)
                        
                        # Apply texture
                        st.session_state.result_image = apply_texture(
                            st.session_state.result_image,
                            st.session_state.shoe_mask,
                            texture
                        )
            
            # Filters tab
            with tabs[4]:
                st.markdown("### Filters")
                
                # Filter options
                filter_options = ["None", "Vintage", "Cool", "Warm", "High Contrast"]
                selected_filter = st.selectbox("Select Filter", filter_options)
                
                if st.button("Apply Filter"):
                    if selected_filter != "None" and st.session_state.result_image is not None:
                        st.session_state.result_image = apply_filter(
                            st.session_state.result_image,
                            selected_filter
                        )

if __name__ == "__main__":
    main()