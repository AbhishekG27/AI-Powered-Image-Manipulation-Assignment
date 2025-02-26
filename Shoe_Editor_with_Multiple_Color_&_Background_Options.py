import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
import colorsys

# Assuming SAM is installed and imported as in your original code
# from segment_anything import sam_model_registry, SamPredictor

class ShoeImageEditor:
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth"):
        """Initialize the shoe editor with SAM model."""
        from segment_anything import sam_model_registry, SamPredictor
        
        # Load the SAM model
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        
        # Store original images, masks, and edited versions
        self.original_shoe = None
        self.shoe_mask = None
        self.available_backgrounds = {}
        self.available_colors = {}
        self.current_edit = None
    
    def load_shoe_image(self, image_path):
        """Load a shoe image and create its mask."""
        # Load the image
        self.original_shoe = cv2.imread(image_path)
        if self.original_shoe is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Create shoe mask
        self.shoe_mask = self.create_mask(self.original_shoe)
        
        # Initialize current edit
        self.current_edit = self.original_shoe.copy()
        
        return self.original_shoe, self.shoe_mask
    
    def create_mask(self, image):
        """Create a mask for the shoe using SAM."""
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in the predictor
        self.predictor.set_image(image_rgb)
        
        # For shoes, we typically want to focus on the center of the image
        h, w = image.shape[:2]
        center_point = np.array([[w//2, h//2]])
        
        # Get masks from the center point
        masks, scores, logits = self.predictor.predict(
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
        
        return shoe_mask
    
    def add_background(self, name, background_image):
        """Add a background option."""
        self.available_backgrounds[name] = background_image
    
    def add_color_option(self, name, color):
        """
        Add a color option for the shoe.
        Color can be a name (string) or RGB tuple.
        """
        if isinstance(color, str):
            # Convert color name to RGB
            color_map = {
                "red": (0, 0, 255),     # BGR format
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "purple": (255, 0, 255),
                "white": (255, 255, 255),
                "black": (0, 0, 0)
            }
            if color.lower() in color_map:
                color = color_map[color.lower()]
            else:
                raise ValueError(f"Unknown color name: {color}")
        
        self.available_colors[name] = color
    
    def change_shoe_color(self, color_name):
        """Change the color of the shoe."""
        if color_name not in self.available_colors:
            raise ValueError(f"Color '{color_name}' not found in available colors")
        
        # Get the target color
        target_color = self.available_colors[color_name]
        
        # Start with a copy of the original
        colored_shoe = self.original_shoe.copy()
        
        # Convert to HSV for easier color manipulation
        hsv_shoe = cv2.cvtColor(colored_shoe, cv2.COLOR_BGR2HSV)
        
        # Get the target color in HSV
        target_bgr = np.uint8([[target_color]])
        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
        target_hue = target_hsv[0, 0, 0]
        
        # Apply the shoe mask
        mask = self.shoe_mask > 0
        
        # Modify only the hue of pixels inside the mask
        # This preserves the original texture, shadows, and highlights
        hsv_shoe_flat = hsv_shoe.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        
        # Modify the hue while preserving saturation and value
        hsv_shoe_flat[mask_flat, 0] = target_hue
        
        # Increase saturation a bit for vibrant colors
        hsv_shoe_flat[mask_flat, 1] = np.clip(hsv_shoe_flat[mask_flat, 1] * 1.2, 0, 255)
        
        # Reshape back
        hsv_shoe = hsv_shoe_flat.reshape(hsv_shoe.shape)
        
        # Convert back to BGR
        colored_shoe = cv2.cvtColor(hsv_shoe, cv2.COLOR_HSV2BGR)
        
        # Apply advanced blending to make the color change look natural
        self.current_edit = self._blend_color_naturally(colored_shoe)
        
        return self.current_edit
    
    def change_background(self, background_name):
        """Change the background of the image."""
        if background_name not in self.available_backgrounds:
            raise ValueError(f"Background '{background_name}' not found in available backgrounds")
        
        background = self.available_backgrounds[background_name].copy()
        
        # Resize background to match shoe image if needed
        if background.shape[:2] != self.original_shoe.shape[:2]:
            background = cv2.resize(background, 
                                   (self.original_shoe.shape[1], self.original_shoe.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
        
        # Create inverse mask for background
        inv_mask = cv2.bitwise_not(self.shoe_mask)
        
        # Extract the current shoe (with its current color)
        shoe_only = self.current_edit.copy()
        for c in range(3):
            shoe_only[:,:,c] = cv2.bitwise_and(shoe_only[:,:,c], self.shoe_mask)
        
        # Place background using the inverse mask
        for c in range(3):
            background[:,:,c] = cv2.bitwise_and(background[:,:,c], inv_mask)
        
        # Combine shoe with new background
        result = cv2.add(shoe_only, background)
        
        # Add realistic shadows
        result = self._add_realistic_shadows(result)
        
        self.current_edit = result
        return result
    
    def _blend_color_naturally(self, colored_shoe):
        """Blend the new color naturally to preserve texture and details."""
        result = self.original_shoe.copy()
        
        # Get luminance from original image to preserve lighting
        orig_gray = cv2.cvtColor(self.original_shoe, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(colored_shoe, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference to preserve highlights and shadows
        lum_factor = np.ones_like(orig_gray, dtype=np.float32)
        
        # Avoid division by zero
        mask = target_gray > 10
        if np.any(mask):
            lum_factor[mask] = orig_gray[mask].astype(np.float32) / target_gray[mask].astype(np.float32)
        
        # Apply a soft limit to the factor to avoid extreme values
        lum_factor = np.clip(lum_factor, 0.5, 1.5)
        
        # Apply the luminance factor to each channel
        for c in range(3):
            adjusted = colored_shoe[:,:,c].astype(np.float32) * lum_factor
            colored_shoe[:,:,c] = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        # Blend with original for realism (preserve some original texture)
        alpha = 0.8  # Blend factor
        mask_3d = np.dstack([self.shoe_mask, self.shoe_mask, self.shoe_mask]) > 0
        result[mask_3d] = (alpha * colored_shoe[mask_3d] + (1-alpha) * self.original_shoe[mask_3d]).astype(np.uint8)
        
        return result
    
    def _add_realistic_shadows(self, image):
        """Add realistic shadows to the shoe on the background."""
        result = image.copy()
        
        # Create a shadow mask under the shoe
        kernel = np.ones((15, 15), np.uint8)
        shadow_mask = cv2.dilate(self.shoe_mask, kernel, iterations=1)
        shadow_mask = cv2.subtract(shadow_mask, self.shoe_mask)
        
        # Apply a Gaussian blur to soften the shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        # Darken the shadow area
        shadow_factor = 0.7  # Darkness of shadow (lower = darker)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if shadow_mask[y, x] > 0:
                    shadow_strength = shadow_mask[y, x] / 255.0 * 0.5
                    for c in range(3):
                        result[y, x, c] = np.clip(result[y, x, c] * (1 - shadow_strength * (1 - shadow_factor)), 0, 255)
        
        return result
    
    def apply_material(self, material_type):
        """Change the material of the shoe (leather, suede, patent, etc.)."""
        result = self.current_edit.copy()
        
        # Different material types have different texture and reflection properties
        if material_type == "leather":
            # Leather has medium shine and subtle texture
            result = self._adjust_texture(result, smoothness=0.3, shine=0.4)
        elif material_type == "suede":
            # Suede has no shine and high texture
            result = self._adjust_texture(result, smoothness=0.8, shine=0.1)
        elif material_type == "patent":
            # Patent leather has high shine and no texture
            result = self._adjust_texture(result, smoothness=0.1, shine=0.8)
        elif material_type == "canvas":
            # Canvas has high texture and no shine
            result = self._adjust_texture(result, smoothness=0.9, shine=0.1)
        else:
            raise ValueError(f"Unknown material type: {material_type}")
        
        self.current_edit = result
        return result
    
    def _adjust_texture(self, image, smoothness=0.5, shine=0.5):
        """Adjust texture and shine of the shoe."""
        result = image.copy()
        
        # Apply to shoe area only
        mask = self.shoe_mask > 0
        
        # Convert to HSV to adjust brightness (V channel)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        
        # Add texture by adding noise proportional to smoothness parameter
        if smoothness > 0:
            # Create noise with intensity based on smoothness
            noise = np.random.normal(0, 20 * smoothness, size=hsv.shape[:2])
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            
            # Add noise to the value channel
            v_channel = hsv[:,:,2].astype(np.float32)
            v_channel[mask] += noise[mask]
            hsv[:,:,2] = np.clip(v_channel, 0, 255).astype(np.uint8)
        
        # Add shine by enhancing certain areas
        if shine > 0:
            # Create a gradient to simulate light reflection
            h, w = hsv.shape[:2]
            y, x = np.mgrid[0:h, 0:w]
            
            # Create simulated light position at top-right
            light_y, light_x = h * 0.2, w * 0.8
            
            # Calculate distance from each pixel to light
            dist = np.sqrt((y - light_y)**2 + (x - light_x)**2)
            max_dist = np.sqrt(h**2 + w**2)
            
            # Normalize distances and invert (closer = brighter)
            norm_dist = 1 - (dist / max_dist)
            
            # Apply highlight based on shine parameter
            highlight = norm_dist * 50 * shine
            highlight = cv2.GaussianBlur(highlight, (21, 21), 0)
            
            # Add highlight to value channel
            v_channel = hsv[:,:,2].astype(np.float32)
            v_channel[mask] += highlight[mask]
            hsv[:,:,2] = np.clip(v_channel, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def generate_color_variants(self, color_names=None):
        """Generate and return variants of the shoe in different colors."""
        if color_names is None:
            color_names = list(self.available_colors.keys())
        
        variants = {}
        # Save current edit state
        temp_edit = self.current_edit.copy()
        
        for color_name in color_names:
            # Change the color
            colored_shoe = self.change_shoe_color(color_name)
            variants[color_name] = colored_shoe.copy()
        
        # Restore original edit
        self.current_edit = temp_edit
        
        return variants
    
    def generate_background_variants(self, background_names=None):
        """Generate and return variants of the shoe with different backgrounds."""
        if background_names is None:
            background_names = list(self.available_backgrounds.keys())
        
        variants = {}
        # Save current edit state
        temp_edit = self.current_edit.copy()
        
        for bg_name in background_names:
            # Change the background
            bg_shoe = self.change_background(bg_name)
            variants[bg_name] = bg_shoe.copy()
        
        # Restore original edit
        self.current_edit = temp_edit
        
        return variants
    
    def generate_material_variants(self, materials=None):
        """Generate and return variants of the shoe with different materials."""
        if materials is None:
            materials = ["leather", "suede", "patent", "canvas"]
        
        variants = {}
        # Save current edit state
        temp_edit = self.current_edit.copy()
        
        for material in materials:
            # Change the material
            mat_shoe = self.apply_material(material)
            variants[material] = mat_shoe.copy()
        
        # Restore original edit
        self.current_edit = temp_edit
        
        return variants
    
    def create_grid_display(self, variants, title, cols=3):
        """Create a grid display of variants."""
        n = len(variants)
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(cols * 5, rows * 5))
        plt.suptitle(title, fontsize=16)
        
        for i, (name, img) in enumerate(variants.items()):
            plt.subplot(rows, cols, i + 1)
            plt.title(name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        plt.tight_layout()
        return plt.gcf()
    
    def save_current_edit(self, output_path):
        """Save the current edit to a file."""
        cv2.imwrite(output_path, self.current_edit)
        return output_path
    
    def save_all_variants(self, output_dir, prefix="shoe"):
        """Save all color, background, and material variants."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save color variants
        color_variants = self.generate_color_variants()
        for color_name, img in color_variants.items():
            path = os.path.join(output_dir, f"{prefix}_color_{color_name}.jpg")
            cv2.imwrite(path, img)
        
        # Save background variants
        bg_variants = self.generate_background_variants()
        for bg_name, img in bg_variants.items():
            path = os.path.join(output_dir, f"{prefix}_bg_{bg_name}.jpg")
            cv2.imwrite(path, img)
        
        # Save material variants
        mat_variants = self.generate_material_variants()
        for mat_name, img in mat_variants.items():
            path = os.path.join(output_dir, f"{prefix}_material_{mat_name}.jpg")
            cv2.imwrite(path, img)
        
        return output_dir
    
    def create_catalog_view(self, output_path="shoe_catalog.jpg"):
        """Create a catalog view with multiple variants like an e-commerce site."""
        # Generate variants
        color_variants = self.generate_color_variants()
        bg_variants = self.generate_background_variants()
        mat_variants = self.generate_material_variants()
        
        # Create a grid layout
        rows, cols = 3, 4  # 3 rows (colors, backgrounds, materials), 4 items each
        plt.figure(figsize=(cols * 4, rows * 4))
        
        # Add color variants on first row
        for i, (name, img) in enumerate(list(color_variants.items())[:cols]):
            plt.subplot(rows, cols, i + 1)
            plt.title(f"Color: {name}")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        # Add background variants on second row
        for i, (name, img) in enumerate(list(bg_variants.items())[:cols]):
            plt.subplot(rows, cols, cols + i + 1)
            plt.title(f"Background: {name}")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        # Add material variants on third row
        for i, (name, img) in enumerate(list(mat_variants.items())[:cols]):
            plt.subplot(rows, cols, 2*cols + i + 1)
            plt.title(f"Material: {name}")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path


# Example usage of the shoe image editor
def demo_shoe_editor(shoe_image_path, output_dir="shoe_variants"):
    """
    Demonstrate the shoe editor functionality with a sample shoe image.
    """
    # Create the editor
    editor = ShoeImageEditor()
    
    # Load the shoe image
    editor.load_shoe_image(shoe_image_path)
    
    # Add color options
    editor.add_color_option("Red", "red")
    editor.add_color_option("Blue", "blue")
    editor.add_color_option("Green", "green")
    editor.add_color_option("Yellow", "yellow")
    editor.add_color_option("Purple", "purple")
    editor.add_color_option("Black", "black")
    editor.add_color_option("White", "white")
    
    # Create and add background options
    # Create solid color backgrounds
    def create_solid_bg(color, size=(800, 600)):
        bg = np.ones((size[1], size[0], 3), dtype=np.uint8)
        if isinstance(color, str):
            color_map = {
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "gray": (128, 128, 128),
                "light_gray": (200, 200, 200)
            }
            color = color_map.get(color, (255, 255, 255))
        bg[:] = color
        return bg
    
    # Add backgrounds
    editor.add_background("White", create_solid_bg("white"))
    editor.add_background("Black", create_solid_bg("black"))
    editor.add_background("Gray", create_solid_bg("gray"))
    editor.add_background("Light Gray", create_solid_bg("light_gray"))
    
    # Create a gradient background
    def create_gradient_bg(start_color, end_color, size=(800, 600)):
        bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for y in range(size[1]):
            # Calculate the blend factor based on y position
            factor = y / size[1]
            # Interpolate between start and end color
            color = tuple(int(start_color[i] * (1 - factor) + end_color[i] * factor) for i in range(3))
            # Fill the row with this color
            bg[y, :] = color
        return bg
    
    # Add gradient backgrounds
    editor.add_background("Blue Gradient", create_gradient_bg((255, 200, 150), (150, 150, 255)))
    editor.add_background("Warm Gradient", create_gradient_bg((255, 200, 150), (255, 150, 150)))
    
    # Generate and display all variants
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate color variants
    color_variants = editor.generate_color_variants()
    color_grid = editor.create_grid_display(color_variants, "Color Variants")
    color_grid.savefig(f"{output_dir}/color_variants.jpg")
    
    # Generate background variants
    bg_variants = editor.generate_background_variants()
    bg_grid = editor.create_grid_display(bg_variants, "Background Variants")
    bg_grid.savefig(f"{output_dir}/background_variants.jpg")
    
    # Generate material variants
    mat_variants = editor.generate_material_variants()
    mat_grid = editor.create_grid_display(mat_variants, "Material Variants")
    mat_grid.savefig(f"{output_dir}/material_variants.jpg")
    
    # Create a catalog view
    catalog_path = editor.create_catalog_view(f"{output_dir}/shoe_catalog.jpg")
    
    # Save individual variants
    editor.save_all_variants(output_dir)
    
    return output_dir

# Functions to create a web interface
def create_web_interface(shoe_editor):
    """Create a simple web interface for the shoe editor."""
    import ipywidgets as widgets
    from IPython.display import display, HTML
    
    # Create dropdown for color selection
    color_dropdown = widgets.Dropdown(
        options=list(shoe_editor.available_colors.keys()),
        description='Color:',
        disabled=False,
    )
    
    # Create dropdown for background selection
    bg_dropdown = widgets.Dropdown(
        options=list(shoe_editor.available_backgrounds.keys()),
        description='Background:',
        disabled=False,
    )
    
    # Create dropdown for material selection
    material_dropdown = widgets.Dropdown(
        options=['leather', 'suede', 'patent', 'canvas'],
        description='Material:',
        disabled=False,
    )
    
    # Create output widget to display the image
    output = widgets.Output()
    
    # Function to update the image
    def update_image(change=None):
        with output:
            output.clear_output()
            # Apply color
            shoe_editor.change_shoe_color(color_dropdown.value)
            # Apply background
            shoe_editor.change_background(bg_dropdown.value)
            # Apply material
            shoe_editor.apply_material(material_dropdown.value)
            # Display result
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(shoe_editor.current_edit, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"{color_dropdown.value} {material_dropdown.value} shoe on {bg_dropdown.value} background")
            plt.show()
    
    # Connect the dropdown to the update function
    color_dropdown.observe(update_image, names='value')
    bg_dropdown.observe(update_image, names='value')
    material_dropdown.observe(update_image, names='value')
    
    # Create a button to show all variants
    show_all_button = widgets.Button(
        description='Show All Variants',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Generate and display all variants',
    )
    
    def on_show_all_button_clicked(b):
        with output:
            output.clear_output()
            # Create a catalog view
            catalog_path = shoe_editor.create_catalog_view("temp_catalog.jpg")
            # Display the catalog
            img = plt.imread(catalog_path)
            plt.figure(figsize=(16, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.title("All Variants")
            plt.show()
    
    show_all_button.on_click(on_show_all_button_clicked)
    
    # Create a button to save the current edit
    save_button = widgets.Button(
        description='Save Current Edit',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Save the current edit',
    )
    
    def on_save_button_clicked(b):
        path = shoe_editor.save_current_edit("current_edit.jpg")
        with output:
            print(f"Saved to {path}")
    
    save_button.on_click(on_save_button_clicked)
    
    # Arrange widgets in a VBox
    ui = widgets.VBox([
        widgets.HBox([color_dropdown, bg_dropdown, material_dropdown]),
        widgets.HBox([show_all_button, save_button]),
        output
    ])
    
    # Initial update
    update_image()
    
    return ui

# Google Colab specific functions
def setup_for_colab():
    """Setup the environment for Google Colab."""
    # Install required packages
    !pip install -q segment-anything ipywidgets matplotlib
    
    # Download SAM model weights if needed
    if not os.path.exists('sam_vit_h_4b8939.pth'):
        !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    
    # Function to upload images
    from google.colab import files
    
    print("Please upload a shoe image:")
    uploaded = files.upload()
    shoe_path = next(iter(uploaded))
    
    # Create and initialize the editor
    editor = ShoeImageEditor()
    editor.load_shoe_image(shoe_path)
    
    # Add default color options
    editor.add_color_option("Red", "red")
    editor.add_color_option("Blue", "blue")
    editor.add_color_option("Green", "green")
    editor.add_color_option("Yellow", "yellow")
    editor.add_color_option("Purple", "purple")
    editor.add_color_option("Black", "black")
    editor.add_color_option("White", "white")
    
    # Add default backgrounds
    def create_solid_bg(color, size=(800, 600)):
        bg = np.ones((size[1], size[0], 3), dtype=np.uint8)
        if isinstance(color, str):
            color_map = {
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "gray": (128, 128, 128),
                "light_gray": (200, 200, 200)
            }
            color = color_map.get(color, (255, 255, 255))
        bg[:] = color
        return bg
    
    editor.add_background("White", create_solid_bg("white"))
    editor.add_background("Black", create_solid_bg("black"))
    editor.add_background("Gray", create_solid_bg("gray"))
    editor.add_background("Light Gray", create_solid_bg("light_gray"))
    
    # Create a gradient background
    def create_gradient_bg(start_color, end_color, size=(800, 600)):
        bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for y in range(size[1]):
            factor = y / size[1]
            color = tuple(int(start_color[i] * (1 - factor) + end_color[i] * factor) for i in range(3))
            bg[y, :] = color
        return bg
    
    editor.add_background("Blue Gradient", create_gradient_bg((255, 200, 150), (150, 150, 255)))
    editor.add_background("Warm Gradient", create_gradient_bg((255, 200, 150), (255, 150, 150)))
    
    # Create and display the UI
    ui = create_web_interface(editor)
    from IPython.display import display
    display(ui)
    
    return editor

# Main execution
if __name__ == "__main__":
    # Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        editor = setup_for_colab()
    else:
        # For non-Colab environments, take a different path
        print("Please provide the path to a shoe image:")
        shoe_path = input().strip()
        output_dir = "shoe_variants"
        demo_shoe_editor(shoe_path, output_dir)
        print(f"Variants have been saved to the '{output_dir}' directory")