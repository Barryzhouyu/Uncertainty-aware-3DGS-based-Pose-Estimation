import os
from PIL import Image, ImageEnhance

# Paths
input_folder = "/home/roar/Desktop/48_-40_-294/"
output_folder = "/home/roar/Desktop/48_-40_-294_better/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to adjust brightness and contrast
def adjust_lighting(img, brightness_factor=0.85, contrast_factor=1.5):
    bright_enhancer = ImageEnhance.Brightness(img)
    img = bright_enhancer.enhance(brightness_factor)

    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(contrast_factor)
    
    return img

# Process all PNG images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Only process PNG files
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure it's RGB format
        
        # Apply lighting adjustments
        modified_img = adjust_lighting(img, brightness_factor=0.9, contrast_factor=1.5)

        # Save to output folder
        output_path = os.path.join(output_folder, filename)
        modified_img.save(output_path)

print(f"Processing complete! Modified images saved in: {output_folder}")
