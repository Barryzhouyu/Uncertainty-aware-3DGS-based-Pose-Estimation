# import os
# from PIL import Image, ImageEnhance

# # Paths
# input_folder = "/home/roar/Desktop/48_-40_-294/"
# output_folder = "/home/roar/Desktop/48_-40_-294_better/"

# # Ensure output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Function to adjust brightness and contrast
# def adjust_lighting(img, brightness_factor=0.85, contrast_factor=1.5):
#     bright_enhancer = ImageEnhance.Brightness(img)
#     img = bright_enhancer.enhance(brightness_factor)

#     contrast_enhancer = ImageEnhance.Contrast(img)
#     img = contrast_enhancer.enhance(contrast_factor)
    
#     return img

# # Process all PNG images in the folder
# for filename in os.listdir(input_folder):
#     if filename.endswith(".png"):  # Only process PNG files
#         img_path = os.path.join(input_folder, filename)
#         img = Image.open(img_path).convert("RGB")  # Ensure it's RGB format
        
#         # Apply lighting adjustments
#         modified_img = adjust_lighting(img, brightness_factor=0.9, contrast_factor=1.5)

#         # Save to output folder
#         output_path = os.path.join(output_folder, filename)
#         modified_img.save(output_path)

# print(f"Processing complete! Modified images saved in: {output_folder}")
import cv2
import os
import glob
import numpy as np

# ===== Configuration =====
image_dir = "/home/roar3/Desktop/pool_test_images"
output_dir = os.path.join(image_dir, "sharpened")
os.makedirs(output_dir, exist_ok=True)

# ===== Sharpening kernel =====
sharpen_kernel = np.array([[2, -3, 2],
                           [-3, 5, -3],
                           [2, -3, 2]])

# Optional: Enhance contrast for murky images
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ===== Process each image =====
for image_path in glob.glob(os.path.join(image_dir, "*.png")):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        continue

    # Optional: enhance contrast first
    # img = enhance_contrast(img)

    # Apply sharpening
    sharpened = cv2.filter2D(img, -1, sharpen_kernel)

    # Save the sharpened image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, sharpened)

print("✅ Sharpening complete. Saved to:", output_dir)
