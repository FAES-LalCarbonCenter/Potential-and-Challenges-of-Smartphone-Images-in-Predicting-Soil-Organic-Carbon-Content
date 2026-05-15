# In this code we are- pics. We have only selected those pictures that have cracks in them. The input images are taken from the enhanced_sample_field folder. That is after performing enhancement on the cropped images.

#Step by Step segmentation process

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define paths
input_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step3_input/cracked_images_raw")
mask_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step3_output/crack_masks")
output_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step3_output/crack_removed")
mask_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Process each image
for img_file in tqdm(sorted(input_dir.glob("*.jpg"))):
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"❌ Could not load {img_file.name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Adaptive thresholding to detect cracks
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=10
    )

    # Step 2: Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save binary mask of cracks
    mask_path = mask_dir / img_file.name
    cv2.imwrite(str(mask_path), cleaned)

    # Step 3: Inpainting cracks from original image
    inpainted = cv2.inpaint(img, cleaned, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    output_path = output_dir / img_file.name
    cv2.imwrite(str(output_path), inpainted)

print(f"\n✅ All crack masks saved in: {mask_dir}")
print(f"✅ All cleaned images saved in: {output_dir}")



