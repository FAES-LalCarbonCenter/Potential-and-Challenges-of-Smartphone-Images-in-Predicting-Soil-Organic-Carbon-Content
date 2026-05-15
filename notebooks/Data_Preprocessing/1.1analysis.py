#this is the code written to resize the already cropped images to 224x224 pixels - (1-205)


import os
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime 

# Define input and output directories
input_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step1_input/1-205")
output_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step1_output/resize224x244(1-205)")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n🗂️ Input directory:  {input_dir}")
print(f"💾 Output directory: {output_dir}\n")


# Step 1: Rename all .JPG to .jpg in input directory
for img_file in input_dir.glob("*.JPG"):
    new_file = img_file.with_suffix(".jpg")
    img_file.rename(new_file)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Renamed {img_file.name} ➜ {new_file.name}")

# Step 2: Resize all .jpg images
resized_count = 0
for img_file in tqdm(sorted(input_dir.glob("*.jpg"))):
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"❌ Could not load {img_file.name}")
        continue

    # Resize image
    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # Save resized image
    out_path = output_dir / img_file.name
    cv2.imwrite(str(out_path), resized_img)
    resized_count += 1

print(f"✅ Total resized and saved images: {resized_count}")
