#Here we are inserting the enhanced_sample_images files which are cropped and enhanced with the erstwhile pictures which have cracks removed of cracks_removed folder back into the parent folder of enhanced_sample_field images


import os
import shutil
from pathlib import Path

# === Define folders ===
crack_removed_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step3_output/crack_removed")
main_dataset_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field")

# === Define allowed extensions ===
valid_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# === Read files from both folders ===
crack_removed_files = [f for f in crack_removed_dir.glob("*") if f.suffix in valid_exts]
main_dataset_files = {f.stem: f for f in main_dataset_dir.glob("*") if f.suffix in valid_exts}

print(f"🧾 Found {len(crack_removed_files)} images in crack_removed folder.")
print(f"🧾 Found {len(main_dataset_files)} images in main dataset folder.\n")

# === Print sample filenames for confirmation ===
print(f"🧪 Example crack_removed filenames (stems):\n{[f.stem for f in crack_removed_files[:10]]}\n")
print(f"📁 Example main dataset filenames (stems):\n{list(main_dataset_files.keys())[:10]}\n")

# === Replace matched files ===
replaced = 0
for crack_file in crack_removed_files:
    stem = crack_file.stem
    if stem in main_dataset_files:
        shutil.copy2(crack_file, main_dataset_files[stem])
        replaced += 1

print(f"✅ Total crack_removed images successfully replaced in main dataset: {replaced}")

