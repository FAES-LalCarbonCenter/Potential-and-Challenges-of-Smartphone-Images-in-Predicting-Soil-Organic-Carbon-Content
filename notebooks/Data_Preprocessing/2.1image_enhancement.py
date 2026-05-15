# #image enhancement on all images


import cv2
from pathlib import Path
from tqdm import tqdm

# List of input folders
input_dirs = [
    Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_input/manual_crop_petri(269-360,437-731)"),
    Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step1_output/resize224x244(1-205)"),
    Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step1_output/resize_field(206-268,361-436)")  # previously used folder
]



output_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field")
output_dir.mkdir(parents=True, exist_ok=True)

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Gather all images
image_files = []
for folder in input_dirs:
    image_files.extend(sorted(folder.glob("*.jpg")))

print(f"Total images: {len(image_files)}")

for img_file in tqdm(image_files):
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Could not read: {img_file.name}")
        continue

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    l_clahe = clahe.apply(l)
    lab_enhanced = cv2.merge((l_clahe, a, b))

    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Save
    out_path = output_dir / img_file.name
    success = cv2.imwrite(str(out_path), enhanced)
    if not success:
        print(f"Failed to save: {img_file.name}")

print(f"\n✅ Enhanced images saved to: {output_dir.resolve()}")
