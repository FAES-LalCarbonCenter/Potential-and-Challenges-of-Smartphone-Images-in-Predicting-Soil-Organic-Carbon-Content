# Here we are writing code to obtain the color space conversions of the soil images into HSV and Gray scale images. This is for getting the Fig. 9 images present in the research paper and doesnt effect the code at all.



# import cv2
# import os
# from tqdm import tqdm

# # ====== 📁 Assign your paths explicitly ======
# input_folder = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field"
# output_gray_folder = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/gray_images"
# output_h_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/hue_images"
# output_s_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/saturation_images"
# output_v_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/value_images"

# # ====== 📂 Create output folders ======
# os.makedirs(output_gray_folder, exist_ok=True)
# os.makedirs(output_h_channel, exist_ok=True)
# os.makedirs(output_s_channel, exist_ok=True)
# os.makedirs(output_v_channel, exist_ok=True)

# # ====== 🔁 Process each image ======
# for i in tqdm(range(1, 732)):
#     filename = f"{i}.jpg"
#     filepath = os.path.join(input_folder, filename)

#     img_bgr = cv2.imread(filepath)
#     if img_bgr is None:
#         print(f"⚠️ Skipping {filename} — could not read.")
#         continue

#     # ---- Grayscale Conversion ----
#     img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(os.path.join(output_gray_folder, filename), img_gray)

#     # ---- HSV Conversion and Split ----
#     img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(img_hsv)

#     # Save H, S, V channels
#     cv2.imwrite(os.path.join(output_h_channel, filename), h)
#     cv2.imwrite(os.path.join(output_s_channel, filename), s)
#     cv2.imwrite(os.path.join(output_v_channel, filename), v)

# print("✅ Saved all grayscale and HSV channel images.")


# import cv2
# import os
# from tqdm import tqdm

# # ====== 📁 Assign your paths ======
# input_folder = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field"

# # Grayscale and HSV
# output_gray_folder = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/gray_images"
# output_h_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/hue_images"
# output_s_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/saturation_images"
# output_v_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/value_images"

# # RGB
# output_r_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/red_images"
# output_g_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/green_images"
# output_b_channel = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/blue_images"

# # ====== 📂 Create output folders ======
# for folder in [output_gray_folder, output_h_channel, output_s_channel, output_v_channel,
#                output_r_channel, output_g_channel, output_b_channel]:
#     os.makedirs(folder, exist_ok=True)

# # ====== 🔁 Process each image ======
# for i in tqdm(range(1, 732)):
#     filename = f"{i}.jpg"
#     filepath = os.path.join(input_folder, filename)

#     img_bgr = cv2.imread(filepath)
#     if img_bgr is None:
#         print(f"⚠️ Skipping {filename} — could not read.")
#         continue

#     # ---- Grayscale ----
#     img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(os.path.join(output_gray_folder, filename), img_gray)

#     # ---- HSV ----
#     img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(img_hsv)
#     cv2.imwrite(os.path.join(output_h_channel, filename), h)
#     cv2.imwrite(os.path.join(output_s_channel, filename), s)
#     cv2.imwrite(os.path.join(output_v_channel, filename), v)

#     # ---- RGB ----
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     r, g, b = cv2.split(img_rgb)
#     cv2.imwrite(os.path.join(output_r_channel, filename), r)
#     cv2.imwrite(os.path.join(output_g_channel, filename), g)
#     cv2.imwrite(os.path.join(output_b_channel, filename), b)

# print("✅ Saved all grayscale, HSV, and RGB channel images.")


# import cv2
# import os
# from tqdm import tqdm
# import numpy as np

# input_folder = "/Users/dharamkar.1/.../enhanced_sample_field"
# output_red = "/Users/dharamkar.1/.../red_images"
# output_green = "/Users/dharamkar.1/.../green_images"
# output_blue = "/Users/dharamkar.1/.../blue_images"

# os.makedirs(output_red, exist_ok=True)
# os.makedirs(output_green, exist_ok=True)
# os.makedirs(output_blue, exist_ok=True)

# for i in tqdm(range(1, 732)):
#     filename = f"{i}.jpg"
#     filepath = os.path.join(input_folder, filename)

#     img = cv2.imread(filepath)
#     if img is None:
#         print(f"⚠️ Skipping {filename}")
#         continue

#     # Split channels
#     b, g, r = cv2.split(img)

#     # Merge to get red-toned image
#     red_img = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
#     green_img = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
#     blue_img = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])

#     # Save
#     cv2.imwrite(os.path.join(output_red, filename), red_img)
#     cv2.imwrite(os.path.join(output_green, filename), green_img)
#     cv2.imwrite(os.path.join(output_blue, filename), blue_img)

# print("✅ Saved red, green, and blue images.")


import os
import cv2
import numpy as np
from tqdm import tqdm

# ====== Set paths ======
input_folder = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field"

# Output folders
base_output = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output"

output_dirs = {
    "gray": "gray_images",
    "hue": "hue_images",
    "saturation": "saturation_images",
    "value": "value_images",
    "red": "red_images",
    "green": "green_images",
    "blue": "blue_images"
}

# ====== Create output folders ======
full_output_paths = {}
for name, subfolder in output_dirs.items():
    path = os.path.join(base_output, subfolder)
    os.makedirs(path, exist_ok=True)
    full_output_paths[name] = path

# ====== Process each image ======
for i in tqdm(range(1, 732)):
    filename = f"{i}.jpg"
    filepath = os.path.join(input_folder, filename)

    img = cv2.imread(filepath)
    if img is None:
        print(f"⚠️ Skipping {filename} — could not read.")
        continue

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(full_output_paths["gray"], filename), gray)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    cv2.imwrite(os.path.join(full_output_paths["hue"], filename), h)
    cv2.imwrite(os.path.join(full_output_paths["saturation"], filename), s)
    cv2.imwrite(os.path.join(full_output_paths["value"], filename), v)

    # RGB channels isolated
    b, g, r = cv2.split(img)
    red_img = cv2.merge([np.zeros_like(r), np.zeros_like(g), r])
    green_img = cv2.merge([np.zeros_like(r), g, np.zeros_like(b)])
    blue_img = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])

    cv2.imwrite(os.path.join(full_output_paths["red"], filename), red_img)
    cv2.imwrite(os.path.join(full_output_paths["green"], filename), green_img)
    cv2.imwrite(os.path.join(full_output_paths["blue"], filename), blue_img)

print("✅ Saved grayscale, HSV (H/S/V), and RGB (R/G/B) channel images.")
