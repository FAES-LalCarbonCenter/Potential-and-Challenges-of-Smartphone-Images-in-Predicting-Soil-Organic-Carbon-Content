#Writing this code in order to add three new columns to the dataset called CIELAB, CIELUV and CIE XYZ in order to aid the usage of device variability. Emphasizes the importance of illumination normalization and color correction. Conversion to CIELAB/XYZ helped minimize variance and improved SOC estimation.

import pandas as pd
import numpy as np
from skimage.color import rgb2lab, rgb2luv, rgb2xyz

# === Load your RGB features ===
features_path = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/soil_image_features.csv"
df = pd.read_csv(features_path)

# Normalize RGB to [0,1] for conversion
df['r_norm'] = df['mean_r'] / 255.0
df['g_norm'] = df['mean_g'] / 255.0
df['b_norm'] = df['mean_b'] / 255.0

# Stack normalized RGB columns
rgb_array = df[['r_norm', 'g_norm', 'b_norm']].values.reshape(-1, 1, 3)

# Convert to CIELAB
lab = rgb2lab(rgb_array).reshape(-1, 3)
df[['lab_L', 'lab_a', 'lab_b']] = lab

# Convert to CIELUV
luv = rgb2luv(rgb_array).reshape(-1, 3)
df[['luv_L', 'luv_u', 'luv_v']] = luv

# Convert to CIE XYZ
xyz = rgb2xyz(rgb_array).reshape(-1, 3)
df[['xyz_X', 'xyz_Y', 'xyz_Z']] = xyz

# Drop temp columns
df.drop(columns=['r_norm', 'g_norm', 'b_norm'], inplace=True)

# === Save updated feature file ===
df.to_csv("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/soil_image_features.csv", index=False)
print("✅ Color space features added and file saved.")
