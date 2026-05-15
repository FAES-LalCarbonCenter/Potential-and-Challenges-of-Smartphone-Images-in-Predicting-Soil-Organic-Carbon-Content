# Here we are writing code to extract the features of the images like mean, median, entropy, contrast etc. the outputof this code is a csv file of data saved under the name soil_image_features.csv that has arround 22+ features extracted for each image.

import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# GLCM texture features
def extract_glcm_features(gray_img):
    gray_img = img_as_ubyte(gray_img)  # Convert to 8-bit
    glcm = graycomatrix(gray_img, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    return contrast, energy, homogeneity, entropy

# Custom color indices
def custom_color_indices(r, g, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        RI = np.where((b * g**3) != 0, r**2 / (b * g**3), 0)
        CI = np.where((r + g) != 0, (r - g) / (r + g), 0)
        HI = np.where((g - b) != 0, (2 * r - g - b) / (g - b), 0)
        SI = np.where((r + b) != 0, (r - b) / (r + b), 0)
    return np.nanmean(RI), np.nanmean(CI), np.nanmean(HI), np.nanmean(SI)

# Main extraction loop
def extract_features_from_folder(folder_path):
    feature_list = []

    for filename in tqdm(sorted(os.listdir(folder_path))):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_gray = rgb2gray(img_rgb)

            # RGB means and medians
            R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
            mean_r, mean_g, mean_b = np.mean(R), np.mean(G), np.mean(B)
            median_r, median_g, median_b = np.median(R), np.median(G), np.median(B)

            # HSV
            H, S, V = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
            mean_h, mean_s, mean_v = np.mean(H), np.mean(S), np.mean(V)
            median_h, median_s, median_v = np.median(H), np.median(S), np.median(V)

            # Grayscale
            mean_gray = np.mean(img_gray)
            median_gray = np.median(img_gray)

            # Texture features from GLCM
            contrast, energy, homogeneity, entropy = extract_glcm_features(img_gray)

            # Custom color indices
            ri, ci, hi, si = custom_color_indices(R, G, B)

            features = {
                'filename': filename,
                'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
                'median_r': median_r, 'median_g': median_g, 'median_b': median_b,
                'mean_h': mean_h, 'mean_s': mean_s, 'mean_v': mean_v,
                'median_h': median_h, 'median_s': median_s, 'median_v': median_v,
                'mean_gray': mean_gray, 'median_gray': median_gray,
                'contrast': contrast, 'energy': energy,
                'homogeneity': homogeneity, 'entropy': entropy,
                'RI': ri, 'CI': ci, 'HI': hi, 'SI': si
            }

            feature_list.append(features)

    return pd.DataFrame(feature_list)

# Example usage
folder_path = '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field'
features_df = extract_features_from_folder(folder_path)
# Save to desired output folder
output_csv_path = '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step4_output/soil_image_features.csv'
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
features_df.to_csv(output_csv_path, index=False)
