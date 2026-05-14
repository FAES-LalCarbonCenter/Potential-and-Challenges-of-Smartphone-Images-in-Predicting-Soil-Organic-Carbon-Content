"""
SOIL TYPE ANALYSIS - STEP 2: Generate Predictions
==================================================
"""

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import pickle
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# PROJECT_ROOT = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline")

# SOIL_META = PROJECT_ROOT / "notebooks/objective_3/obj3_output_data/step1_output/soil_type_metadata.csv"
# MODEL_PATH = PROJECT_ROOT / "notebooks/objective_1/output_data/step3_output/models/models/LightGBM/model.pkl"
# CALIB_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/calibration_selected_features.csv"
# VALID_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/validation_selected_features.csv"

# OUT_DIR = PROJECT_ROOT / "notebooks/objective_3/obj3_output_data/step2_output"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# OUT_CSV = OUT_DIR / "lightgbm_soil_type_predictions.csv"

# print("\n" + "="*80)
# print("SOIL TYPE ANALYSIS - STEP 2: GENERATE PREDICTIONS")
# print("="*80)

# print("\nLoading soil type metadata...")
# df_soil = pd.read_csv(SOIL_META)
# print(f"  Total samples: {len(df_soil)}")

# print("\nLoading feature data...")
# df_calib = pd.read_csv(CALIB_FEATURES)
# df_valid = pd.read_csv(VALID_FEATURES)
# df_features = pd.concat([df_calib, df_valid], ignore_index=True)
# print(f"  Combined features: {df_features.shape}")

# print("\nMerging...")
# df = pd.merge(df_features, df_soil[['image_no', 'soil_type', 'split']], on='image_no', how='inner')
# print(f"  Merged: {df.shape}")
# print(f"\n  Soil type distribution:")
# print(df['soil_type'].value_counts())

# print("\nLoading LightGBM model...")
# with open(MODEL_PATH, 'rb') as f:
#     model = pickle.load(f)
# print(f"  ✅ Loaded")

# print("\nPreparing features...")
# y_true = df['soc'].values
# X = df.drop(columns=['soc', 'soil_type', 'split'])

# print(f"  X shape: {X.shape}")
# print(f"  y_true shape: {y_true.shape}")

# print("\nGenerating predictions...")
# y_pred = model.predict(X)
# print(f"  ✅ Generated {len(y_pred)} predictions")

# df_pred = pd.DataFrame({
#     'image_no': df['image_no'].values,
#     'soc_actual': y_true,
#     'soc_predicted': y_pred,
#     'soil_type': df['soil_type'].values,
#     'split': df['split'].values
# })

# print("\n" + "="*80)
# print("OVERALL PERFORMANCE")
# print("="*80)

# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_true, y_pred)
# rpd = y_true.std() / rmse
# q75, q25 = np.percentile(y_true, [75, 25])
# rpiq = (q75 - q25) / rmse

# print(f"\n  R²    = {r2:.4f}")
# print(f"  RMSE  = {rmse:.4f}")
# print(f"  RPD   = {rpd:.4f}")

# print("\n" + "="*80)
# print("BY SOIL TYPE")
# print("="*80)

# for soil_type in sorted(df_pred['soil_type'].unique()):
#     subset = df_pred[df_pred['soil_type'] == soil_type]
    
#     y_t = subset['soc_actual'].values
#     y_p = subset['soc_predicted'].values
    
#     mae_s = mean_absolute_error(y_t, y_p)
#     rmse_s = np.sqrt(mean_squared_error(y_t, y_p))
#     r2_s = r2_score(y_t, y_p)
#     rpd_s = y_t.std() / rmse_s
#     q75_s, q25_s = np.percentile(y_t, [75, 25])
#     rpiq_s = (q75_s - q25_s) / rmse_s
    
#     rpd_cat = "EXCELLENT" if rpd_s > 2.0 else ("GOOD" if rpd_s > 1.4 else "MODERATE")
    
#     print(f"\n{soil_type} (n={len(subset)}):")
#     print(f"  R²    = {r2_s:.4f}")
#     print(f"  RMSE  = {rmse_s:.4f}")
#     print(f"  RPD   = {rpd_s:.4f}  ({rpd_cat})")

# df_pred.to_csv(OUT_CSV, index=False)
# print(f"\n✅ Saved: {OUT_CSV}")

# print("\n" + "="*80)
# print("✅ STEP 2 COMPLETE")
# print("="*80)
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline")

FIELD_LAB_META = PROJECT_ROOT / "notebooks/objective_3/obj3_output_data/step1_output/field_lab_metadata.csv"
MODEL_PATH = PROJECT_ROOT / "notebooks/objective_1/output_data/step3_output/models/models/LightGBM/model.pkl"
CALIB_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/calibration_selected_features.csv"
VALID_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/validation_selected_features.csv"

OUT_DIR = PROJECT_ROOT / "notebooks/objective_3/obj3_output_data/step2_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "lightgbm_field_lab_predictions.csv"

print("="*80)
print("FIELD vs LAB - STEP 2")
print("="*80)

df_meta = pd.read_csv(FIELD_LAB_META)
df_calib = pd.read_csv(CALIB_FEATURES)
df_valid = pd.read_csv(VALID_FEATURES)
df_features = pd.concat([df_calib, df_valid], ignore_index=True)

df = pd.merge(df_features, df_meta[['image_no', 'image_type', 'split']], on='image_no', how='inner')

print(f"\nMerged: {df.shape}")
print("\nImage type distribution:")
print(df['image_type'].value_counts())

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

y_true = df['soc'].values
X = df.drop(columns=['soc', 'image_type', 'split'])

y_pred = model.predict(X)

df_pred = pd.DataFrame({
    'image_no': df['image_no'].values,
    'soc_actual': y_true,
    'soc_predicted': y_pred,
    'image_type': df['image_type'].values,
    'split': df['split'].values
})

print("\n" + "="*80)
print("PERFORMANCE BY IMAGE TYPE")
print("="*80)

for img_type in ['Field', 'Lab']:
    subset = df_pred[df_pred['image_type'] == img_type]
    if len(subset) == 0:
        continue
    
    y_t = subset['soc_actual'].values
    y_p = subset['soc_predicted'].values
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    rpd = y_t.std() / rmse
    mae_s = mean_absolute_error(y_t, y_p)
    rpd_cat = "EXCELLENT" if rpd > 2.0 else ("GOOD" if rpd > 1.4 else "MODERATE")
    
    print(f"\n{img_type} (n={len(subset)}):")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  RPD  = {rpd:.4f}  ({rpd_cat})")
    print(f" MAE = {mae_s: .4f}")

df_pred.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved: {OUT_CSV}")