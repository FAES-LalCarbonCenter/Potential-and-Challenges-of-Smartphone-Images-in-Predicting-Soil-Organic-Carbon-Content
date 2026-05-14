
"""
MOISTURE ANALYSIS - STEP 2: Generate LightGBM Predictions
==========================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline")

MOISTURE_META = PROJECT_ROOT / "notebooks/objective_2/obj2_output_data/step1_output/moisture_metadata.csv"
MODEL_PATH = PROJECT_ROOT / "notebooks/objective_1/output_data/step3_output/models/models/LightGBM/model.pkl"
CALIB_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/calibration_selected_features.csv"
VALID_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/validation_selected_features.csv"

OUT_DIR = PROJECT_ROOT / "notebooks/objective_2/obj2_output_data/step2_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "lightgbm_moisture_predictions.csv"

print("\n" + "="*80)
print("MOISTURE ANALYSIS - STEP 2: GENERATE PREDICTIONS")
print("="*80)

print("\nLoading moisture metadata...")
df_moisture = pd.read_csv(MOISTURE_META)
print(f"  Total samples: {len(df_moisture)}")

print("\nLoading feature data (12 selected features)...")
df_calib_feat = pd.read_csv(CALIB_FEATURES)
df_valid_feat = pd.read_csv(VALID_FEATURES)
print(f"  Calibration features: {df_calib_feat.shape}")
print(f"  Validation features:  {df_valid_feat.shape}")

df_features = pd.concat([df_calib_feat, df_valid_feat], ignore_index=True)
print(f"  Combined features: {df_features.shape}")

print("\nMerging moisture categories with feature data...")
df = pd.merge(
    df_features,
    df_moisture[['image_no', 'Moisture_Category', 'split']],
    on='image_no',
    how='inner'
)

print(f"  Merged dataset: {df.shape}")
print(f"  Moisture distribution:")
print(df['Moisture_Category'].value_counts())

print("\nLoading LightGBM model...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"  ✅ Loaded model from pickle: {MODEL_PATH}")

print("\nPreparing features for prediction...")
y_true = df['soc'].values

# Keep ALL columns except soc and the ones we just added
X = df.drop(columns=['soc', 'Moisture_Category', 'split'])

print(f"  Feature columns ({len(X.columns)}): {X.columns.tolist()}")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Target array shape: {y_true.shape}")

print("\nGenerating predictions...")
try:
    y_pred = model.predict(X)
    print(f"  ✅ Generated {len(y_pred)} predictions")
except Exception as e:
    print(f"  ❌ Error during prediction: {e}")
    exit(1)

print("\nCreating predictions dataframe...")
df_pred = pd.DataFrame({
    'image_no': df['image_no'].values,
    'soc_actual': y_true,
    'soc_predicted': y_pred,
    'Moisture_Category': df['Moisture_Category'].values,
    'split': df['split'].values,
    'moisture': df['moisture'].values
})

print(f"  Predictions dataframe: {df_pred.shape}")

print("\n" + "="*80)
print("OVERALL PERFORMANCE (ALL SAMPLES)")
print("="*80)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
rpd = y_true.std() / rmse if rmse > 0 else np.nan
q75, q25 = np.percentile(y_true, [75, 25])
rpiq = (q75 - q25) / rmse if rmse > 0 else np.nan

print(f"\n  R²    = {r2:.4f}")
print(f"  RMSE  = {rmse:.4f} g/kg")
print(f"  MAE   = {mae:.4f} g/kg")
print(f"  RPD   = {rpd:.4f}")
print(f"  RPIQ  = {rpiq:.4f}")

print("\n" + "="*80)
print("PERFORMANCE BY MOISTURE CATEGORY")
print("="*80)

for category in ['Dry', 'Moist', 'Wet']:
    subset = df_pred[df_pred['Moisture_Category'] == category]
    
    if len(subset) == 0:
        print(f"\n{category}: No samples")
        continue
    
    y_t = subset['soc_actual'].values
    y_p = subset['soc_predicted'].values
    
    mae_cat = mean_absolute_error(y_t, y_p)
    rmse_cat = np.sqrt(mean_squared_error(y_t, y_p))
    r2_cat = r2_score(y_t, y_p)
    rpd_cat = y_t.std() / rmse_cat if rmse_cat > 0 else np.nan
    q75_cat, q25_cat = np.percentile(y_t, [75, 25])
    rpiq_cat = (q75_cat - q25_cat) / rmse_cat if rmse_cat > 0 else np.nan
    
    print(f"\n{category} (n={len(subset)}):")
    print(f"  R²    = {r2_cat:.4f}")
    print(f"  RMSE  = {rmse_cat:.4f} g/kg")
    print(f"  MAE   = {mae_cat:.4f} g/kg")
    print(f"  RPD   = {rpd_cat:.4f}", end="")
    
    if rpd_cat > 2.0:
        print("  (EXCELLENT)")
    elif rpd_cat > 1.4:
        print("  (GOOD)")
    else:
        print("  (MODERATE)")
    
    print(f"  RPIQ  = {rpiq_cat:.4f}")

print("\n" + "="*80)
print("SAVING PREDICTIONS")
print("="*80)

df_pred.to_csv(OUT_CSV, index=False)

print(f"\n✅ Saved predictions: {OUT_CSV}")
print(f"   Total predictions: {len(df_pred)}")
print(f"   Columns: {df_pred.columns.tolist()}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n  Total samples: {len(df_pred)}")
print(f"\n  By moisture category:")
for cat in ['Dry', 'Moist', 'Wet']:
    count = len(df_pred[df_pred['Moisture_Category'] == cat])
    print(f"    {cat:8s}: {count:3d} samples")

print(f"\n  By split:")
for split in ['calibration', 'validation']:
    count = len(df_pred[df_pred['split'] == split])
    print(f"    {split:12s}: {count:3d} samples")

print("\n" + "="*80)
print("✅ STEP 2 COMPLETE")
print("="*80)
print(f"\nNext step: Run moisture_step3_plot_and_analyze.py")
print("="*80)