
"""
AutoML - Standard Approach
==========================
- AutoML uses 461 calibration samples (internally splits for CV)
- We validate final model on independent 197 samples
"""

import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step2_output/calibration_selected_features.csv')
VALID_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step2_output/validation_selected_features.csv')
OUT_DIR = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step3_output')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = 'soc'

print("\n" + "="*80)
print("AUTOML - STANDARD APPROACH")
print("="*80)
print("  AutoML trains on 461 samples (internal CV)")
print("  Final validation on independent 197 samples")

# Load data
print("\nLoading data...")
train_df = pd.read_csv(CALIB_CSV)
test_df = pd.read_csv(VALID_CSV)

print(f"  Calibration (for AutoML): {train_df.shape}")
print(f"  Validation (held out): {test_df.shape}")

# Train AutoML
print("\n" + "="*80)
print("TRAINING AUTOML")
print("="*80)
print("  Time limit: 30 minutes")
print("  AutoML will internally split 461 samples")
print("  (likely 80/20 → ~368 train, ~93 validation)")

model_path = os.path.join(OUT_DIR, 'models')

predictor = TabularPredictor(
    label=TARGET_COL,
    eval_metric='root_mean_squared_error',
    path=model_path
).fit(
    train_data=train_df,  # 461 samples - let AutoML split internally
    time_limit=1800,
    presets='medium_quality',
    verbosity=2
)

# Get leaderboard (from AutoML's internal validation)
print("\n" + "="*80)
print("AUTOML RESULTS (Internal Validation)")
print("="*80)

leaderboard = predictor.leaderboard(train_df, silent=True)
print("\nTOP 10 MODELS (ranked by AutoML's internal validation):")
print(leaderboard[['model', 'score_val', 'pred_time_val']].head(10))

best_model = predictor.model_best
print(f"\n🏆 AutoML selected: {best_model}")

# NOW test on YOUR independent validation set
print("\n" + "="*80)
print("YOUR INDEPENDENT VALIDATION (197 samples)")
print("="*80)

y_test = test_df[TARGET_COL].values
y_pred = predictor.predict(test_df)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
rpd = y_test.std() / rmse
q75, q25 = np.percentile(y_test, [75, 25])
rpiq = (q75 - q25) / rmse

print(f"\n  Best Model: {best_model}")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")
print(f"  RPD  = {rpd:.4f}")
print(f"  RPIQ = {rpiq:.4f}")

if rpd > 2.0:
    capability = "EXCELLENT"
elif rpd > 1.4:
    capability = "GOOD"
else:
    capability = "MODERATE"

print(f"\n  📊 Prediction Capability: {capability}")

# Save results
leaderboard.to_csv(os.path.join(OUT_DIR, 'model_leaderboard_internal.csv'), index=False)

results_df = pd.DataFrame([{
    'Best_Model': best_model,
    'Internal_Validation': 'AutoML 80/20 split on 461 samples',
    'Independent_Validation': '197 samples (Kennard-Stone)',
    'R²': r2,
    'RMSE': rmse,
    'MAE': mae,
    'RPD': rpd,
    'RPIQ': rpiq,
    'Capability': capability
}])
results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS.csv'), index=False)

# Save predictions
pred_df = test_df.copy()
pred_df['Predicted_SOC'] = y_pred
pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

print(f"\n📁 Results saved to: {OUT_DIR}/")

print("\n" + "="*80)
print("✅ COMPLETE")
print("="*80)
print(f"\n  AutoML trained on: 461 samples")
print(f"  Validated on: 197 independent samples")
print(f"\n  Best Model: {best_model}")
print(f"  R² = {r2:.4f}")
print(f"  RPD = {rpd:.4f} ({capability})")
print("\n" + "="*80)
"""
AutoML with CUSTOM COMPOSITE METRIC
====================================
AutoGluon will select models based on composite score:
0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE

Instead of just RMSE!
"""

import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step2_output/calibration_selected_features.csv')
VALID_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step2_output/validation_selected_features.csv')
OUT_DIR = os.path.join(HERE, 'automl_COMPOSITE_METRIC')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = 'soc'

print("\n" + "="*80)
print("AUTOML WITH CUSTOM COMPOSITE METRIC")
print("="*80)
print("  AutoGluon will select models based on:")
print("  Composite = 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE")
print("  NOT just RMSE!")
print("="*80)


# ═══════════════════════════════════════════════════════════════════════════
# DEFINE CUSTOM COMPOSITE METRIC
# ═══════════════════════════════════════════════════════════════════════════

# Weights (same as your multi-metric selection)
WEIGHTS = {
    'RPD':   0.35,
    'RPIQ':  0.25,
    'RMSE':  0.20,
    'R²':    0.15,
    'MAE':   0.05
}

print(f"\n⚖️  WEIGHTING:")
for metric, weight in WEIGHTS.items():
    print(f"  {metric:6s}: {weight:.0%}")


def composite_score_metric(y_true, y_pred, sample_weight=None):
    """
    Custom metric: Composite score for soil science
    
    Higher is better!
    
    Combines:
    - RPD (35%) - soil science standard
    - RPIQ (25%) - robust performance
    - RMSE (20%) - prediction error
    - R² (15%) - variance explained
    - MAE (5%) - average error
    
    Returns a score where HIGHER = BETTER
    """
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate individual metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # RPD
    rpd = y_true.std() / rmse if rmse > 0 else 0
    
    # RPIQ
    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse if rmse > 0 else 0
    
    # Normalize each metric to 0-1 scale
    # We need reference values - use typical ranges for SOC prediction
    
    # RPD: 0-4 range (typical in soil science)
    rpd_norm = np.clip(rpd / 4.0, 0, 1)
    
    # RPIQ: 0-4 range
    rpiq_norm = np.clip(rpiq / 4.0, 0, 1)
    
    # RMSE: inverse (lower is better), assume 0-2 g/kg range
    rmse_norm = 1 - np.clip(rmse / 2.0, 0, 1)
    
    # R²: already 0-1, but can be negative
    r2_norm = np.clip(r2, 0, 1)
    
    # MAE: inverse (lower is better), assume 0-2 g/kg range
    mae_norm = 1 - np.clip(mae / 2.0, 0, 1)
    
    # Calculate composite (weighted sum)
    composite = (
        WEIGHTS['RPD'] * rpd_norm +
        WEIGHTS['RPIQ'] * rpiq_norm +
        WEIGHTS['RMSE'] * rmse_norm +
        WEIGHTS['R²'] * r2_norm +
        WEIGHTS['MAE'] * mae_norm
    )
    
    # Return composite (higher is better)
    return composite


# Create AutoGluon scorer from custom metric
composite_scorer = make_scorer(
    name='composite_score',
    score_func=composite_score_metric,
    optimum=1.0,          # Best possible score
    greater_is_better=True  # Higher is better
)

print("\n✅ Custom composite metric defined")
print("   AutoGluon will now optimize this instead of RMSE!")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_df = pd.read_csv(CALIB_CSV)
test_df = pd.read_csv(VALID_CSV)

print(f"  Calibration: {train_df.shape}")
print(f"  Validation: {test_df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN AUTOGLUON WITH CUSTOM METRIC
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TRAINING AUTOML WITH CUSTOM COMPOSITE METRIC")
print("="*80)
print("  Time limit: 30 minutes")
print("  Selection criterion: COMPOSITE SCORE (not RMSE)")

model_path = os.path.join(OUT_DIR, 'models')

predictor = TabularPredictor(
    label=TARGET_COL,
    eval_metric=composite_scorer,  # ✅ CUSTOM METRIC!
    path=model_path
).fit(
    train_data=train_df,
    time_limit=1800,
    presets='medium_quality',
    verbosity=2
)

print("\n✅ Training complete using COMPOSITE METRIC!")


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("AUTOML RESULTS (Selected by Composite Score)")
print("="*80)

leaderboard = predictor.leaderboard(train_df, silent=True)
print("\nTOP 10 MODELS (ranked by COMPOSITE SCORE):")
print(leaderboard[['model', 'score_val', 'pred_time_val']].head(10))

best_model = predictor.model_best
print(f"\n🏆 AutoML selected (by composite): {best_model}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST ON INDEPENDENT VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("INDEPENDENT VALIDATION (197 samples)")
print("="*80)

y_test = test_df[TARGET_COL].values
y_pred = predictor.predict(test_df)

# Calculate all metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rpd = y_test.std() / rmse
q75, q25 = np.percentile(y_test, [75, 25])
rpiq = (q75 - q25) / rmse

# Calculate composite score
composite = composite_score_metric(y_test, y_pred)

print(f"\n  Best Model: {best_model}")
print(f"\n  COMPOSITE METRICS:")
print(f"    RPD  = {rpd:.4f} (weight: {WEIGHTS['RPD']:.0%})")
print(f"    RPIQ = {rpiq:.4f} (weight: {WEIGHTS['RPIQ']:.0%})")
print(f"    RMSE = {rmse:.4f} (weight: {WEIGHTS['RMSE']:.0%})")
print(f"    R²   = {r2:.4f} (weight: {WEIGHTS['R²']:.0%})")
print(f"    MAE  = {mae:.4f} (weight: {WEIGHTS['MAE']:.0%})")
print(f"\n  📊 COMPOSITE SCORE = {composite:.4f}")

if rpd > 2.0:
    capability = "EXCELLENT"
elif rpd > 1.4:
    capability = "GOOD"
else:
    capability = "MODERATE"

print(f"\n  📈 Prediction Capability: {capability} (RPD-based)")


# ═══════════════════════════════════════════════════════════════════════════
# DETAILED LEADERBOARD WITH ALL METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("DETAILED LEADERBOARD (All Metrics)")
print("="*80)

# Get predictions for all models
model_names = leaderboard['model'].tolist()

detailed_results = []

for model_name in model_names:
    try:
        y_pred_model = predictor.predict(test_df, model=model_name)
        
        mae_m = mean_absolute_error(y_test, y_pred_model)
        rmse_m = np.sqrt(mean_squared_error(y_test, y_pred_model))
        r2_m = r2_score(y_test, y_pred_model)
        rpd_m = y_test.std() / rmse_m
        rpiq_m = (q75 - q25) / rmse_m
        composite_m = composite_score_metric(y_test, y_pred_model)
        
        detailed_results.append({
            'Model': model_name,
            'Composite': composite_m,
            'RPD': rpd_m,
            'RPIQ': rpiq_m,
            'RMSE': rmse_m,
            'R²': r2_m,
            'MAE': mae_m
        })
    except:
        pass

df_detailed = pd.DataFrame(detailed_results)
df_detailed = df_detailed.sort_values('Composite', ascending=False)
df_detailed.insert(0, 'Rank', range(1, len(df_detailed) + 1))

print("\n" + df_detailed.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

leaderboard.to_csv(os.path.join(OUT_DIR, 'model_leaderboard_composite.csv'), index=False)
df_detailed.to_csv(os.path.join(OUT_DIR, 'detailed_metrics_all_models.csv'), index=False)

results_df = pd.DataFrame([{
    'Best_Model': best_model,
    'Selection_Criterion': 'Composite Score (0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE)',
    'Composite_Score': composite,
    'RPD': rpd,
    'RPIQ': rpiq,
    'RMSE': rmse,
    'R²': r2,
    'MAE': mae,
    'Capability': capability
}])
results_df.to_csv(os.path.join(OUT_DIR, 'FINAL_RESULTS_COMPOSITE.csv'), index=False)

# Save predictions
pred_df = test_df.copy()
pred_df['Predicted_SOC'] = y_pred
pred_df.to_csv(os.path.join(OUT_DIR, 'validation_predictions.csv'), index=False)

print(f"\n📁 Results saved to: {OUT_DIR}/")
print(f"  Models saved at: {model_path}")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("✅ AUTOML COMPLETE (CUSTOM COMPOSITE METRIC)")
print("="*80)

print(f"\n  🎯 KEY DIFFERENCE:")
print(f"     Standard AutoML: Selects by RMSE only")
print(f"     This AutoML: Selects by COMPOSITE SCORE")
print(f"     (35% RPD + 25% RPIQ + 20% RMSE + 15% R² + 5% MAE)")

print(f"\n  🏆 BEST MODEL: {best_model}")
print(f"     Composite Score = {composite:.4f}")
print(f"     RPD = {rpd:.4f} ({capability})")
print(f"     R² = {r2:.4f}")
print(f"     RMSE = {rmse:.4f}")

print(f"\n  ✅ Model was selected based on ALL 5 metrics")
print(f"     NOT just RMSE!")

print("\n" + "="*80)

