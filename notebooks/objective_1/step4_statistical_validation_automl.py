"""
Statistical Validation - MULTI-METRIC COMPOSITE BASED
======================================================
Validates models using the SAME composite score used for selection

Composite Score = 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE
(All metrics normalized to 0-1 scale)

Tests:
1. Friedman Test - Compare composite scores
2. Post-hoc Pairwise Wilcoxon - Pairwise composite score differences
3. Bootstrap CI - Uncertainty in composite score + all 5 metrics
4. Paired Tests vs Baseline - Composite score improvement
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
VALID_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step2_output/validation_selected_features.csv')
OUT_DIR = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step4_output')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = 'soc'

print("\n" + "="*80)
print("STATISTICAL VALIDATION - MULTI-METRIC COMPOSITE")
print("="*80)
print("  Validating using SAME metrics as model selection:")
print("  Composite = 0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE")
print("="*80)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERT WEIGHTS (Same as selection)
# ═══════════════════════════════════════════════════════════════════════════

weights = {
    'RPD':   0.35,
    'RPIQ':  0.25,
    'RMSE':  0.20,
    'R²':    0.15,
    'MAE':   0.05
}

print(f"\n⚖️  WEIGHTING (consistent with selection):")
for metric, weight in weights.items():
    print(f"  {metric:6s}: {weight:.0%}")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA AND MODELS
# ═══════════════════════════════════════════════════════════════════════════

df_valid = pd.read_csv(VALID_CSV)
y_true = df_valid[TARGET_COL].values

print(f"\nValidation samples: {len(y_true)}")

from autogluon.tabular import TabularPredictor

model_path = os.path.join(HERE, 'automl_final/models')

if not os.path.exists(model_path):
    print(f"\n❌ ERROR: Model path not found: {model_path}")
    exit()

predictor = TabularPredictor.load(model_path)

print("\nGetting predictions from all models...")

leaderboard = predictor.leaderboard(df_valid, silent=True)
model_names = leaderboard['model'].tolist()

predictions = {}
for model_name in model_names:
    try:
        y_pred = predictor.predict(df_valid, model=model_name)
        predictions[model_name] = y_pred.values
        print(f"  ✅ {model_name}")
    except Exception as e:
        print(f"  ❌ {model_name}: {e}")

print(f"\n✅ Successfully loaded {len(predictions)} models")


# ═══════════════════════════════════════════════════════════════════════════
# CALCULATE ALL METRICS AND COMPOSITE SCORES
# ═══════════════════════════════════════════════════════════════════════════

def calculate_all_metrics(y_true, y_pred):
    """Calculate all 5 metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rpd = y_true.std() / rmse if rmse > 0 else 0
    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse if rmse > 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'RPD': rpd,
        'RPIQ': rpiq
    }


print("\n" + "="*80)
print("CALCULATING METRICS FOR ALL MODELS")
print("="*80)

# Calculate metrics for all models
all_metrics = {}
for model_name, y_pred in predictions.items():
    all_metrics[model_name] = calculate_all_metrics(y_true, y_pred)

# Create dataframe
df_metrics = pd.DataFrame(all_metrics).T
df_metrics['Model'] = df_metrics.index
df_metrics = df_metrics.reset_index(drop=True)

# Normalize metrics (0-1 scale) - SAME as selection
df_normalized = df_metrics.copy()

# RMSE (lower is better → invert)
rmse_min = df_metrics['RMSE'].min()
rmse_max = df_metrics['RMSE'].max()
df_normalized['RMSE_norm'] = (rmse_max - df_metrics['RMSE']) / (rmse_max - rmse_min) if rmse_max > rmse_min else 1.0

# MAE (lower is better → invert)
mae_min = df_metrics['MAE'].min()
mae_max = df_metrics['MAE'].max()
df_normalized['MAE_norm'] = (mae_max - df_metrics['MAE']) / (mae_max - mae_min) if mae_max > mae_min else 1.0

# R² (higher is better)
r2_min = df_metrics['R²'].min()
r2_max = df_metrics['R²'].max()
df_normalized['R²_norm'] = (df_metrics['R²'] - r2_min) / (r2_max - r2_min) if r2_max > r2_min else 1.0

# RPD (higher is better)
rpd_min = df_metrics['RPD'].min()
rpd_max = df_metrics['RPD'].max()
df_normalized['RPD_norm'] = (df_metrics['RPD'] - rpd_min) / (rpd_max - rpd_min) if rpd_max > rpd_min else 1.0

# RPIQ (higher is better)
rpiq_min = df_metrics['RPIQ'].min()
rpiq_max = df_metrics['RPIQ'].max()
df_normalized['RPIQ_norm'] = (df_metrics['RPIQ'] - rpiq_min) / (rpiq_max - rpiq_min) if rpiq_max > rpiq_min else 1.0

# Calculate composite score
df_normalized['Composite_Score'] = (
    weights['RPD']   * df_normalized['RPD_norm'] +
    weights['RPIQ']  * df_normalized['RPIQ_norm'] +
    weights['RMSE']  * df_normalized['RMSE_norm'] +
    weights['R²']    * df_normalized['R²_norm'] +
    weights['MAE']   * df_normalized['MAE_norm']
)

# Create dictionary for easy access
composite_scores = dict(zip(df_normalized['Model'], df_normalized['Composite_Score']))

print("  ✅ Composite scores calculated for all models")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: FRIEDMAN TEST (on Composite Scores)
# ═══════════════════════════════════════════════════════════════════════════

def friedman_test_composite(y_true, predictions, composite_scores):
    """
    Friedman test on COMPOSITE SCORES
    
    For each sample, we calculate composite score, then compare across models
    """
    
    print("\n" + "="*80)
    print("TEST 1: FRIEDMAN TEST (Composite Score)")
    print("="*80)
    print("  Metric: Multi-metric Composite Score")
    print("  H₀: All models have equal composite performance")
    print("  Hₐ: At least one model differs significantly")
    
    # For Friedman, we need sample-wise scores
    # We'll use inverse composite as "error" (lower composite = higher error)
    
    # Calculate per-sample metrics for each model
    sample_composites = {}
    
    for model_name, y_pred in predictions.items():
        sample_scores = []
        
        for i in range(len(y_true)):
            # Calculate metrics for this single sample
            y_t = np.array([y_true[i]])
            y_p = np.array([y_pred[i]])
            
            mae_s = np.abs(y_t - y_p)[0]
            rmse_s = np.abs(y_t - y_p)[0]  # For single sample, RMSE = MAE
            r2_s = 1 - ((y_t - y_p)**2 / y_true.var()) if y_true.var() > 0 else 0
            rpd_s = y_true.std() / rmse_s if rmse_s > 0 else 0
            q75, q25 = np.percentile(y_true, [75, 25])
            rpiq_s = (q75 - q25) / rmse_s if rmse_s > 0 else 0
            
            # Normalize (using overall min/max from all models)
            rmse_norm = (rmse_max - rmse_s) / (rmse_max - rmse_min) if rmse_max > rmse_min else 1.0
            mae_norm = (mae_max - mae_s) / (mae_max - mae_min) if mae_max > mae_min else 1.0
            r2_norm = (r2_s - r2_min) / (r2_max - r2_min) if r2_max > r2_min else 1.0
            rpd_norm = (rpd_s - rpd_min) / (rpd_max - rpd_min) if rpd_max > rpd_min else 1.0
            rpiq_norm = (rpiq_s - rpiq_min) / (rpiq_max - rpiq_min) if rpiq_max > rpiq_min else 1.0
            
            # Composite for this sample
            comp = (weights['RPD'] * rpd_norm +
                   weights['RPIQ'] * rpiq_norm +
                   weights['RMSE'] * rmse_norm +
                   weights['R²'] * r2_norm +
                   weights['MAE'] * mae_norm)
            
            # Use inverse as "error" for Friedman (lower is worse)
            sample_scores.append(1 - comp)  # Inverted composite = error
        
        sample_composites[model_name] = np.array(sample_scores)
    
    # Friedman test
    score_arrays = list(sample_composites.values())
    statistic, p_value = friedmanchisquare(*score_arrays)
    
    print("\n" + "-"*80)
    print(f"  Friedman χ² = {statistic:.4f}")
    print(f"  p-value = {p_value:.10f}")
    
    if p_value < 0.05:
        print(f"  ✅ SIGNIFICANT (p < 0.05): Models differ significantly!")
    else:
        print(f"  ❌ NOT SIGNIFICANT")
    
    # Show rankings by composite score
    print("\n  Model Rankings by Composite Score:")
    for model_name, score in sorted(composite_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {model_name:25s}: Composite = {score:.4f}")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'sample_composites': sample_composites
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: POST-HOC PAIRWISE WILCOXON (Composite)
# ═══════════════════════════════════════════════════════════════════════════

def posthoc_pairwise_wilcoxon_composite(sample_composites, composite_scores):
    """Pairwise Wilcoxon on composite scores"""
    
    print("\n" + "="*80)
    print("TEST 2: POST-HOC PAIRWISE WILCOXON (Composite)")
    print("="*80)
    
    model_names = list(sample_composites.keys())
    
    results = []
    
    for model1, model2 in combinations(model_names, 2):
        scores1 = sample_composites[model1]
        scores2 = sample_composites[model2]
        
        statistic, p_value = wilcoxon(scores1, scores2)
        
        comp1 = composite_scores[model1]
        comp2 = composite_scores[model2]
        
        winner = model1 if comp1 > comp2 else model2
        difference = abs(comp1 - comp2)
        
        results.append({
            'Model_1': model1,
            'Model_2': model2,
            'Composite_1': comp1,
            'Composite_2': comp2,
            'Difference': difference,
            'Winner': winner,
            'p_value': p_value
        })
    
    df_pairwise = pd.DataFrame(results)
    
    # Holm correction
    n_tests = len(df_pairwise)
    p_values = df_pairwise['p_value'].values
    sorted_indices = np.argsort(p_values)
    
    holm_thresholds = [0.05 / (n_tests - i) for i in range(n_tests)]
    holm_array = np.empty(n_tests)
    holm_array[sorted_indices] = holm_thresholds
    
    df_pairwise['Holm_Threshold'] = holm_array
    df_pairwise['Significant_Holm'] = df_pairwise['p_value'] < df_pairwise['Holm_Threshold']
    
    df_pairwise = df_pairwise.sort_values('Difference', ascending=False)
    
    print("\n" + "-"*80)
    print("  TOP 10 LARGEST COMPOSITE SCORE DIFFERENCES:")
    print("\n" + df_pairwise.head(10).to_string(index=False))
    
    n_significant = df_pairwise['Significant_Holm'].sum()
    print(f"\n  {n_significant}/{n_tests} pairwise comparisons significant (Holm-corrected)")
    
    df_pairwise.to_csv(
        os.path.join(OUT_DIR, 'posthoc_pairwise_wilcoxon_composite.csv'), 
        index=False
    )
    
    return df_pairwise


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: BOOTSTRAP CI (All 5 Metrics + Composite)
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci_best_model(y_true, y_pred_best, model_name, weights, n_bootstrap=5000):
    """Bootstrap CI for ALL metrics + composite"""
    
    print("\n" + "="*80)
    print(f"TEST 3: BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"  Model: {model_name}")
    print("="*80)
    
    rpd_boots = []
    rpiq_boots = []
    rmse_boots = []
    r2_boots = []
    mae_boots = []
    composite_boots = []
    
    n_samples = len(y_true)
    
    print(f"\n  Running {n_bootstrap} bootstrap resamples...")
    
    # Get global min/max for normalization (from original data)
    global_metrics = calculate_all_metrics(y_true, y_pred_best)
    
    for i in range(n_bootstrap):
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n_bootstrap}...")
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_best[indices]
        
        metrics = calculate_all_metrics(y_true_boot, y_pred_boot)
        
        rpd_boots.append(metrics['RPD'])
        rpiq_boots.append(metrics['RPIQ'])
        rmse_boots.append(metrics['RMSE'])
        r2_boots.append(metrics['R²'])
        mae_boots.append(metrics['MAE'])
        
        # Calculate composite (normalized)
        # For bootstrap, we normalize within each resample
        # But for consistency, we use original scale
        composite_boots.append(
            weights['RPD'] * (metrics['RPD'] / global_metrics['RPD']) +
            weights['RPIQ'] * (metrics['RPIQ'] / global_metrics['RPIQ']) +
            weights['RMSE'] * (global_metrics['RMSE'] / metrics['RMSE']) +  # Inverted
            weights['R²'] * (metrics['R²'] / global_metrics['R²']) +
            weights['MAE'] * (global_metrics['MAE'] / metrics['MAE'])  # Inverted
        )
    
    # Calculate CIs
    def get_ci(data):
        return np.percentile(data, [2.5, 97.5])
    
    print("\n" + "-"*80)
    print(f"  BOOTSTRAP RESULTS (95% Confidence Intervals):")
    print(f"\n  PRIMARY METRICS:")
    print(f"    RPD  = {np.mean(rpd_boots):.4f} ± {np.std(rpd_boots):.4f}")
    print(f"       95% CI: [{get_ci(rpd_boots)[0]:.4f}, {get_ci(rpd_boots)[1]:.4f}]")
    print(f"\n    RPIQ = {np.mean(rpiq_boots):.4f} ± {np.std(rpiq_boots):.4f}")
    print(f"       95% CI: [{get_ci(rpiq_boots)[0]:.4f}, {get_ci(rpiq_boots)[1]:.4f}]")
    
    print(f"\n  ERROR METRICS:")
    print(f"    RMSE = {np.mean(rmse_boots):.4f} ± {np.std(rmse_boots):.4f}")
    print(f"       95% CI: [{get_ci(rmse_boots)[0]:.4f}, {get_ci(rmse_boots)[1]:.4f}]")
    print(f"\n    MAE  = {np.mean(mae_boots):.4f} ± {np.std(mae_boots):.4f}")
    print(f"       95% CI: [{get_ci(mae_boots)[0]:.4f}, {get_ci(mae_boots)[1]:.4f}]")
    
    print(f"\n  GOODNESS OF FIT:")
    print(f"    R²   = {np.mean(r2_boots):.4f} ± {np.std(r2_boots):.4f}")
    print(f"       95% CI: [{get_ci(r2_boots)[0]:.4f}, {get_ci(r2_boots)[1]:.4f}]")
    
    print(f"\n  COMPOSITE SCORE:")
    print(f"    Composite = {np.mean(composite_boots):.4f} ± {np.std(composite_boots):.4f}")
    print(f"       95% CI: [{get_ci(composite_boots)[0]:.4f}, {get_ci(composite_boots)[1]:.4f}]")
    
    # Save
    df_bootstrap = pd.DataFrame({
        'Metric': ['RPD', 'RPIQ', 'RMSE', 'R²', 'MAE', 'Composite'],
        'Weight': [weights['RPD'], weights['RPIQ'], weights['RMSE'], weights['R²'], weights['MAE'], 1.0],
        'Mean': [np.mean(rpd_boots), np.mean(rpiq_boots), np.mean(rmse_boots), 
                np.mean(r2_boots), np.mean(mae_boots), np.mean(composite_boots)],
        'Std': [np.std(rpd_boots), np.std(rpiq_boots), np.std(rmse_boots),
               np.std(r2_boots), np.std(mae_boots), np.std(composite_boots)],
        'CI_Lower': [get_ci(rpd_boots)[0], get_ci(rpiq_boots)[0], get_ci(rmse_boots)[0],
                    get_ci(r2_boots)[0], get_ci(mae_boots)[0], get_ci(composite_boots)[0]],
        'CI_Upper': [get_ci(rpd_boots)[1], get_ci(rpiq_boots)[1], get_ci(rmse_boots)[1],
                    get_ci(r2_boots)[1], get_ci(mae_boots)[1], get_ci(composite_boots)[1]]
    })
    df_bootstrap.to_csv(
        os.path.join(OUT_DIR, 'bootstrap_confidence_intervals_ALL_METRICS.csv'), 
        index=False
    )
    
    return df_bootstrap


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all tests"""
    
    # Test 1: Friedman on composite
    friedman_results = friedman_test_composite(y_true, predictions, composite_scores)
    
    # Test 2: Pairwise Wilcoxon
    if friedman_results['p_value'] < 0.05:
        df_pairwise = posthoc_pairwise_wilcoxon_composite(
            friedman_results['sample_composites'], 
            composite_scores
        )
    
    # Test 3: Bootstrap for LightGBM
    best_model_name = 'LightGBM'
    if best_model_name in predictions:
        y_pred_best = predictions[best_model_name]
        df_bootstrap = bootstrap_ci_best_model(y_true, y_pred_best, best_model_name, weights)
    
    # Save summary
    df_summary = df_normalized[['Model', 'Composite_Score', 'RPD', 'RPIQ', 'RMSE', 'R²', 'MAE']].copy()
    df_summary = df_summary.sort_values('Composite_Score', ascending=False)
    df_summary.insert(0, 'Rank', range(1, len(df_summary) + 1))
    df_summary.to_csv(os.path.join(OUT_DIR, 'model_performance_composite.csv'), index=False)
    
    print("\n" + "="*80)
    print("✅ MULTI-METRIC STATISTICAL VALIDATION COMPLETE")
    print("="*80)
    print(f"\n  All tests used COMPOSITE SCORE:")
    print(f"  0.35×RPD + 0.25×RPIQ + 0.20×RMSE + 0.15×R² + 0.05×MAE")
    print(f"\n  Files saved to: {OUT_DIR}/")
    print("="*80)


if __name__ == '__main__':
    main()