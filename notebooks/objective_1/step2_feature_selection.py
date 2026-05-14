"""
Multi-Criteria Feature Selection (MCFS) - CALIBRATION SET ONLY
===============================================================
Proper cross-validation approach to prevent data leakage

Dataset: calibration_set.csv (70% from Kennard-Stone split)
Input:   calibration_set.csv (~460 samples)
Output:  
  - Features selected from calibration set ONLY
  - Same features applied to validation set
  - Ready for model training

Pipeline (Ding et al., 2025; Viscarra Rossel & Behrens, 2010):
  Stage 1: Quality Pre-filtering (calibration data only)
  Stage 2: Mutual Information Scoring (calibration data only)
  Stage 3: PCA-MI Hybrid Scoring (70-30, calibration data only)
  Stage 4: LASSO Regularization Refinement (calibration data only)
  
  Then: Apply selected features to validation set

Outputs saved to ./feature_selection_calibration_output/
  CALIBRATION SET:
    calibration_selected_features.csv  - calibration data with selected features
    feature_selection_report.txt       - complete summary
    feature_selection_plots.png        - 4-panel visualization
  
  VALIDATION SET:
    validation_selected_features.csv   - validation data with SAME features
  
  FEATURE LIST:
    selected_features_list.txt         - just the feature names (for reference)

Requirements: pip install pandas numpy matplotlib scipy scikit-learn
Usage:        python feature_selection_calibration_only.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
CALIB_CSV    = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step1_output/calibration_set.csv')
VALID_CSV    = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step1_output/validation_set.csv')
OUT_DIR      = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step2_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# Target and identifier columns
TARGET_COL = 'soc'
ID_COL     = 'image_no'
EXCLUDE_COLS = ['filename', 'Numeric numbers', 'image_path', 'image_no', 
                'soil_type', 'moisture', 'soc', 'Mahalanobis_Distance']

# Stage 1: Quality filtering thresholds
MIN_VARIANCE      = 0.01    # Remove features with variance < 0.01
MAX_MISSING_PCT   = 0.20    # Remove features with >20% missing values
P_VALUE_THRESHOLD = 0.10    # Keep features with p < 0.10 (marginal significance)

# Stage 3: Hybrid scoring weights
PCA_WEIGHT = 0.70           # Weight for PCA score (multivariate patterns)
MI_WEIGHT  = 0.30           # Weight for MI score (non-linear predictive power)

# Stage 3: Feature selection targets
N_FEATURES_TARGET = 20      # Target number of features
MIN_FEATURES      = 12      # Minimum features to retain
MAX_FEATURES      = 25      # Maximum features to retain

# Random seed
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: QUALITY PRE-FILTERING
# ═══════════════════════════════════════════════════════════════════════════════
def stage1_quality_filter(df, candidate_features):
    """Stage 1: Quality-based pre-filtering (CALIBRATION SET ONLY)"""
    
    print("\n" + "="*80)
    print("STAGE 1: QUALITY PRE-FILTERING (CALIBRATION SET ONLY)")
    print("="*80)
    print(f"  Input features: {len(candidate_features)}")
    
    filtered_features = []
    removal_log = []
    
    for feature in candidate_features:
        # Check 1: Variance
        var = df[feature].var()
        if var < MIN_VARIANCE:
            removal_log.append({
                'feature': feature,
                'reason': 'low_variance',
                'detail': f'variance = {var:.6f}',
                'threshold': f'< {MIN_VARIANCE}'
            })
            continue
        
        # Check 2: Missing values
        missing_pct = df[feature].isna().sum() / len(df)
        if missing_pct > MAX_MISSING_PCT:
            removal_log.append({
                'feature': feature,
                'reason': 'missing_values',
                'detail': f'{missing_pct*100:.2f}% missing',
                'threshold': f'> {MAX_MISSING_PCT*100}%'
            })
            continue
        
        # Check 3: Statistical significance with SOC
        x = df[feature].dropna()
        y = df.loc[x.index, TARGET_COL]
        
        if len(x) < 10:
            removal_log.append({
                'feature': feature,
                'reason': 'insufficient_data',
                'detail': f'only {len(x)} valid observations',
                'threshold': '< 10 samples'
            })
            continue
        
        try:
            r, p = pearsonr(x, y)
            if p >= P_VALUE_THRESHOLD:
                removal_log.append({
                    'feature': feature,
                    'reason': 'not_significant',
                    'detail': f'p = {p:.4f}, r = {r:.4f}',
                    'threshold': f'p ≥ {P_VALUE_THRESHOLD}'
                })
                continue
        except Exception as e:
            removal_log.append({
                'feature': feature,
                'reason': 'correlation_error',
                'detail': str(e),
                'threshold': 'N/A'
            })
            continue
        
        # Passed all checks
        filtered_features.append(feature)
    
    print(f"\n  Results:")
    print(f"    Passed filter: {len(filtered_features)}")
    print(f"    Removed:       {len(removal_log)}")
    
    if len(removal_log) > 0:
        df_removed = pd.DataFrame(removal_log)
        print(f"\n  Removal breakdown:")
        for reason, count in df_removed['reason'].value_counts().items():
            print(f"    {reason:20s}: {count:3d}")
        
        df_removed.to_csv(os.path.join(OUT_DIR, 'stage1_removal_log.csv'), 
                         index=False)
        print(f"  Saved: stage1_removal_log.csv")
    
    return filtered_features, removal_log


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: MUTUAL INFORMATION SCORING
# ═══════════════════════════════════════════════════════════════════════════════
def stage2_mutual_information(df, features):
    """Stage 2: Compute Mutual Information scores (CALIBRATION SET ONLY)"""
    
    print("\n" + "="*80)
    print("STAGE 2: MUTUAL INFORMATION SCORING (CALIBRATION SET ONLY)")
    print("="*80)
    print(f"  Computing MI for {len(features)} features...")
    
    X = df[features].values
    y = df[TARGET_COL].values
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    mi_scores = mutual_info_regression(X_imputed, y, 
                                       random_state=SEED, 
                                       n_neighbors=5)
    
    mi_max = mi_scores.max()
    mi_scores_norm = mi_scores / mi_max if mi_max > 0 else mi_scores
    
    df_mi = pd.DataFrame({
        'feature': features,
        'mi_score_raw': mi_scores,
        'mi_score_norm': mi_scores_norm,
        'rank_by_mi': range(1, len(features) + 1)
    }).sort_values('mi_score_raw', ascending=False).reset_index(drop=True)
    
    df_mi['rank_by_mi'] = range(1, len(df_mi) + 1)
    
    print(f"\n  MI Statistics:")
    print(f"    Mean:   {mi_scores.mean():.4f}")
    print(f"    Median: {np.median(mi_scores):.4f}")
    print(f"    Max:    {mi_scores.max():.4f}")
    print(f"    Min:    {mi_scores.min():.4f}")
    
    print(f"\n  Top 10 features by Mutual Information:")
    print(f"  {'Rank':<6} {'Feature':<30} {'MI Score':<12} {'Normalized'}")
    print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12}")
    for idx, row in df_mi.head(10).iterrows():
        print(f"  {int(row['rank_by_mi']):<6} {row['feature']:<30} "
              f"{row['mi_score_raw']:<12.4f} {row['mi_score_norm']:<12.4f}")
    
    df_mi.to_csv(os.path.join(OUT_DIR, 'stage2_mutual_information.csv'), 
                 index=False)
    print(f"\n  Saved: stage2_mutual_information.csv")
    
    return df_mi, imputer


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: PCA-MI HYBRID SCORING
# ═══════════════════════════════════════════════════════════════════════════════
def stage3_pca_scoring(df, features):
    """Stage 3A: Compute PCA-based importance scores (CALIBRATION SET ONLY)"""
    
    print("\n" + "="*80)
    print("STAGE 3A: PCA VARIANCE SCORING (CALIBRATION SET ONLY)")
    print("="*80)
    
    X = df[features].values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=0.95, random_state=SEED)
    pca.fit(X_scaled)
    
    n_components = pca.n_components_
    loadings = pca.components_
    variance_explained = pca.explained_variance_ratio_
    
    print(f"  PCA retained {n_components} components")
    print(f"  Total variance explained: {variance_explained.sum():.4f}")
    
    pca_scores = np.zeros(len(features))
    for i in range(len(features)):
        pca_scores[i] = np.sum(np.abs(loadings[:, i]) * variance_explained)
    
    pca_max = pca_scores.max()
    pca_scores_norm = pca_scores / pca_max if pca_max > 0 else pca_scores
    
    df_pca = pd.DataFrame({
        'feature': features,
        'pca_score_raw': pca_scores,
        'pca_score_norm': pca_scores_norm,
        'rank_by_pca': range(1, len(features) + 1)
    }).sort_values('pca_score_raw', ascending=False).reset_index(drop=True)
    
    df_pca['rank_by_pca'] = range(1, len(df_pca) + 1)
    
    print(f"\n  Top 10 features by PCA score:")
    print(f"  {'Rank':<6} {'Feature':<30} {'PCA Score':<12} {'Normalized'}")
    print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12}")
    for idx, row in df_pca.head(10).iterrows():
        print(f"  {int(row['rank_by_pca']):<6} {row['feature']:<30} "
              f"{row['pca_score_raw']:<12.4f} {row['pca_score_norm']:<12.4f}")
    
    df_pca.to_csv(os.path.join(OUT_DIR, 'stage3_pca_scores.csv'), index=False)
    print(f"\n  Saved: stage3_pca_scores.csv")
    
    return df_pca, pca, scaler


def stage3_combine_scores(df_pca, df_mi):
    """Stage 3B: Combine PCA and MI scores with 70-30 weighting"""
    
    print("\n" + "="*80)
    print(f"STAGE 3B: HYBRID SCORING ({int(PCA_WEIGHT*100)}-{int(MI_WEIGHT*100)} PCA-MI)")
    print("="*80)
    
    df_combined = df_pca[['feature', 'pca_score_raw', 'pca_score_norm']].merge(
        df_mi[['feature', 'mi_score_raw', 'mi_score_norm']], 
        on='feature'
    )
    
    df_combined['combined_score'] = (
        PCA_WEIGHT * df_combined['pca_score_norm'] +
        MI_WEIGHT * df_combined['mi_score_norm']
    )
    
    df_combined = df_combined.sort_values('combined_score', ascending=False)
    df_combined['rank'] = range(1, len(df_combined) + 1)
    df_combined = df_combined.reset_index(drop=True)
    
    print(f"\n  Weighting scheme:")
    print(f"    PCA (multivariate patterns):  {PCA_WEIGHT*100:.0f}%")
    print(f"    MI (non-linear predictive):   {MI_WEIGHT*100:.0f}%")
    
    print(f"\n  Top 15 features by combined score:")
    print(f"  {'Rank':<6} {'Feature':<25} {'Combined':<10} {'PCA':<10} {'MI':<10}")
    print(f"  {'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for idx, row in df_combined.head(15).iterrows():
        print(f"  {int(row['rank']):<6} {row['feature']:<25} "
              f"{row['combined_score']:<10.4f} {row['pca_score_norm']:<10.4f} "
              f"{row['mi_score_norm']:<10.4f}")
    
    df_combined.to_csv(os.path.join(OUT_DIR, 'stage3_combined_scores.csv'), 
                       index=False)
    print(f"\n  Saved: stage3_combined_scores.csv")
    
    return df_combined


def stage3_select_top_features(df_combined):
    """Stage 3C: Select top N features by combined score"""
    
    print("\n" + "="*80)
    print("STAGE 3C: TOP FEATURE SELECTION")
    print("="*80)
    
    n_select = max(MIN_FEATURES, 
                   min(N_FEATURES_TARGET, MAX_FEATURES, len(df_combined)))
    
    top_features = df_combined.head(n_select)['feature'].tolist()
    
    print(f"  Target features: {N_FEATURES_TARGET}")
    print(f"  Bounds: [{MIN_FEATURES}, {MAX_FEATURES}]")
    print(f"  Selected: {len(top_features)} features")
    
    return top_features


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: LASSO REGULARIZATION REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def stage4_lasso_refinement(df, features, df_combined):
    """Stage 4: LASSO-based automatic redundancy removal (CALIBRATION SET ONLY)"""
    
    print("\n" + "="*80)
    print("STAGE 4: LASSO REGULARIZATION REFINEMENT (CALIBRATION SET ONLY)")
    print("="*80)
    print(f"  Input features: {len(features)}")
    
    X = df[features].values
    y = df[TARGET_COL].values
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print(f"\n  Running LassoCV (5-fold cross-validation)...")
    print(f"    Testing 100 alpha values...")
    
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=10000, 
                    n_alphas=100, n_jobs=-1)
    lasso.fit(X_scaled, y)
    
    print(f"\n  LASSO Results:")
    print(f"    Optimal alpha:  {lasso.alpha_:.6f}")
    print(f"    Training R²:    {lasso.score(X_scaled, y):.4f}")
    
    non_zero_mask = lasso.coef_ != 0
    lasso_features = [f for f, keep in zip(features, non_zero_mask) if keep]
    lasso_coefs = lasso.coef_[non_zero_mask]
    
    n_removed = len(features) - len(lasso_features)
    print(f"\n    Features retained:  {len(lasso_features)}")
    print(f"    Features removed:   {n_removed} (coefficient shrunk to 0)")
    
    # Fallback if LASSO is too aggressive
    if len(lasso_features) < MIN_FEATURES:
        print(f"\n  ⚠️  LASSO selected < {MIN_FEATURES} features")
        print(f"      Fallback: keeping top {MIN_FEATURES} by combined score")
        lasso_features = df_combined.head(MIN_FEATURES)['feature'].tolist()
        lasso_mask = [f in lasso_features for f in features]
        lasso_coefs = lasso.coef_[lasso_mask]
    
    df_lasso = pd.DataFrame({
        'feature': lasso_features,
        'lasso_coefficient': lasso_coefs if len(lasso_coefs) > 0 else [0]*len(lasso_features),
        'abs_coefficient': np.abs(lasso_coefs) if len(lasso_coefs) > 0 else [0]*len(lasso_features)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\n  Top 10 features by |LASSO coefficient|:")
    print(f"  {'Rank':<6} {'Feature':<30} {'Coefficient':<15} {'|Coefficient|'}")
    print(f"  {'-'*6} {'-'*30} {'-'*15} {'-'*15}")
    for idx, row in df_lasso.head(10).iterrows():
        print(f"  {idx+1:<6} {row['feature']:<30} "
              f"{row['lasso_coefficient']:<15.4f} {row['abs_coefficient']:<15.4f}")
    
    df_lasso.to_csv(os.path.join(OUT_DIR, 'stage4_lasso_coefficients.csv'), 
                    index=False)
    print(f"\n  Saved: stage4_lasso_coefficients.csv")
    
    return lasso_features, lasso, df_lasso


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY TO VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════════
def apply_to_validation(validation_csv, selected_features):
    """Apply selected features to validation set (NO feature selection on validation!)"""
    
    print("\n" + "="*80)
    print("APPLYING SELECTED FEATURES TO VALIDATION SET")
    print("="*80)
    print(f"  ⚠️  IMPORTANT: Validation set was NOT used for feature selection!")
    print(f"  ✅  Features selected from calibration set only")
    
    df_valid = pd.read_csv(validation_csv)
    
    print(f"\n  Validation set: {len(df_valid)} samples")
    print(f"  Extracting {len(selected_features)} selected features...")
    
    # Check if all selected features exist
    missing_features = [f for f in selected_features if f not in df_valid.columns]
    if missing_features:
        print(f"\n  ⚠️  WARNING: {len(missing_features)} features missing in validation set:")
        for f in missing_features:
            print(f"    - {f}")
        selected_features = [f for f in selected_features if f in df_valid.columns]
        print(f"  Using {len(selected_features)} available features")
    
    # Extract selected features + metadata
    cols_to_keep = [ID_COL, TARGET_COL] + selected_features
    
    # Add metadata if exists
    for col in ['moisture', 'soil_type']:
        if col in df_valid.columns and col not in cols_to_keep:
            cols_to_keep.append(col)
    
    df_valid_selected = df_valid[cols_to_keep].copy()
    
    output_path = os.path.join(OUT_DIR, 'validation_selected_features.csv')
    df_valid_selected.to_csv(output_path, index=False)
    
    print(f"\n  Saved: validation_selected_features.csv ({len(df_valid_selected)} × {len(df_valid_selected.columns)})")
    
    return df_valid_selected


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def create_visualization(df_combined, final_features):
    """Create 4-panel visualization"""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        'Multi-Criteria Feature Selection (CALIBRATION SET ONLY)\n'
        f'PCA-MI Hybrid (70-30) + LASSO  |  '
        f'Final: {len(final_features)} features selected',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    # Panel 1: Combined Scores
    ax = axes[0, 0]
    df_plot = df_combined.head(20)
    colors = ['#2E7D32' if f in final_features else '#BDBDBD' 
              for f in df_plot['feature']]
    
    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot['combined_score'], color=colors, 
            edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'], fontsize=9)
    ax.set_xlabel('Combined Score (70% PCA + 30% MI)', 
                  fontweight='bold', fontsize=11)
    ax.set_title('Top 20 Features by Multi-Criteria Score\n'
                 '(Green = Selected by LASSO)', 
                 fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3, ls=':')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    
    # Panel 2: PCA vs MI Scatter
    ax = axes[0, 1]
    selected_mask = df_combined['feature'].isin(final_features)
    
    ax.scatter(df_combined[~selected_mask]['pca_score_norm'],
               df_combined[~selected_mask]['mi_score_norm'],
               s=70, c='#BDBDBD', alpha=0.5, edgecolors='black', 
               linewidth=0.5, label='Not Selected', zorder=2)
    
    ax.scatter(df_combined[selected_mask]['pca_score_norm'],
               df_combined[selected_mask]['mi_score_norm'],
               s=140, c='#2E7D32', alpha=0.9, edgecolors='black',
               linewidth=1.5, label='Selected', marker='s', zorder=3)
    
    for idx, row in df_combined[selected_mask].iterrows():
        ax.annotate(row['feature'],
                   (row['pca_score_norm'], row['mi_score_norm']),
                   fontsize=7, alpha=0.75, xytext=(4, 4),
                   textcoords='offset points')
    
    ax.set_xlabel('PCA Score (Multivariate Patterns)', 
                  fontweight='bold', fontsize=11)
    ax.set_ylabel('MI Score (Non-linear Predictive Power)', 
                  fontweight='bold', fontsize=11)
    ax.set_title('Feature Selection Space',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3, ls=':')
    ax.spines[['top', 'right']].set_visible(False)
    
    # Panel 3: Score Composition
    ax = axes[1, 0]
    df_sel = df_combined[df_combined['feature'].isin(final_features)].copy()
    df_sel = df_sel.sort_values('combined_score', ascending=True)
    
    y_pos = np.arange(len(df_sel))
    pca_contrib = df_sel['pca_score_norm'] * PCA_WEIGHT
    mi_contrib = df_sel['mi_score_norm'] * MI_WEIGHT
    
    ax.barh(y_pos, pca_contrib, height=0.7, 
            label=f'PCA ({int(PCA_WEIGHT*100)}%)',
            color='#1565C0', edgecolor='black', linewidth=1, alpha=0.85)
    ax.barh(y_pos, mi_contrib, height=0.7, left=pca_contrib,
            label=f'MI ({int(MI_WEIGHT*100)}%)',
            color='#FF6F00', edgecolor='black', linewidth=1, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sel['feature'], fontsize=8)
    ax.set_xlabel('Score Contribution', fontweight='bold', fontsize=11)
    ax.set_title('Score Composition (Selected Features)',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3, ls=':')
    ax.spines[['top', 'right']].set_visible(False)
    
    # Panel 4: Score Distribution
    ax = axes[1, 1]
    
    ax.hist(df_combined['combined_score'], bins=25, alpha=0.5,
            color='#757575', edgecolor='black', linewidth=0.8,
            label='All Features', zorder=2)
    
    selected_scores = df_combined[selected_mask]['combined_score']
    ax.hist(selected_scores, bins=12, alpha=0.85,
            color='#2E7D32', edgecolor='black', linewidth=1.2,
            label=f'Selected (n={len(final_features)})', zorder=3)
    
    threshold = selected_scores.min()
    ax.axvline(threshold, color='#C62828', linestyle='--', linewidth=2.5,
               label=f'Threshold ({threshold:.3f})', zorder=4)
    
    ax.set_xlabel('Combined Score', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Score Distribution',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls=':')
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'feature_selection_plots.png'),
                dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: feature_selection_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def write_summary_report(df_calib, df_valid, candidate_features, 
                        filtered_features, top_features, final_features, 
                        df_combined, df_lasso):
    """Generate comprehensive summary report"""
    
    df_final = df_combined[df_combined['feature'].isin(final_features)].copy()
    df_final = df_final.merge(df_lasso[['feature', 'lasso_coefficient']], 
                              on='feature', how='left')
    df_final = df_final.sort_values('combined_score', ascending=False)
    
    lines = [
        "="*80,
        "MULTI-CRITERIA FEATURE SELECTION (MCFS)",
        "CALIBRATION SET ONLY - NO DATA LEAKAGE",
        "="*80,
        "",
        "── PROPER CROSS-VALIDATION METHODOLOGY ──────────────────────────────",
        "  ✅ Feature selection performed on CALIBRATION SET ONLY",
        "  ✅ Validation set was NOT used for feature selection",
        "  ✅ Same features applied to validation set",
        "  ✅ No data leakage - publication-ready approach",
        "",
        "── DATASET ─────────────────────────────────────────────────────────",
        f"  Calibration set:  {len(df_calib)} samples (used for feature selection)",
        f"  Validation set:   {len(df_valid)} samples (features applied only)",
        f"  Target variable:  {TARGET_COL}",
        f"  Candidate features: {len(candidate_features)}",
        "",
        "── STAGE 1: QUALITY PRE-FILTERING ──────────────────────────────────",
        f"  Thresholds:",
        f"    Min variance:        > {MIN_VARIANCE}",
        f"    Max missing:         < {MAX_MISSING_PCT*100}%",
        f"    Significance:        p < {P_VALUE_THRESHOLD}",
        "",
        f"  Results (calibration set):",
        f"    Input features:      {len(candidate_features)}",
        f"    Passed filter:       {len(filtered_features)}",
        f"    Removed:             {len(candidate_features) - len(filtered_features)}",
        "",
        "── STAGE 2: MUTUAL INFORMATION ─────────────────────────────────────",
        f"  Method: sklearn.feature_selection.mutual_info_regression",
        f"  Random state: {SEED}",
        f"  N neighbors: 5",
        f"  Data: Calibration set only ({len(df_calib)} samples)",
        "",
        "── STAGE 3: PCA-MI HYBRID SCORING ──────────────────────────────────",
        f"  Weighting scheme:",
        f"    PCA weight (multivariate):    {PCA_WEIGHT*100:.0f}%",
        f"    MI weight (non-linear):       {MI_WEIGHT*100:.0f}%",
        "",
        f"  Combined Score = {PCA_WEIGHT} × PCA_Score + {MI_WEIGHT} × MI_Score",
        "",
        f"  Feature selection:",
        f"    Target features:     {N_FEATURES_TARGET}",
        f"    Bounds:              [{MIN_FEATURES}, {MAX_FEATURES}]",
        f"    Selected for LASSO:  {len(top_features)}",
        "",
        "── STAGE 4: LASSO REGULARIZATION ───────────────────────────────────",
        f"  Method: LassoCV (5-fold cross-validation)",
        f"  Alpha values tested: 100",
        f"  Data: Calibration set only",
        "",
        f"  Results:",
        f"    Input to LASSO:      {len(top_features)}",
        f"    Final selected:      {len(final_features)}",
        f"    Removed by LASSO:    {len(top_features) - len(final_features)}",
        "",
        "── FINAL SELECTED FEATURES ─────────────────────────────────────────",
        f"  Total: {len(final_features)} features",
        f"  Reduction: {(1 - len(final_features)/len(candidate_features))*100:.1f}% "
        f"({len(candidate_features)} → {len(final_features)})",
        "",
        "  ✅ These features selected from CALIBRATION SET",
        "  ✅ Applied to VALIDATION SET (no leakage)",
        "",
        "  Ranked by combined score:",
        "",
        f"  {'Rank':<6} {'Feature':<25} {'Combined':<10} {'PCA':<10} {'MI':<10} {'LASSO Coef'}",
        f"  {'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*12}",
    ]
    
    for idx, row in df_final.iterrows():
        lines.append(
            f"  {int(row['rank']):<6} {row['feature']:<25} "
            f"{row['combined_score']:<10.4f} {row['pca_score_norm']:<10.4f} "
            f"{row['mi_score_norm']:<10.4f} {row['lasso_coefficient']:<12.4f}"
        )
    
    lines.extend([
        "",
        "── OUTPUT FILES ────────────────────────────────────────────────────",
        f"  CALIBRATION SET:",
        f"    calibration_selected_features.csv - calibration with selected features",
        f"    stage1_removal_log.csv            - removed features with reasons",
        f"    stage2_mutual_information.csv     - MI scores",
        f"    stage3_pca_scores.csv             - PCA scores",
        f"    stage3_combined_scores.csv        - PCA-MI hybrid scores",
        f"    stage4_lasso_coefficients.csv     - LASSO coefficients",
        "",
        f"  VALIDATION SET:",
        f"    validation_selected_features.csv  - validation with SAME features",
        "",
        f"  FEATURE LIST:",
        f"    selected_features_list.txt        - just feature names",
        "",
        f"  REPORTS:",
        f"    feature_selection_plots.png       - 4-panel visualization",
        f"    feature_selection_report.txt      - this report",
        "",
        "── READY FOR MODEL TRAINING ────────────────────────────────────────",
        "  Next steps:",
        "    1. Train models on calibration_selected_features.csv",
        "    2. Validate on validation_selected_features.csv",
        "    3. Both datasets have SAME {len(final_features)} features",
        "",
        "="*80,
    ])
    
    report = "\n".join(lines)
    print("\n" + report)
    
    with open(os.path.join(OUT_DIR, 'feature_selection_report.txt'), 'w') as f:
        f.write(report)
    print(f"\n  Saved: feature_selection_report.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Main execution pipeline"""
    
    print("\n" + "="*80)
    print("MULTI-CRITERIA FEATURE SELECTION")
    print("CALIBRATION SET ONLY (Proper Cross-Validation)")
    print("="*80)
    print(f"\nCalibration: {CALIB_CSV}")
    print(f"Validation:  {VALID_CSV}")
    print(f"Output:      {OUT_DIR}/")
    
    # Load calibration set
    print("\n" + "="*80)
    print("LOADING CALIBRATION SET")
    print("="*80)
    df_calib = pd.read_csv(CALIB_CSV)
    print(f"  Calibration set: {len(df_calib)} samples × {len(df_calib.columns)} columns")
    
    # Get candidate features
    candidate_features = [col for col in df_calib.columns 
                         if col not in EXCLUDE_COLS]
    candidate_features = [col for col in candidate_features 
                         if df_calib[col].dtype in [np.float64, np.int64]]
    
    print(f"  Candidate features: {len(candidate_features)}")
    print(f"  Target: {TARGET_COL}")
    
    # Run 4-stage MCFS on calibration set ONLY
    filtered_features, _ = stage1_quality_filter(df_calib, candidate_features)
    df_mi, _ = stage2_mutual_information(df_calib, filtered_features)
    df_pca, _, _ = stage3_pca_scoring(df_calib, filtered_features)
    df_combined = stage3_combine_scores(df_pca, df_mi)
    top_features = stage3_select_top_features(df_combined)
    final_features, _, df_lasso = stage4_lasso_refinement(
        df_calib, top_features, df_combined
    )
    
    # Save calibration set with selected features
    print("\n" + "="*80)
    print("SAVING CALIBRATION SET WITH SELECTED FEATURES")
    print("="*80)
    
    cols_to_keep = [ID_COL, TARGET_COL] + final_features
    for col in ['moisture', 'soil_type']:
        if col in df_calib.columns and col not in cols_to_keep:
            cols_to_keep.append(col)
    
    df_calib_selected = df_calib[cols_to_keep].copy()
    calib_output = os.path.join(OUT_DIR, 'calibration_selected_features.csv')
    df_calib_selected.to_csv(calib_output, index=False)
    
    print(f"  Calibration dataset shape: {df_calib_selected.shape}")
    print(f"  Saved: calibration_selected_features.csv")
    
    # Apply to validation set
    df_valid_selected = apply_to_validation(VALID_CSV, final_features)
    
    # Save feature list
    feature_list_path = os.path.join(OUT_DIR, 'selected_features_list.txt')
    with open(feature_list_path, 'w') as f:
        f.write("SELECTED FEATURES (from calibration set)\n")
        f.write("="*60 + "\n\n")
        for i, feat in enumerate(final_features, 1):
            f.write(f"{i:2d}. {feat}\n")
    print(f"\n  Saved: selected_features_list.txt")
    
    # Create visualizations
    create_visualization(df_combined, final_features)
    
    # Generate summary report
    df_valid = pd.read_csv(VALID_CSV)
    write_summary_report(df_calib, df_valid, candidate_features, 
                        filtered_features, top_features, final_features, 
                        df_combined, df_lasso)
    
    print("\n" + "="*80)
    print("✅ FEATURE SELECTION COMPLETE (NO DATA LEAKAGE)")
    print("="*80)
    print(f"\n  Selected {len(final_features)} features from {len(candidate_features)} candidates")
    print(f"  Reduction: {(1 - len(final_features)/len(candidate_features))*100:.1f}%")
    print(f"\n  ✅ Calibration: {len(df_calib_selected)} samples × {len(final_features)} features")
    print(f"  ✅ Validation:  {len(df_valid_selected)} samples × {len(final_features)} features")
    print(f"\n  All outputs saved to: {OUT_DIR}/")
    print("\n  🎯 READY FOR MODEL TRAINING!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()