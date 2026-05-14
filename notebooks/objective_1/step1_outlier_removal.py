#The only difference from the paper's code is the covariance inverse — they used np.linalg.inv, which would crash on your data because your 33 features are correlated (rank-deficient matrix). The script uses np.linalg.pinv (pseudo-inverse) which handles this correctly while keeping everything else identical.

#removing outliers from entire dataset, soc and moisture include ( X variable and Y variable included)

"""
Mahalanobis Distance Outlier Removal + Kennard-Stone Split
============================================================
Dataset : soil image features + SOC metadata
Merge key: 'Numeric numbers' (features) == 'image_no' (metadata)

Pipeline (De Maesschalck et al., 2000) — same approach as reference code:
  1. Merge the two CSV files into a master dataset
  2. Select all numeric feature columns
  3. Compute Mahalanobis distance for each sample
     - Uses np.linalg.pinv (pseudo-inverse) to handle correlated features
       that cause a non-full-rank covariance matrix
  4. Flag outliers: D > sqrt(chi2(p=0.99, df=n_vars))
     - 99% confidence threshold
  5. Remove outliers, save cleaned dataset
  6. Split clean data via Kennard-Stone algorithm:
       Calibration : 70%
       Validation  : 30%

Outputs saved to ./outlier_output/
  master_merged.csv           — merged dataset before outlier removal
  outlier_detection_results.csv — full dataset with distance + flag columns
  soil_features_cleaned.csv   — cleaned dataset (outliers removed)
  calibration_set.csv         — 70% calibration (Kennard-Stone)
  validation_set.csv          — 30% validation
  mahalanobis_plot.png        — distance plot with threshold line
  summary_report.txt          — summary

Requirements: pip install pandas numpy matplotlib scipy scikit-learn
Usage:        python mahalanobis_final.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ── PATHS  (edit to match your local file locations) ──────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/data/raw/soil_image_features_without_commas.csv')
METADATA_CSV = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/data/raw/image_with_soc_metadata.csv')
OUT_DIR      = os.path.join(HERE, '/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline/notebooks/objective_1/output_data/step1_output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
# Columns to exclude from Mahalanobis (identifiers, not features)
EXCLUDE_COLS = ['filename', 'Numeric numbers', 'image_path', 'image_no', 'soil_type']

# Confidence level for chi-squared outlier threshold
CONFIDENCE   = 0.99     # 99% — matches De Maesschalck et al. (2000)

# Kennard-Stone split
CALIB_RATIO  = 0.70     # 70% calibration, 30% validation
SEED         = 42


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & MERGE
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_merge():
    print("=" * 60)
    print("STEP 1: Loading and merging files")
    print("=" * 60)
    feat   = pd.read_csv(FEATURES_CSV)
    meta   = pd.read_csv(METADATA_CSV)
    merged = pd.merge(feat, meta,
                      left_on='Numeric numbers', right_on='image_no',
                      how='inner')
    print(f"  Features file : {feat.shape[0]} rows x {feat.shape[1]} cols")
    print(f"  Metadata file : {meta.shape[0]} rows x {meta.shape[1]} cols")
    print(f"  Merged master : {merged.shape[0]} rows x {merged.shape[1]} cols")
    merged.to_csv(os.path.join(OUT_DIR, 'master_merged.csv'), index=False)
    print(f"  Saved: master_merged.csv")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MAHALANOBIS DISTANCE OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def mahalanobis_outlier_removal(df):
    print("\n" + "=" * 60)
    print("STEP 2: Mahalanobis Distance Outlier Detection")
    print("        (De Maesschalck et al., 2000)")
    print("=" * 60)

    # Select numeric feature columns only (drop identifier columns)
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop(
        columns=[c for c in EXCLUDE_COLS if c in numeric_df.columns])

    n_vars = numeric_df.shape[1]
    n_obs  = numeric_df.shape[0]
    print(f"  Variables : {n_vars}")
    print(f"  Samples   : {n_obs}")
    print(f"  Columns   : {list(numeric_df.columns)}")

    # Compute mean vector and covariance matrix
    mean_vec    = numeric_df.mean().values
    cov_matrix  = np.cov(numeric_df.T)

    # Use pseudo-inverse — handles non-full-rank covariance (correlated features)
    inv_cov = np.linalg.pinv(cov_matrix)
    print(f"\n  Computing Mahalanobis distances ...")
    distances = numeric_df.apply(
        lambda row: mahalanobis(row, mean_vec, inv_cov), axis=1)

    # 99% chi-squared threshold (same convention as reference paper)
    threshold = chi2.ppf(CONFIDENCE, df=n_vars)
    threshold_d = np.sqrt(threshold)     # convert to distance scale
    print(f"  Chi-sq threshold (99%, df={n_vars}): {threshold:.4f}")
    print(f"  Distance threshold (sqrt):           {threshold_d:.4f}")

    # Flag outliers: D > sqrt(chi2 threshold)
    df = df.copy()
    df['Mahalanobis_Distance'] = distances.values
    df['Is_Outlier']           = distances > threshold_d

    outlier_count = df['Is_Outlier'].sum()
    print(f"\n  Outliers detected : {outlier_count}")
    print(f"  Clean samples     : {n_obs - outlier_count}")

    # Save full results with flags
    full_out = os.path.join(OUT_DIR, 'outlier_detection_results.csv')
    df.to_csv(full_out, index=False)
    print(f"  Saved: outlier_detection_results.csv")

    # Save cleaned dataset
    df_cleaned = df[~df['Is_Outlier']].drop(
        columns=['Is_Outlier']).reset_index(drop=True)
    clean_out = os.path.join(OUT_DIR, 'soil_features_cleaned.csv')
    df_cleaned.to_csv(clean_out, index=False)
    print(f"  Saved: soil_features_cleaned.csv  ({len(df_cleaned)} rows)")

    # Outlier image numbers
    df_outliers = df[df['Is_Outlier']].copy()
    print(f"\n  Outlier image numbers:")
    print(f"    {sorted(df_outliers['Numeric numbers'].tolist())}")

    _plot_mahalanobis(distances.values, threshold_d, df['Is_Outlier'].values,
                      n_obs, n_vars)

    return df_cleaned, df_outliers, distances.values, threshold_d, numeric_df.columns.tolist()


def _plot_mahalanobis(distances, threshold, is_outlier, n, n_vars):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Mahalanobis Distance Outlier Detection  (De Maesschalck et al., 2000)\n'
        f'n={n} images  |  {n_vars} features  |  '
        f'99% confidence threshold D={threshold:.4f}  |  '
        f'Outliers removed: {is_outlier.sum()}',
        fontsize=12, fontweight='bold'
    )

    idx = np.arange(n)

    # Index plot
    ax = axes[0]
    ax.scatter(idx[~is_outlier], distances[~is_outlier],
               color='#1565C0', s=14, alpha=0.55, label='Clean', zorder=3)
    ax.scatter(idx[is_outlier], distances[is_outlier],
               color='#C62828', s=55, alpha=0.95, zorder=5,
               label=f'Outlier (n={is_outlier.sum()})')
    ax.axhline(threshold, color='#E65100', lw=2.2, ls='--',
               label=f'99% threshold (D={threshold:.2f})', zorder=6)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Mahalanobis Distance (D)', fontsize=11)
    ax.set_title('Distance per Sample', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.98, 0.97,
            f'Outliers : {is_outlier.sum()}\nClean    : {(~is_outlier).sum()}',
            transform=ax.transAxes, fontsize=9.5, va='top', ha='right',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor='#AAAAAA', alpha=0.92))

    # Histogram
    ax2 = axes[1]
    ax2.hist(distances[~is_outlier], bins=35, color='#1565C0', alpha=0.70,
             edgecolor='white', lw=0.5, label='Clean')
    if is_outlier.sum() > 0:
        ax2.hist(distances[is_outlier], bins=8, color='#C62828', alpha=0.90,
                 edgecolor='white', lw=0.5, label=f'Outlier (n={is_outlier.sum()})')
    ax2.axvline(threshold, color='#E65100', lw=2.2, ls='--',
                label=f'99% threshold (D={threshold:.2f})')
    ax2.set_xlabel('Mahalanobis Distance (D)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distance Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, ls=':', alpha=0.35)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mahalanobis_plot.png'),
                dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mahalanobis_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — KENNARD-STONE SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
def kennard_stone(X_scaled, n_select):
    n           = X_scaled.shape[0]
    dist_matrix = np.sum(
        (X_scaled[:, None, :] - X_scaled[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected  = [i, j]
    remaining = list(range(n))
    remaining.remove(i); remaining.remove(j)
    while len(selected) < n_select:
        rem_arr   = np.array(remaining)
        sel_arr   = np.array(selected)
        min_dists = dist_matrix[np.ix_(rem_arr, sel_arr)].min(axis=1)
        best      = remaining[np.argmax(min_dists)]
        selected.append(best)
        remaining.remove(best)
    return selected, remaining


def kennard_stone_split(df_clean, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 3: Kennard-Stone Calibration / Validation Split")
    print("=" * 60)
    n_total = len(df_clean)
    n_calib = int(round(n_total * CALIB_RATIO))
    n_valid = n_total - n_calib
    print(f"  Total clean samples : {n_total}")
    print(f"  Calibration  (70%)  : {n_calib}")
    print(f"  Validation   (30%)  : {n_valid}")

    X_scaled = StandardScaler().fit_transform(
        df_clean[feature_cols].values.astype(float))
    print("  Running Kennard-Stone algorithm ...")
    calib_idx, valid_idx = kennard_stone(X_scaled, n_calib)

    df_calib = df_clean.iloc[calib_idx].reset_index(drop=True)
    df_valid = df_clean.iloc[valid_idx].reset_index(drop=True)
    df_calib.to_csv(os.path.join(OUT_DIR, 'calibration_set.csv'), index=False)
    df_valid.to_csv(os.path.join(OUT_DIR, 'validation_set.csv'),  index=False)
    print(f"  Saved: calibration_set.csv  ({len(df_calib)} rows)")
    print(f"  Saved: validation_set.csv   ({len(df_valid)} rows)")
    return df_calib, df_valid


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def write_summary(merged, df_clean, df_outliers, df_calib, df_valid,
                  distances, threshold, n_vars):
    outlier_ids = sorted(df_outliers['Numeric numbers'].tolist())
    lines = [
        "=" * 60,
        "MAHALANOBIS OUTLIER REMOVAL - SUMMARY REPORT",
        "Reference: De Maesschalck et al. (2000)",
        "=" * 60,
        "",
        "-- DATASET -------------------------------------------",
        f"  Total images (master merged)  : {len(merged)}",
        f"  Feature variables             : {n_vars}",
        "",
        "-- MAHALANOBIS CONFIGURATION -------------------------",
        f"  Covariance inverse : np.linalg.pinv (pseudo-inverse)",
        f"  Confidence level   : 99%",
        f"  Chi-sq threshold   : chi2.ppf(0.99, df={n_vars}) = {chi2.ppf(CONFIDENCE, df=n_vars):.4f}",
        f"  Distance threshold : sqrt(chi2) = {threshold:.4f}",
        f"  Outlier criterion  : D > {threshold:.4f}",
        "",
        "-- OUTLIER DETECTION RESULTS -------------------------",
        f"  Outliers removed : {len(df_outliers)}",
        f"  Clean samples    : {len(df_clean)}",
        "",
        f"  Outlier image numbers:",
        f"    {outlier_ids}",
        "",
        "-- KENNARD-STONE SPLIT --------------------------------",
        f"  Calibration set : {len(df_calib)} images  ({100*len(df_calib)/len(df_clean):.1f}%)",
        f"  Validation set  : {len(df_valid)} images  ({100*len(df_valid)/len(df_clean):.1f}%)",
        "",
        "-- OUTPUT FILES --------------------------------------",
        f"  master_merged.csv              - {len(merged)} rows",
        f"  outlier_detection_results.csv  - {len(merged)} rows + distance & flag cols",
        f"  soil_features_cleaned.csv      - {len(df_clean)} rows (outliers removed)",
        f"  calibration_set.csv            - {len(df_calib)} rows (Kennard-Stone 70%)",
        f"  validation_set.csv             - {len(df_valid)} rows (Kennard-Stone 30%)",
        f"  mahalanobis_plot.png           - distance plot",
        "",
        "-- DISTANCE STATISTICS --------------------------------",
        f"  Min    : {distances.min():.4f}",
        f"  Max    : {distances.max():.4f}",
        f"  Mean   : {distances.mean():.4f}",
        f"  Median : {np.median(distances):.4f}",
        f"  Std    : {distances.std():.4f}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, 'summary_report.txt'), 'w') as f:
        f.write(report)
    print(f"\n  Saved: summary_report.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    merged = load_and_merge()
    df_clean, df_outliers, distances, threshold, feature_cols = \
        mahalanobis_outlier_removal(merged)
    df_calib, df_valid = kennard_stone_split(df_clean, feature_cols)
    write_summary(merged, df_clean, df_outliers, df_calib, df_valid,
                  distances, threshold, len(feature_cols))
    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == '__main__':
    main()