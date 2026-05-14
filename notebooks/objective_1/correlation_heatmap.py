"""
Correlation heatmap for 12 selected features + SOC
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline")

# Load data with 12 features
CALIB_FEATURES = PROJECT_ROOT / "notebooks/objective_1/output_data/step2_output/calibration_selected_features.csv"
OUT_DIR = PROJECT_ROOT / "notebooks/objective_1/output_data/output_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CALIB_FEATURES)

# 12 selected features + SOC
features = ['entropy', 'median_h', 'mean_s', 'lab_b', 'luv_u', 
            'CI', 'luv_v', 'median_s', 'median_r', 'SI', 
            'median_v', 'mean_r', 'soc']

df_corr = df[features]

# Compute correlation matrix
corr = df_corr.corr()

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title('Correlation Matrix: 12 Selected Features + SOC', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

out_path = OUT_DIR / "correlation_heatmap_12features.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: {out_path}")

# Print SOC correlations sorted
print("\nSOC Correlations (sorted):")
soc_corr = corr['soc'].drop('soc').sort_values(ascending=False)
print(soc_corr)