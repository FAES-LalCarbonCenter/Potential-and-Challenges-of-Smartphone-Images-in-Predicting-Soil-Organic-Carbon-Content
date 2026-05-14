"""
MOISTURE ANALYSIS - STEP 1: Prepare Moisture Metadata
======================================================
Load cleaned calibration + validation sets from outlier removal
Add moisture categories (Dry, Moist, Wet)
Save combined metadata for moisture stratification analysis

Input:
  - outlier_output/calibration_set.csv (461 samples)
  - outlier_output/validation_set.csv (197 samples)

Output:
  - objective_2/obj2_output_data/step1_output/moisture_metadata.csv
  
Moisture Categories:
  - Dry: 0% ≤ moisture < 10%
  - Moist: 10% ≤ moisture ≤ 30%
  - Wet: moisture > 30%
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline")

CALIB_CSV = PROJECT_ROOT / "notebooks/objective_1//output_data/step1_output/calibration_set.csv"
VALID_CSV = PROJECT_ROOT / "notebooks/objective_1//output_data/step1_output/validation_set.csv"

OUT_DIR = PROJECT_ROOT / "notebooks/objective_2/obj2_output_data/step1_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "moisture_metadata.csv"

print("\n" + "="*80)
print("MOISTURE ANALYSIS - STEP 1: PREPARE METADATA")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# MOISTURE CATEGORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def moisture_to_category(moisture):
    """
    Categorize moisture content into Dry/Moist/Wet
    
    Categories (based on gravimetric moisture %):
      - Dry:   0% ≤ moisture < 10%
      - Moist: 10% ≤ moisture ≤ 30%
      - Wet:   moisture > 30%
    """
    if pd.isna(moisture):
        return "Unknown"
    
    if 0 <= moisture < 10:
        return "Dry"
    elif 10 <= moisture <= 30:
        return "Moist"
    elif moisture > 30:
        return "Wet"
    else:
        return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════
# LOAD CLEANED DATASETS
# ═══════════════════════════════════════════════════════════════════════════

print("\nLoading cleaned datasets...")

# Load calibration set (461 samples)
df_calib = pd.read_csv(CALIB_CSV)
df_calib['split'] = 'calibration'
print(f"  Calibration: {df_calib.shape}")

# Load validation set (197 samples)
df_valid = pd.read_csv(VALID_CSV)
df_valid['split'] = 'validation'
print(f"  Validation:  {df_valid.shape}")

# Combine
df = pd.concat([df_calib, df_valid], ignore_index=True)
print(f"  Combined:    {df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK FOR MOISTURE COLUMN
# ═══════════════════════════════════════════════════════════════════════════

print("\nChecking for moisture column...")

if 'moisture' not in df.columns:
    print("\n❌ ERROR: 'moisture' column not found in datasets!")
    print(f"   Available columns: {df.columns.tolist()}")
    print("\n⚠️  Moisture data may be in the original metadata file.")
    print("   You may need to merge with image_with_soc_metadata.csv")
    exit(1)

print("  ✅ Moisture column found")


# ═══════════════════════════════════════════════════════════════════════════
# CREATE MOISTURE CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════

print("\nCreating moisture categories...")

df['Moisture_Category'] = df['moisture'].apply(moisture_to_category)

print("\n  Moisture Category Distribution:")
print(df['Moisture_Category'].value_counts().sort_index())

print("\n  Distribution by Split:")
print(pd.crosstab(df['split'], df['Moisture_Category'], margins=True))


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS BY CATEGORY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("MOISTURE STATISTICS BY CATEGORY")
print("="*80)

for category in ['Dry', 'Moist', 'Wet']:
    subset = df[df['Moisture_Category'] == category]
    
    if len(subset) == 0:
        continue
    
    print(f"\n{category} (moisture: ", end="")
    if category == 'Dry':
        print("0-10%)")
    elif category == 'Moist':
        print("10-30%)")
    else:
        print(">30%)")
    
    print(f"  Total samples: {len(subset)}")
    print(f"    Calibration: {len(subset[subset['split'] == 'calibration'])}")
    print(f"    Validation:  {len(subset[subset['split'] == 'validation'])}")
    
    moisture_vals = subset['moisture'].dropna()
    if len(moisture_vals) > 0:
        print(f"  Moisture range: {moisture_vals.min():.2f}% - {moisture_vals.max():.2f}%")
        print(f"  Moisture mean:  {moisture_vals.mean():.2f}%")
    
    soc_vals = subset['soc'].dropna()
    if len(soc_vals) > 0:
        print(f"  SOC range: {soc_vals.min():.2f} - {soc_vals.max():.2f} g/kg")
        print(f"  SOC mean:  {soc_vals.mean():.2f} g/kg")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE METADATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SAVING METADATA")
print("="*80)

df.to_csv(OUT_CSV, index=False)

print(f"\n✅ Saved moisture metadata: {OUT_CSV}")
print(f"   Total samples: {len(df)}")
print(f"   Columns: {len(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n  Total samples: {len(df)}")
print(f"    Calibration: {len(df[df['split'] == 'calibration'])}")
print(f"    Validation:  {len(df[df['split'] == 'validation'])}")

print(f"\n  Moisture categories:")
for cat in ['Dry', 'Moist', 'Wet', 'Unknown']:
    count = len(df[df['Moisture_Category'] == cat])
    pct = 100 * count / len(df)
    print(f"    {cat:8s}: {count:3d} samples ({pct:5.1f}%)")

print("\n" + "="*80)
print("✅ STEP 1 COMPLETE")
print("="*80)
print(f"\nNext step: Run moisture_step2_generate_predictions.py")
print("="*80)