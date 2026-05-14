"""
SOIL TYPE ANALYSIS - STEP 1: Add Soil Type Categories
======================================================
# """
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project_pipeline")

MOISTURE_META = PROJECT_ROOT / "notebooks/objective_2/obj2_output_data/step1_output/moisture_metadata.csv"
OUT_DIR = PROJECT_ROOT / "notebooks/objective_3/obj3_output_data/step1_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "field_lab_metadata.csv"

def label_image_type(img_no):
    if (1 <= img_no <= 19) or (206 <= img_no <= 268) or (361 <= img_no <= 436):
        return "Field"
    elif (20 <= img_no <= 205) or (269 <= img_no <= 360) or (437 <= img_no <= 731):
        return "Lab"
    return "Unknown"

print("="*80)
print("FIELD vs LAB - STEP 1")
print("="*80)

df = pd.read_csv(MOISTURE_META)
df["image_type"] = df["image_no"].apply(label_image_type)

print(f"\nTotal: {len(df)}")
print(df["image_type"].value_counts())

df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved: {OUT_CSV}")