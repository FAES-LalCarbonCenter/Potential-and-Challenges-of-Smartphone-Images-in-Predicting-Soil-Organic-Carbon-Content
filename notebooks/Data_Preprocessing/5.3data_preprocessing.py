#Creating master csv for CNN input

# Loads your Excel file with SOC and moisture values (e.g., image_processing_soc_values.xlsx).

# Adds the full image path by combining:

# The image folder path (until enhanced_sample_field)

# The Image No. column

# The .jpg file extension

# Outputs a CSV with the following columns:

# image_path

# image_no

# soc

# moisture



import pandas as pd
import os

# === Constants ===
excel_path = "data/Step5_input/image_processing_soc_values.xlsx"  # Adjust if needed
sheet_name = "Sheet1"
image_dir = "/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step2_output/enhanced_sample_field"
output_csv = "data/Step6_input/image_with_soc_metadata.csv"

# === Load Excel data ===
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# === Construct image paths ===
df["image_no"] = df["Image No."]
df["image_filename"] = df["Image No."].astype(str) + ".jpg"
df["image_path"] = df["image_filename"].apply(lambda x: os.path.join(image_dir, x))

# === Prepare final DataFrame ===
final_df = df[["image_path", "image_no", "SOC(%)", "Moisture Level (%)", "Soil type"]]
final_df.columns = ["image_path", "image_no", "soc", "moisture", "soil_type"]

# === Save to CSV ===
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
final_df.to_csv(output_csv, index=False)

print(f"✅ Metadata CSV with soil type saved to: {output_csv}")
