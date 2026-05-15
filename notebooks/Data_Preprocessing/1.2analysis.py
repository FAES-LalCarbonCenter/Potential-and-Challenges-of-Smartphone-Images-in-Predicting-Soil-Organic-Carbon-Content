    #this is the code written to resize the field images to 224x224 pixels - 206 -268, 361-437(Field Images manually croppped)



    import os
    import cv2
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm

    # Define input and output directories
    input_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step1_input/Cropped Field Images_206----436")
    output_dir = Path("/Users/dharamkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversity/VSCode_image_processing_project/data/Step1_output/resize_field(206-268,361-436)")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    image_files = sorted(list(input_dir.glob("*.[jJ][pP][gG]")))
    print(f"📁 Found {len(image_files)} image(s) in the input directory.")

    # Loop through each image
    resized_count = 0
    for img_file in tqdm(image_files):
        print(f"🔍 Processing: {img_file.name}")
        
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"❌ Could not load {img_file.name}")
            continue

        # Resize the image
        resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Define the output path
        out_path = output_dir / img_file.name

        # Try writing the image
        success = cv2.imwrite(str(out_path), resized_img)
        if success:
            print(f"✅ Saved: {out_path.name}")
            resized_count += 1
        else:
            print(f"⚠️ Failed to save: {out_path.name}")

    print(f"\n✅ Total resized and saved images: {resized_count}")
