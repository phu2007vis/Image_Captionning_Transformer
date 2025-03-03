import os
import shutil
from os.path import join
import pandas as pd
import argparse

def process_images_to_csv(root_folder):
    img_folder = join(root_folder, "images")
    csv_output_file = join(root_folder, "labels.csv")
    
    # Initialize dictionary for storing data
    csv_map = {
        'names': [],
        'plate': []
    }
    
    # Process each image in the image folder
    for img_name in os.listdir(img_folder):
        plate_text = img_name.split("_")[0]
        csv_map['names'].append(img_name)
        csv_map['plate'].append(plate_text)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_map)
    df.to_csv(csv_output_file, index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images and generate a labels CSV file.")
    parser.add_argument(
        '--root_folder',
        type=str,
        default="/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/",
        help="Path to the root folder containing images and where labels.csv will be saved"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the images and generate CSV with the provided root folder
    process_images_to_csv(args.root_folder)

if __name__ == "__main__":
    main()