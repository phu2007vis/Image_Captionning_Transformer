import os
import shutil
import argparse

def process_images(root_folder):
    image_folder = os.path.join(root_folder, "images")
    save_folder = os.path.join(root_folder, "labels")
    
    # Remove and recreate the save folder
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    
    # Process each image in the image folder
    for image_name in os.listdir(image_folder):
        id = os.path.splitext(image_name)[0]
        plate_text = id.split('_')[0]
        label_path = os.path.join(save_folder, f"{id}.txt")
        
        with open(label_path, 'w') as f:
            f.write(plate_text)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images and generate label files.")
    parser.add_argument(
        '--root_folder', 
        type=str, 
        # default="/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val",
        help="Path to the root folder containing images and where labels will be saved",
        required=True
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the images with the provided root folder
    process_images(args.root_folder)

if __name__ == "__main__":
    main()