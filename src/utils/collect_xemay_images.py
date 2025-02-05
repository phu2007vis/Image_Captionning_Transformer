import os
import shutil


input_folders = ['/work/21013187/phuoc/License_Plate/data/LP_detection/images/train','/work/21013187/phuoc/License_Plate/data/LP_detection/images/val']

output_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/xemay"
os.makedirs(output_folder,exist_ok=True)
from tqdm import tqdm

for input_folder in input_folders:
    img_names =  os.listdir(input_folder)

    for img_name in tqdm(img_names):
        img_path = os.path.join(input_folder, img_name)
        if img_name.startswith("xemay"):
            
            new_img_name = f"{os.path.splitext(img_name)[0]}.jpg"
            new_img_path = os.path.join(output_folder, new_img_name)
            shutil.copy(img_path, new_img_path)