import os
import shutil
from os.path import join

root_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1/images"
output_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number/images"

os.makedirs(output_folder,exist_ok=True)


for image_name in os.listdir(root_folder):
    id = image_name.split("_")[0]
    image_path = os.path.join(root_folder, image_name)
    save_path = os.path.join(output_folder,image_name)
    if id[3].isdigit():
        shutil.copy(image_path, save_path)
        