import os 
from os.path import splitext,join
from os import listdir
from tqdm import tqdm
import shutil


input_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/split_ted2/train"
save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/split_ted2_doi_ten/train"

img_folder = join(input_folder,"images")
txt_folder = join(input_folder,"labels")

img_save_folder = join(save_folder,"images")
os.makedirs(img_save_folder,exist_ok=True)
txt_save_folder = join(save_folder,"labels")
os.makedirs(txt_save_folder,exist_ok=True)

img_names = listdir(img_folder)

for img_name in tqdm(img_names):
    img_path = join(img_folder, img_name)
    txt_path = join(txt_folder, splitext(img_name)[0] + ".txt")

    with open(txt_path, 'r') as f:
        plate = f.readline().strip()
        
    img_save_path = join(img_save_folder,f"{plate}_{ splitext(img_name)[0]}.jpg")
    new_id = f"{plate}_{ splitext(img_name)[0]}"
    txt_save_path = join(txt_save_folder,new_id+ ".txt")
    
    shutil.copy(img_path, img_save_path)
    shutil.copy(txt_path, txt_save_path)