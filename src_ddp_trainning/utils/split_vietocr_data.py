import os
from os.path import join
from  os import listdir as ls
import shutil
import random 
import cv2

root_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/plate_xemay_ver2"
image_folder = join(root_folder, "images")
txt_folder = join(root_folder, "labels")

save_folder   = r"/work/21013187/phuoc/Image_Captionning_Transformer/data/plate_xemay_ver2_splited"
map_name = {}


image_names = ls(image_folder)

random.seed(42)
random.shuffle(image_names)

for i,img_name in enumerate(image_names):
    img_path = os.path.join(image_folder, img_name)
    id_img = os.path.splitext(img_name)[0]
    

    if i < len(image_names) *0.8:
        phase = 'train'
    else:
        phase = 'val'
    
    save_folder_sub = os.path.join(save_folder,phase,'images')
    os.makedirs(save_folder_sub,exist_ok=True)
    save_path = os.path.join(save_folder_sub,img_name)
    shutil.copy(img_path, save_path)

    id_txt = os.path.splitext(img_name)[0]
    txt_name = id_txt+".txt"
    txt_file = os.path.join(txt_folder,txt_name)
    
    save_folder_sub_txt = os.path.join(save_folder,phase,'labels')
    os.makedirs(save_folder_sub_txt,exist_ok=True)
    save_path = os.path.join(save_folder_sub_txt,txt_name)
    shutil.copy(txt_file,save_path)

    