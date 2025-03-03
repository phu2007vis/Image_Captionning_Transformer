import os
from os.path import splitext,join
from os import listdir
import shutil
from tqdm import tqdm
import random

root_folder  = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted2_doiten_split_manual"
save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted2_doiten_split_manual_splited"

# save_img_folder = join(save_folder)
# save_txt_folder = join(save_folder)

for type_name in listdir(root_folder):
    sub_folder = join(root_folder, type_name)
    
    img_folder = join(root_folder,type_name,"images")
    txt_folder = join(root_folder,type_name,"labels")
    img_names = listdir(img_folder)
    
    
    
    random.seed(1000)
    for i in range(5):
        random.shuffle(img_names)
        
    for i,img_name in tqdm(enumerate(img_names)):
        
        if i < len(img_names) * 0.8:
            phase = 'train'
        else:
            phase = 'val'
            
        save_img_folder = join(save_folder,phase,"images")
        save_txt_folder = join(save_folder,phase,"labels")
        
        img_path = join(img_folder, img_name)
        id_img = splitext(img_name)[0]
        txt_path = join(txt_folder,f"{id_img}.txt")
        
        # img_sub_save_folder = join(save_img_folder,phase)
        img_sub_save_folder = save_img_folder
        os.makedirs(img_sub_save_folder,exist_ok=True)
        
        # txt_sub_save_folder = join(save_txt_folder,phase)
        txt_sub_save_folder = save_txt_folder
        os.makedirs(txt_sub_save_folder,exist_ok=True)
        
        save_img_path = join(img_sub_save_folder, img_name)
        save_txt_path = join(txt_sub_save_folder, f"{id_img}.txt")
        
        shutil.copy(img_path,save_img_path)
        shutil.copy(txt_path,save_txt_path)
        
    
