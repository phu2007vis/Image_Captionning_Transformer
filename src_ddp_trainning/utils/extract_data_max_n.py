import os
import shutil
from os.path import join
from os import listdir
import cv2
import random

root_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/split_ted2_doi_ten/train"
image_folder = join(root_folder,'images')
label_file = join(root_folder,'labels')
save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted_split_all"

count = {
    'one': 0,
    'two': 0,
}
max_size = 100000

def check_type(img_path):
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    if w /h >2:
        return 'one'
    elif w /h < 1.5:
        return 'two'
    else:
        return None
    
image_names = listdir(image_folder)
random.seed(42)
for i in range(10):
    random.shuffle(image_names)

for img_name in image_names:
    
    id = os.path.splitext(img_name)[0]
    
    img_path = join(image_folder, img_name)
    txt_path = join(label_file, f"{id}.txt")
    
    type_plate = check_type(img_path)
    
    if type_plate is None or count[type_plate] > max_size:
        continue
    count[type_plate]+=1
    
    sub_save_folder = join(save_folder, type_plate)
    
    img_save_folder = join(sub_save_folder,'images')
    os.makedirs(img_save_folder, exist_ok=True)
    
    txt_savefolder = join(sub_save_folder,'labels')
    os.makedirs(txt_savefolder, exist_ok=True)
    
    img_save_path = join(img_save_folder, f"{id}.jpg")
    txt_save_path = join(txt_savefolder, f"{id}.txt")
    
    shutil.copy(img_path, img_save_path)
    shutil.copy(txt_path,txt_save_path)
    