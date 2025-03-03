import os
import shutil
from os.path import join
from os import listdir
from collections import defaultdict
import random
random.seed(42)

root_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted_split_all/two"
image_folder = join(root_folder,'images')

save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted2_doiten_split_manual/two"

count = defaultdict(int)
# dau 36, 17 ... max 
max_size =  25
total = 0 

def get_plate(image_name):
    return image_name.split('_')[0]
    
image_names = listdir(image_folder)
for i in range(10):
    random.shuffle(image_names)

for img_name in image_names:
    
    id = os.path.splitext(img_name)[0]
    
    img_path = join(image_folder, img_name)
   
    plate_text = get_plate(img_name)
    dau_plate = plate_text[:2]
    
    if count[dau_plate] > max_size:
        continue
    
    count[dau_plate]+=1
    total +=1
    
    
    img_save_folder = join(save_folder,'images')
    os.makedirs(img_save_folder, exist_ok=True)
    
    
    
    img_save_path = join(img_save_folder, f"{id}.jpg")
    shutil.copy(img_path, img_save_path)
  
print(f"Total : {total}")