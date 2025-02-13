import os
from os.path import join
import shutil
from collections import defaultdict

def get_plate(image_name):
    return image_name.split('_')[0][:2]



img_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted_split_all/two/images"

count = defaultdict(int)
for img_name in os.listdir(img_folder):
    plate_name = get_plate(img_name)
    count[plate_name] += 1
max = 20
total = 0
for key,value in count.items():
    print(f"{key} : {value}")
    if value > max:
        total += max
    else:
        total += value
print(f"Total : {total}")
    
    