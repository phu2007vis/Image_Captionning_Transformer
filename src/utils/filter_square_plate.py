
import os 
import shutil
from os.path import join
import pandas as pd 
import cv2

img_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number/images"
output_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number_square/images"
os.makedirs(output_folder,exist_ok=True)
count = 0 
for img_name in os.listdir(img_folder):
	
	img_path = os.path.join(img_folder,img_name)
	img = cv2.imread(img_path)
	h,w,_ = img.shape
	if  h*1.7 < w:
		continue
	save_path = os.path.join(output_folder,img_name)
	shutil.copy(img_path, save_path)
	count +=1 
print("count: %d" % count)

