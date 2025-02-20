import os 
import shutil
from os.path import join
import pandas as pd 

img_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1/images"
csv_output_file = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1/labels.csv"
csv_map = {
	'names': [],
	'plate': []
}

for img_name in os.listdir(img_folder):
    plate_text = img_name.split("_")[0]
    csv_map['names'].append(img_name)
    csv_map['plate'].append(plate_text)


df = pd.DataFrame(csv_map)
df.to_csv(csv_output_file, index=False)

