import os
import shutil

image_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/synthetic_data2/images"
save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/synthetic_data2/labels"
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder,exist_ok=True)

for image_name in os.listdir(image_folder):
    id = os.path.splitext(image_name)[0]
   
    plate_text = id.split('_')[0]
    label_path = os.path.join(save_folder, f"{id}.txt")
    
    with open(label_path, 'w') as f:
        f.write(plate_text)