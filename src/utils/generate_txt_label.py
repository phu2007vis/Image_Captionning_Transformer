import os
import shutil

root_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number_square"
image_folder = os.path.join(root_folder,"images")
save_folder = os.path.join(root_folder,"labels")
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder,exist_ok=True)

for image_name in os.listdir(image_folder):
    id = os.path.splitext(image_name)[0]
   
    plate_text = id.split('_')[0]
    label_path = os.path.join(save_folder, f"{id}.txt")
    
    with open(label_path, 'w') as f:
        f.write(plate_text)