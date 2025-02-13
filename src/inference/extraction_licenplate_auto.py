import os
from tqdm import tqdm
import cv2
from test_gradio import top1_plate,convert_float_to_int,crop_img
from ocr_new_ver import get_text_from_plate
import pandas as pd 
from utils.resize_image import ProcessImageV2
from PIL import Image
import numpy as np

resize = ProcessImageV2(size = 1240,color='black')

img_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_f"
save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_f_only"
output_label_file = os.path.join(save_folder,"labels.csv")
img_save_folder = os.path.join(save_folder,"images")
os.makedirs(img_save_folder, exist_ok=True)

ouput_dict = {'names':[],
              'plate':[]
              }

img_names = os.listdir(img_folder)

for img_name in tqdm(img_names,total=len(img_names)):
    id = os.path.splitext(img_name)[0]
    
    img_path = os.path.join(img_folder,img_name)
    img = cv2.imread(img_path)
    pil_img = Image.fromarray(img)
    pil_img = resize(pil_img)
    img2 = np.asarray(pil_img,dtype=np.uint8)
    img = np.copy(img2)
    
    plate_location,img  = top1_plate(img)
    
    if plate_location is None:
        continue
    
    plate_location = convert_float_to_int(plate_location)
    plate = crop_img(img,plate_location)
    predict_text,_,_,_ = get_text_from_plate(plate,use_heristic=True)
    ouput_dict['names'].append(id)
    ouput_dict['plate'].append(predict_text)
    new_name_plate = f"{predict_text}_{id}.jpg"
    save_path = os.path.join(img_save_folder,new_name_plate)
    
    cv2.imwrite(save_path,plate)

df = pd.DataFrame(ouput_dict)
df.to_csv(output_label_file,index=False)