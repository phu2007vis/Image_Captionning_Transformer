
import time
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ocr_new_ver import predict_cls2
from utils_2 import extract_real_name
from utils.resize_image import ProcessImageV2
from PIL import Image

resize = ProcessImageV2(size = 1240,color='black')
names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', '0']
names = sorted(names)
threshold = 0.9
count_map = {
    'true': 0,
    'detect': 0,
    'wrong': 0,
    'classification': 0
}
count_classification = {cls: 0 for cls in names}
total_classification = {cls: 0 for cls in names}

image_folder = r'/work/21013187/phuoc/Image_Captionning_Transformer/data/test_dataset/images'


label_path = r"/work/21013187/phuoc/Image_Captionning_Transformer/data/test_dataset/labels_2.csv"


label_map = pd.read_csv(label_path)
label_map['names'] = label_map['names'].apply(extract_real_name)
root_save = r"/work/21013187/phuoc/Image_Captionning_Transformer/results/datatest"
import shutil 
if os.path.exists(root_save):
    shutil.rmtree(root_save)

label_names = names
all_label_number,all_predict_number = [],[]
count_image = 0 
for image_name in os.listdir(image_folder):
    img_id = os.path.splitext(image_name)[0]
    txt_name = f"{img_id}.txt"


    image_path = os.path.join(image_folder, image_name)
    try:
        real_label = label_map[label_map['names'] == extract_real_name(image_name)]['plate'].values[0]
    except:
        print(image_name)
        continue
    count_image+=1
    real_label = "".join([c for c in real_label if c in label_names])
    img = cv2.imread(image_path)
    pil_img = Image.fromarray(img)
    pil_img = resize(pil_img)
    img2 = np.asarray(pil_img,dtype=np.uint8)
    img = np.copy(img2)

    img_height, img_width, _ = img.shape
    t= time.time()
    text, plate, box_number,[label_number,predict_number] = predict_cls2(img,
                                                                        use_heristic=True, 
                                                                      real_label=real_label)
    cv2.putText(img,text,(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,color =(0,0,255),thickness=3)
    # print(time.time()-t)
    if plate is None:
        continue
    all_label_number.extend(label_number)
    all_predict_number.extend(predict_number)

    text = text.replace("-", '').replace(".", "")
    real_label = real_label.replace("-", '').replace(".", "")

    image_name = image_name.replace(".jpg", "") + f"_{text}_{real_label}.jpg"
    if text == real_label:
        count_map['true'] += 1
        os.makedirs(os.path.join(root_save, "True"), exist_ok=True)
        path_save = os.path.join(root_save, "True", image_name)
        if path_save is not None:
            cv2.imwrite(path_save, img)
    else:
        os.makedirs(os.path.join(root_save, "wrong"), exist_ok=True)
        path_save = os.path.join(root_save, "wrong", image_name)
        count_map['wrong'] += 1
        if path_save is not None:
            cv2.imwrite(path_save, img)

        if len(text) != len(real_label):
            count_map['detect'] += 1
            os.makedirs(os.path.join(root_save, "detect"), exist_ok=True)
            path_save = os.path.join(root_save, "detect", image_name)
            if path_save is not None:
                cv2.imwrite(path_save, img)
        else:
            count_map['classification'] += 1
            os.makedirs(os.path.join(root_save, "classification"), exist_ok=True)
            path_save = os.path.join(root_save, "classification", image_name)

            if path_save is not None:
                cv2.imwrite(path_save, img)
            for i in range(len(text)):
                cls_label = real_label[i]
                cls_predict = text[i]
                total_classification[cls_label] += 1
                if cls_label != cls_predict:
                    count_classification[cls_label] += 1

# Print classification counts
for key, value in count_map.items():
    print(f"{key}: {value}")

print("Total: {}".format(count_image))
