

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ocr_new_ver import predict_cls,predict_cls_with_box
from utils_2 import read_yolo_annotation_x1_y1_x2_y2,draw_bounding_boxes,extract_real_name

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

image_folder = r'/work/21013187/phuoc/msi_license_plate/license_plate_0-7/train/images'
txt_folder = r'/work/21013187/phuoc/msi_license_plate/license_plate_0-7/train/labels'
# image_folder = r"/work/21013187/phuoc/License_Plate/data/OCR/val/vn_images"
# txt_folder = r"/work/21013187/phuoc/License_Plate/data/OCR/val/vn_labels"
label_path = r"/work/21013187/phuoc/msi_license_plate/phuong_lp_map.csv"
# label_path = r"/work/21013187/phuoc/License_Plate/data/OCR/valid_label.csv"

label_map = pd.read_csv(label_path)
label_map['names'] = label_map['names'].apply(extract_real_name)
root_save = r"/work/21013187/phuoc/Image_Captionning_Transformer/results/results5"
import shutil 
if os.path.exists(root_save):
    shutil.rmtree(root_save)

label_names = names
all_label_number,all_predict_number = [],[]

for image_name in os.listdir(image_folder):
    img_id = os.path.splitext(image_name)[0]
    txt_name = f"{img_id}.txt"
    txt_path = os.path.join(txt_folder, txt_name)

    image_path = os.path.join(image_folder, image_name)
    
    real_label = label_map[label_map['names'] == extract_real_name(image_name)]['plate'].values[0]
    
    real_label = "".join([c for c in real_label if c in label_names])
    img = cv2.imread(image_path)
    # text, plate, box_number = predict_cls(img, is_plate=True,use_heristic = True,real_label=real_label)


    img_height, img_width, _ = img.shape

    # Convert YOLO annotations to bounding boxes
    bboxes_label = read_yolo_annotation_x1_y1_x2_y2(txt_path, img_width, img_height,names = names)
    # img = draw_bounding_boxes(img,bboxes_label)
    # cv2.imwrite("test.jpg",img)
    # exit()
    text, plate, box_number,[label_number,predict_number] = predict_cls_with_box(img,
                                                                                 box_label=bboxes_label,
                                                                                 is_plate=True,
                                                                                 use_heristic=True, 
                                                                                 real_label=real_label,
                                                                                 original_image_name=extract_real_name(image_name),
                                                                                 version = "2")
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

# Plot classification counts per class
correct_classifications = {cls: total_classification[cls] - count_classification[cls] for cls in names}
incorrect_classifications = count_classification

x = np.arange(len(names))
bar_width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width / 2, [correct_classifications[cls] for cls in names], bar_width, label="Correct")
plt.bar(x + bar_width / 2, [incorrect_classifications[cls] for cls in names], bar_width, label="Incorrect")
plt.xticks(x, names, rotation=45)
plt.xlabel("Classes")
plt.ylabel("Count")
plt.title("Classification Results per Class")
plt.legend()
plt.tight_layout()
plt.savefig("classification_counts.png")
plt.close()

# Plot percentage distribution of predictions
total_predictions = sum(total_classification.values())
percentages = [total_classification[cls] / total_predictions * 100 for cls in names]

plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=names, autopct="%1.1f%%", startangle=140, textprops={'fontsize': 8})
plt.title("Percentage Distribution of Predictions per Class")
plt.tight_layout()
plt.savefig("classification_percentages.png")
plt.close()

# import pickle
# with open("result_number.pkl", "wb") as data:
#     pickle.dump({"all_label_number": all_label_number, "all_predict_number": all_predict_number}, data)
