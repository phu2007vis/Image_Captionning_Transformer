import os
import shutil
import cv2
from  tqdm import tqdm
from test_gradio import number_0,runner


img_folder = r"/work/21013187/phuoc/Vietnam-license-plate-1/plate"
save_folder = r"/work/21013187/phuoc/Vietnam-license-plate-1/number"
names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', '0']
names = sorted(names)

if os.path.exists(save_folder):
	shutil.rmtree(save_folder)
	
os.makedirs(save_folder,exist_ok=True)
total = 0 
for img_name in tqdm(os.listdir(img_folder),total = len(os.listdir(img_folder))):
	img_id = img_name.split('.')[0]
	
	img_path = os.path.join(img_folder,img_name)
	img = cv2.imread(img_path)
	h,w,_ = img.shape

	crop_img = img
	result = number_0(crop_img,conf = 0.35,verbose = False)[0]
	
 
	count = 0 
	all_box = []
	all_number = []
	all_label_cls = []
	
	

	for box in result.boxes.xyxy:
		padding = 2
		x1,y1,x2,y2 = list(map(int, box.tolist()))
		y1 = max(2,y1-padding)
		x1 = max(2,x1-padding)
		y2 = min(crop_img.shape[0]-padding,y2+padding)
		x2 = min(crop_img.shape[1]-padding,x2+padding)
		
		number = crop_img[y1:y2,x1:x2,:]
		all_number.append(number)
		all_box.append(box.tolist())
		
	total += len(all_box)
	_,cls = runner.predict_from_list_image(all_number,top_k=1)
 
	for i,b in enumerate(all_box):
		b.append(cls[i])
		all_box[i] = b
			
	for number_image,cls_index  in zip(all_number,cls):
	
		cls_name = names[int(cls_index)]
		crop_img = number_image
		sub_folder = os.path.join(save_folder,str(cls_name))
		os.makedirs(sub_folder,exist_ok=True)
		img_path = os.path.join(sub_folder,f"{img_id}_{count}.jpg")
		cv2.imwrite(img_path,crop_img)
	
				
	
		
print(total)