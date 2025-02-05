import cv2
import os
import shutil


image_folder = r"/work/21013187/phuoc/License_Plate/data/OCR/val/vn_images"
txt_folder = r"/work/21013187/phuoc/License_Plate/data/OCR/val/vn_labels"
save_folder = r"/work/21013187/phuoc/License_Plate/data/vn_number_images"
names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

def main():


	for img_name in os.listdir(image_folder):
		img_path = os.path.join(image_folder,img_name)
		img = cv2.imread(img_path)
		id = img_name.replace(".jpg","")
		txt_name = id+".txt"
		txt_path = os.path.join(txt_folder,txt_name)
		h,w, _ = img.shape
		with open(txt_path,'r') as f:
			data = [list(map(float,d.strip().split(" "))) for d in f.readlines()]
		for i,box in enumerate(data):
			cls_index,xc,yc,w_crop,h_crop = box
			cls = names[int(cls_index)]
			cls = str(cls)
			xc = xc*w
			yc = yc*h
			w_crop = w_crop*w
			h_crop = h_crop*h
			x1 = max(int(xc - w_crop / 2)-5,2)
			y1 = max(int(yc - h_crop / 2) -5,2)
			x2 = min(int(xc + w_crop / 2)+5,w-2)
			y2 = min(int(yc + h_crop / 2)+5,h-2)
			image_crop = img[y1:y2,x1:x2,:]
			save_name = f"{id}_{i}.jpg"
			sub_save_folder = os.path.join(save_folder,cls)
			save_path = os.path.join(sub_save_folder,save_name)
			os.makedirs(sub_save_folder,exist_ok=True)
			cv2.imwrite(save_path,image_crop)
		

main()