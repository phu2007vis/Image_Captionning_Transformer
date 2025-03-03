from test_gradio import yolov11,cls_model,number_0
import torch
from utils_2 import get_number,get_number_her
import cv2
from utils.resize_image import ProcessImageV2
resize = ProcessImageV2(size = 640,color='black')
import numpy as np
from test_gradio import runner
from PIL import Image

def predict_cls(img,return_pl = True,is_plate = False,use_heristic = False,real_label = None):
	if is_plate:
		crop_img = img
	else:
		results = yolov11(img,verbose = False,conf = 0.35)[0] 
		if len(results) == 0:
			return 'unknown',None,None
		
		boxes = results.boxes
		xyxy =  boxes.xyxy.tolist()
		conf = boxes.conf
		index  = torch.argmax(conf).item()
		plate  = xyxy[index]
		
		x = int(plate[0]) # xmin
		y = int(plate[1]) # ymin
		w = int(plate[2] - plate[0]) # xmax - xmin
		h = int(plate[3] - plate[1]) # ymax - ymin  
		
		crop_img = img[y:y+h, x:x+w]
		
	
	lp = "unknown"
	if True:
			result = number_0(crop_img,conf = 0.35)[0]
			
			all_number = []
			all_box = []
			
			for box in result.boxes.xyxy:
				padding = 20
				x1,y1,x2,y2 = list(map(int, box.tolist()))
				y1 = max(2,y1-padding)
				x1 = max(2,x1-padding)
				y2 = min(crop_img.shape[0]-padding,y2+padding)
				x2 = min(crop_img.shape[1]-padding,x2+padding)
				
				number = crop_img[y1-2:y2+2,x1-2:x2+2,:]
				all_number.append(number)
				all_box.append(box.tolist())
				
			if len(all_number) == 0 :
				return None,None,None
			cls = cls_model(all_number,verbose = False)
			if use_heristic:
				cls = [cls[i].probs.top5 for i in range(len(cls))]
			else:
				cls = [cls[i].probs.top1 for i in range(len(cls))]
			
			for i,b in enumerate(all_box):
		
				b.append(cls[i])
				all_box[i] = b
			

			if return_pl:
				if use_heristic:
					lp,crop_img = get_number_her(all_box,crop_img,real_label  = real_label)
				else:
					lp  = get_number(all_box)

	return lp,crop_img,all_box
import  os


def predict_cls_with_box(img,
						 box_label,
						 original_image_name,
						 return_pl = True,
						 is_plate = False,
						 use_heristic = True,
						 real_label = None,
						 save_folder = r"number_analytics",
						 version = '1'):
	if is_plate:
		crop_img = img
	lp = "unknown"
	if True:	
			
			all_box = []
			all_number = []
			all_label_cls = []
			for save_index,box in enumerate(box_label):
							# cls = box[0]
				all_label_cls.append(box[0])
				box = list(box[1:])
				padding_h = 7
				padding_x = 2
				x1,y1,x2,y2 = list(map(int, box))
				y1 = max(2,y1-padding_h)
				x1 = max(2,x1-padding_x)
				y2 = min(crop_img.shape[0]-padding_h,y2+padding_h)
				x2 = min(crop_img.shape[1]-padding_x,x2+padding_x)
				
				number = crop_img[y1-2:y2+2,x1-2:x2+2,:]
				all_number.append(number)
				box = [x1,y1,x2,y2]
				all_box.append(box)
				
			if len(all_number) == 0 :
				return None,None,None,None

			if version == '1':
				
				results = cls_model(all_number,verbose = False)
				if use_heristic:
					cls = [results[i].probs.top5 for i in range(len(results))]
					top1  = [cls_model.names[results[i].probs.top1] for i in range(len(results))]
					# conf = [results[i].probs.top1conf.item() for i in range(len(results))]
					for i,c in enumerate(top1):
						c_label = all_label_cls[i]
						frame = np.copy(all_number[i])
						if c != c_label:
							save_path  =  os.path.join(save_folder,"sai")
						else:
							save_path = os.path.join(save_folder,"dung")
							# cv2.putText(frame,str(int(conf[i]*100)),(8,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
						os.makedirs(save_path,exist_ok=True)
						
					
						id = original_image_name.replace(".jpg","")
					
						save_name = f"{c_label}->{c}->{id}->{save_index}.jpg"
				
						cv2.imwrite(os.path.join(save_path, save_name), frame)

						save_path = os.path.join(save_folder,"all",str(c_label))
						os.makedirs(save_path,exist_ok=True)
						save_path = os.path.join(save_path,original_image_name)
						cv2.imwrite(save_path, frame)
				
				else:
					if version == '1':
						cls = [results[i].probs.top1 for i in range(len(results))]
					top1 = cls
			else:
				top_k = 5 if use_heristic else 1
				conf,cls = runner.predict_from_list_image(all_number,top_k=top_k)
				if use_heristic:
					top1 = [c[0] for c in cls]
				else:
					top1 = cls 

			
			for i,b in enumerate(all_box):
		
				b.append(cls[i])
				all_box[i] = b
			

			if return_pl:
				if use_heristic:
					lp,crop_img = get_number_her(all_box,crop_img,real_label  = real_label)
				else:
					lp  = get_number(all_box)
	lp = lp.replace("-","")
	return lp,crop_img,all_box,[all_label_cls,top1]

def predict_cls2(img,
				use_heristic = True,
				real_label = None,
				):

	
	results = yolov11(img,verbose = False,conf = 0.35)[0] 
	if len(results) == 0:
		return 'unknown',None,None,None
	
	boxes = results.boxes
	xyxy =  boxes.xyxy.tolist()
	conf = boxes.conf
	index  = torch.argmax(conf).item()
	plate  = xyxy[index]
	
	x = int(plate[0]) # xmin
	y = int(plate[1]) # ymin
	w = int(plate[2] - plate[0]) # xmax - xmin
	h = int(plate[3] - plate[1]) # ymax - ymin  
	
	crop_img = img[y:y+h, x:x+w]
	cv2.rectangle(img,(x,y),(x+w,y+h),color = (255,0,0),thickness=1)
	
	
	lp = "unknown"
 
	if True:	
			
			all_box = []
			all_number = []
			all_label_cls = []
			result = number_0(crop_img,conf = 0.35,imgsz = 640,verbose=False,iou = 0.4)[0]
			
			all_number = []
			all_box = []
			
			for save_index,box in enumerate(result.boxes.xyxy):
				padding_h = 0
				padding_x = 0
				x1,y1,x2,y2 = list(map(int, box.tolist()))
		
				y1 = max(2,y1-padding_h)
				x1 = max(2,x1-padding_x)
				y2 = min(crop_img.shape[0],y2+padding_h)
				x2 = min(crop_img.shape[1],x2+padding_x)
				
				number = crop_img[y1-2:y2+2,x1-2:x2+2,:]
				all_number.append(number)
				box = [x1,y1,x2,y2]
				all_box.append(box)
			
				
			if len(all_number) == 0 :
				return None,None,None,None

		
			top_k = 5 if use_heristic else 1
			conf,cls = runner.predict_from_list_image(all_number,top_k=top_k)
			if use_heristic:
				top1 = [c[0] for c in cls]
			else:
				top1 = cls 

			
			for i,b in enumerate(all_box):
		
				b.append(cls[i])
				all_box[i] = b
			
			if use_heristic:
				lp,crop_img = get_number_her(all_box,crop_img,real_label  = real_label)
			else:
				lp  = get_number(all_box)
	lp = lp.replace("-","")
	return lp,crop_img,all_box,[all_label_cls,top1]
def get_text_from_plate(crop_img,use_heristic = True):
    
		all_box = []
		all_number = []
		all_label_cls = []
		all_number = []
		all_box = []
  
		result = number_0(crop_img,conf = 0.35,imgsz = 640,verbose=False)[0]
		
		for _,box in enumerate(result.boxes.xyxy):
			padding_h = 0
			padding_x = 0
			x1,y1,x2,y2 = list(map(int, box.tolist()))

			y1 = max(2,y1-padding_h)
			x1 = max(2,x1-padding_x)
			y2 = min(crop_img.shape[0],y2+padding_h)
			x2 = min(crop_img.shape[1],x2+padding_x)
			
			number = crop_img[y1-2:y2+2,x1-2:x2+2,:]
			all_number.append(number)
			box = [x1,y1,x2,y2]
			all_box.append(box)
		
			
		if len(all_number) == 0 :
			return None,None,None,None


		top_k = 5 if use_heristic else 1
		conf,cls = runner.predict_from_list_image(all_number,top_k=top_k)
		if use_heristic:
			top1 = [c[0] for c in cls]
		else:
			top1 = cls 

		
		for i,b in enumerate(all_box):

			b.append(cls[i])
			all_box[i] = b
		
		if use_heristic:
			lp,crop_img = get_number_her(all_box,crop_img)
		else:
			lp  = get_number(all_box)
   
		lp = lp.replace("-","")
		return lp,crop_img,all_box,[all_label_cls,top1]
if __name__ == "__main__":
	import cv2
	img = cv2.imread(r"C:\Users\Admin\Downloads\license_plate2\images2\5a79ebac-20210519_075831.jpg")
	lp,plate = predict_cls(img)
	print("License plate:", lp)
	cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
	cv2.imwrite("output.jpg", img)
	cv2.imshow("Output", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Program finished.")
