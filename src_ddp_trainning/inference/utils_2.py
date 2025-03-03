import math
import cv2
import numpy as np

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result
def compute_skew(src_img, center_thres):
	if len(src_img.shape) == 3:
		h, w, _ = src_img.shape
	elif len(src_img.shape) == 2:
		h, w = src_img.shape
	else:
		print('upsupported image type')
	img = cv2.medianBlur(src_img, 3)
	edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
	lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
	if lines is None:
		return 1

	min_line = 100
	min_line_pos = 0
	for i in range (len(lines)):
		for x1, y1, x2, y2 in lines[i]:
			center_point = [((x1+x2)/2), ((y1+y2)/2)]
			if center_thres == 1:
				if center_point[1] < 7:
					continue
			if center_point[1] < min_line:
				min_line = center_point[1]
				min_line_pos = i

	angle = 0.0
	nlines = lines.size
	cnt = 0
	for x1, y1, x2, y2 in lines[min_line_pos]:
		ang = np.arctan2(y2 - y1, x2 - x1)
		if math.fabs(ang) <= 30: # excluding extreme rotations
			angle += ang
			cnt += 1
	if cnt == 0:
		return 0.0
	return (angle / cnt)*180/math.pi

def changeContrast(img):
	lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l_channel, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l_channel)
	limg = cv2.merge((cl,a,b))
	enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return enhanced_img

def deskew(src_img, change_cons, center_thres):
	if change_cons == 1:
		return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
	else:
		return rotate_image(src_img, compute_skew(src_img, center_thres))
	
# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
	b = y1 - (y2 - y1) * x1 / (x2 - x1)
	a = (y1 - b) / x1
	return a, b
def predict_point(x,point1,point2):
	x1,y1= point1
	x2,y2 = point2
	a,b = linear_equation(x1,y1,x2,y2)
	return a*x+b

def check_point_linear(x, y, x1, y1, x2, y2,threshold = 20):
	a, b = linear_equation(x1, y1, x2, y2)
	y_pred = a*x+b
	return(math.isclose(y_pred, y, abs_tol = threshold))

def read_plate(yolo_license_plate, im,testing = False):
	LP_type = "1"
	
	results = yolo_license_plate(im,conf = 0.4)[0]
	
	bb_list =  results.boxes.xyxy.tolist()
	if len(bb_list) == 0:
		return "unknown"
	# if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
	#     return "unknown"
	
	center_list = []
	y_mean = 0
	y_sum = 0
	names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
	for i,bb in enumerate(bb_list):
		x_c = (bb[0]+bb[2])/2
		y_c = (bb[1]+bb[3])/2
		y_sum += y_c
		
		# class_name = yolo_license_plate.names[int(results.boxes.cls[i].item())]
		class_name =names[int(results.boxes.cls[i].item())]
		center_list.append([x_c,y_c,class_name])

	# find 2 point to draw line
	l_point = center_list[0]
	r_point = center_list[0]
	for cp in center_list:
		if cp[0] < l_point[0]:
			l_point = cp
		if cp[0] > r_point[0]:
			r_point = cp

	for ct in center_list:
		if l_point[0] != r_point[0]:
			if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
				LP_type = "2"

	y_mean = int(int(y_sum) / len(bb_list))
   

	# 1 line plates and 2 line plates
	line_1 = []
	line_2 = []
	license_plate = ""

	if LP_type == "2":
		for c in center_list:
			if int(c[1]) > y_mean:
				line_2.append(c)
			else:
				line_1.append(c)
		for l1 in sorted(line_1, key = lambda x: x[0]):
			license_plate += str(l1[2])
		license_plate += "-"
		for l2 in sorted(line_2, key = lambda x: x[0]):
			license_plate += str(l2[2])
	else:
		for l in sorted(center_list, key = lambda x: x[0]):
			license_plate += str(l[2])
	if testing:
		return y_mean,l_point,r_point
	return license_plate


def compute_iou(box1, box2):

	# Unpack the coordinates
	x1_1, y1_1, x2_1, y2_1 = box1
	x1_2, y1_2, x2_2, y2_2 = box2

	# Calculate the (x, y) coordinates of the intersection rectangle
	inter_x1 = max(x1_1, x1_2)
	inter_y1 = max(y1_1, y1_2)
	inter_x2 = min(x2_1, x2_2)
	inter_y2 = min(y2_1, y2_2)

	# Compute the area of intersection rectangle
	inter_width = max(0, inter_x2 - inter_x1)
	inter_height = max(0, inter_y2 - inter_y1)
	inter_area = inter_width * inter_height

	# Compute the area of both bounding boxes
	area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
	area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

	# Compute the union area
	union_area = area_box1 + area_box2 - inter_area

	# Compute IoU
	iou = inter_area / union_area if union_area > 0 else 0.0

	return iou
def padding(box,img, p = 0.3):
	x1, y1, x2, y2 = box
	padding_x = (x2 - x1) * p
	padding_y = (y2 - y1) * p
	x1 = max(x1-padding_x ,2)
	y1 = max(y1-padding_y ,2)
	x2 = min(x2+padding_x,img.shape[1] -2 )
	y2 = min(y2+padding_y, img.shape[0] -2 )
	return [x1, y1, x2, y2]
def get_number(bb_list):
	center_list = []
	y_mean = 0
	y_sum = 0
	names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
	names = sorted(names)
	for i,bb in enumerate(bb_list):
	   
		x_c = (bb[0]+bb[2])/2
		y_c = (bb[1]+bb[3])/2
		y_sum += y_c
		
	   
		class_name =names[bb[-1]]
		center_list.append([x_c,y_c,class_name])
	LP_type = 1
	# find 2 point to draw line
	l_point = center_list[0]
	r_point = center_list[0]
	for cp in center_list:
		if cp[0] < l_point[0]:
			l_point = cp
		if cp[0] > r_point[0]:
			r_point = cp

	for ct in center_list:
		if l_point[0] != r_point[0]:
			if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
				LP_type = "2"

	y_mean = int(int(y_sum) / len(bb_list))
   

	# 1 line plates and 2 line plates
	line_1 = []
	line_2 = []
	license_plate = ""

	if LP_type == "2":
		for c in center_list:
			if int(c[1]) > y_mean:
				line_2.append(c)
			else:
				line_1.append(c)
		for l1 in sorted(line_1, key = lambda x: x[0]):
			license_plate += str(l1[2])
		license_plate += "-"
		for l2 in sorted(line_2, key = lambda x: x[0]):
			license_plate += str(l2[2])
	else:
		for l in sorted(center_list, key = lambda x: x[0]):
			license_plate += str(l[2])
   
	return license_plate
import os

def draw_box(image,l,text,real_label,check,i):
	return image
	if check: 
		text_label = real_label[i] 
	#  save_folder = f"C:\\Users\\Admin\\Downloads\\license_plate2\\number_test"
	
	
	x1,y1,x2,y2,_ = l[3]
	x1 = int(x1)
	y1 = int(y1)
	x2 = int(x2)
	y2 = int(y2)
	# if check:
	#     all_save = os.path.join(save_folder,"ALL",text_label)
	#     if text == text_label:
	#         save_folder = os.path.join(save_folder,'True',text_label)
	#     else:
	#         save_folder = os.path.join(save_folder,'Wrong',text_label)
	#     os.makedirs(all_save,exist_ok=True)
	#     os.makedirs(save_folder,exist_ok=True)

	#     index = str(len(os.listdir(save_folder))+1)
	#     save_path = os.path.join(save_folder,f"{index}.jpg")
	#     cv2.imwrite(save_path,image[y1:y2,x1:x2])

	#     index_all = str(len(os.listdir(all_save))+1)
	#     save_all_path = os.path.join(all_save,f"{index_all}.jpg")
	#     cv2.imwrite(save_all_path,image[y1:y2,x1:x2])
		
	# cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2 , (0, 0, 255), 2)

	return cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,255,0),thickness=1)

def get_number_her(bb_list,image,real_label = None):
	center_list = []
	y_mean = 0
	y_sum = 0
	names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
	names = sorted(names)
	for i,bb in enumerate(bb_list):
	   
		x_c = (bb[0]+bb[2])/2
		y_c = (bb[1]+bb[3])/2
		y_sum += y_c
		
		class_name =bb[-1]
		
		center_list.append([x_c,y_c,class_name,bb])
	LP_type = 1
	# find 2 point to draw line
	l_point = center_list[0]
	r_point = center_list[0]
	for cp in center_list:
		if cp[0] < l_point[0]:
			l_point = cp
		if cp[0] > r_point[0]:
			r_point = cp

	for ct in center_list:
		if l_point[0] != r_point[0]:
			if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
				LP_type = "2"

	y_mean = int(int(y_sum) / len(bb_list))
   

	# 1 line plates and 2 line plates
	line_1 = []
	line_2 = []
	license_plate = ""
	
	if LP_type == "2":
		
		for c in center_list:
			if int(c[1]) > y_mean:
				line_2.append(c)
			else:
				line_1.append(c)
		if real_label is None:
			check = False
		else:
			check = True if (len(line_1)+ len(line_2 ) == len(real_label)) else False
 
		i = 0 
		oto = True if  len(line_1) == 3 else False
		for l1 in sorted(line_1, key = lambda x: x[0]):
			
			
			# c_real = real_label[i]
			class_index = l1[2]
			class_names_index  = [names[index] for index in class_index]
			if i == 2:
				class_names = [name for name in class_names_index if not name.isdigit()]

			elif i == 3:
				if not oto:
					class_names = [name for name in class_names_index]
				else:
					class_names = [name for name in class_names_index if name.isdigit()]
			else:
				
				
				class_names = [name for name in class_names_index if name.isdigit()]
			  

			try:
				class_name = class_names[0]
			except:
				class_name = class_names_index[0]
			i+=1
			license_plate += str(class_name)
			image  = draw_box(image,l1,class_name,check = check,i = i,real_label=real_label)
		license_plate += "-"
		for l2 in sorted(line_2, key = lambda x: x[0]):
		   
			class_index = l2[2]
			class_names_index  = [names[index] for index in class_index]
			
			class_names = [name for name in class_names_index if name.isdigit()]
	
			try:
				class_name = class_names[0]
			except:
				class_name = class_names_index[0]
		 
			license_plate += str(class_name)
			image  = draw_box(image,l2,class_name,check = check,i = i,real_label = real_label)
	else:
		if real_label is None:
			check = False
		else:
			check = True if len(real_label) == len(center_list) else False
		for i,l in enumerate(sorted(center_list, key = lambda x: x[0])):
			
			class_index = l[2]
			class_names_index  = [names[index] for index in class_index]
			if i!= 2:
				class_names = [name for name in class_names_index if name.isdigit()]
			else:
				class_names = [name for name in class_names_index if not name.isdigit()]
			try:
				class_name = class_names[0]
			except:
				class_name = class_names_index[0]
			license_plate += str(class_name)
			image  = draw_box(image,l,class_name,check = check,i = i,real_label=real_label)
	
	return license_plate,image

import os

def read_yolo_annotation_x1_y1_x2_y2(txt_path, image_width, image_height,names):
	"""
	Reads a YOLO annotation file and converts it to bounding box coordinates.

	Parameters:
		txt_path (str): Path to the YOLO annotation file.
		image_width (int): Width of the image.
		image_height (int): Height of the image.

	Returns:
		List[Tuple[int, int, int, int]]: List of bounding boxes in (x1, y1, x2, y2) format.
	"""
	bounding_boxes = []
	
	if not os.path.exists(txt_path):
		print(f"Annotation file not found: {txt_path}")
		return bounding_boxes

	with open(txt_path, 'r') as file:
		lines = file.readlines()
		for line in lines:
			parts = line.strip().split()
			if len(parts) != 5:
				print(f"Invalid annotation format in file: {txt_path}")
				continue
			
			id, x_center, y_center, width, height = map(float, parts)
			id = names[int(id)]
			# Convert YOLO format to bounding box format
			x1 = int((x_center - width / 2) * image_width)
			y1 = int((y_center - height / 2) * image_height)
			x2 = int((x_center + width / 2) * image_width)
			y2 = int((y_center + height / 2) * image_height)
			
			bounding_boxes.append((id,x1, y1, x2, y2))
	
	return bounding_boxes

def draw_bounding_boxes(image, bboxes, color=(0, 255, 0), thickness=2):
	"""
	Draws bounding boxes on an image.

	Parameters:
		image (numpy.ndarray): The image on which to draw.
		bboxes (List[Tuple[int, int, int, int]]): List of bounding boxes in (x1, y1, x2, y2) format.
		color (Tuple[int, int, int]): Color of the bounding box (default: green).
		thickness (int): Thickness of the bounding box lines (default: 2).
		label_text (str): Optional label text to add near the boxes.

	Returns:
		numpy.ndarray: The image with bounding boxes drawn.
	"""
	for bbox in bboxes:
		id,x1, y1, x2, y2 = bbox
		cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
		
		cv2.putText(image, str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	return image

def extract_real_name(image_name):
	id = image_name.split('.rf')[0]
	id = id.replace('.jpg','').replace('.png','').replace('.jpeg','')
	if '_0_' in id:
		id = id.split('_')
		id = "_".join(id[:-3])
	id = id.replace("_jpg","")
	return id+".jpg"

    