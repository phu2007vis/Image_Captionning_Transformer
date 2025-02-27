from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import os
import sys
torch.cuda.set_device(0) 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Load the trained YOLO model
# yolov8 = YOLO(r"C:\Users\Admin\Downloads\weights\best.pt")  
# yolov11 = YOLO(r"weights\yolov11m_detection.pt")
yolov11 = YOLO(r"/work/21013187/phuoc/best_25k.pt").cuda()


from inference.utils_2 import padding
# from inference.inception_inference import runner
# number_0 = YOLO(r"weights\best_0.pt")
# number_0 = YOLO(r"/work/21013187/phuoc/best_uc3_val.pt")
# cls_model = YOLO(r"C:\Users\Admin\Downloads\license_plate2\weights\best_cls_2.pt")
# cls_model = YOLO(r"/work/21013187/phuoc/best_300_v2.pt")
# cls_model = YOLO(r"C:\Users\Admin\Downloads\best_300_v2.pt")
# yolov11 = YOLO(r"C:\Users\Admin\Downloads\best (2).pt")
yolov8 = None
# Path to the best weights file
register_model = {
	'yolov8': yolov8,
	'yolov11': yolov11,
}
def predict_yolo(image,model_name):
	"""
	Perform YOLO prediction on the uploaded image.
	"""
	model = register_model[model_name]
	results = model(image)  # Run prediction
	result_image = results[0].plot()    
	return Image.fromarray(result_image)

# Define the Gradio interface
model_name = ['yolov8','yolov11']


def convert_float_to_int(box):
	"""
	Convert float boxes to integer boxes.
	"""
	return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
def crop_img(image,box):
	"""
	Crop the image using the given box.
	"""
	x1, y1, x2, y2 = box
	if isinstance(image,Image.Image):
		return image.crop((x1, y1, x2, y2))

	return image[y1:y2, x1:x2]
def top1_plate(img):
	result = yolov11(img,verbose = False,conf = 0.35)[0]
	if len(result.boxes) == 0:
		return None,img
	conf = result.boxes.conf
	value,index = conf.max(0)
	index = index.item()
	plate = result.boxes.xyxy[index].tolist()
	return plate,img

def license_plate_recognition(image):

	results = license_plate_recognition_model(image,conf = 0.3)[0]  # Run prediction
	c_list = []
	for  i,number in enumerate(results.boxes.xyxy.tolist()):
		x1,y1,x2,y2 = list(map(int,number))
		cv2.rectangle(image, (x1,y1),(x2,y2), color = (0,0,225), thickness = 2)
		names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
		class_name =names[int(results.boxes.cls[i].item())]
		cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
		x_c = (number[0]+number[2])/2
		y_c = (number[1]+number[3])/2
		c_list.append([int(x_c), int(y_c)])
	y_mean,l_point,r_point = read_plate(license_plate_recognition_model,image,True)
	l_point = list(map(int,l_point[:2]))
	r_point = list(map(int,r_point[:2]))
	y_mean  = int(y_mean)
	h,w,_ = image.shape

	for x,y in c_list:
		cv2.circle(image, (x,y), 5, (255, 255, 0), -1)

	cv2.line(image,(2,y_mean),(w-2,y_mean),(255,0,0),2,cv2.LINE_AA)
	cv2.putText(image, "y mean", (int(w//2), int(y_mean)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

	cv2.circle(image, l_point, 10, (0, 255, 0), -1)  # Green circle
	cv2.putText(image, "l_point", (int(l_point[0] + 10), int(l_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# Draw and label r_point
	cv2.circle(image, r_point, 10, (0, 0, 255), -1)  # Red circle
	cv2.putText(image, "r_point", (int(r_point[0] + 10), int(r_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# #draw linear line
	x1 = 2
	y1 = int(predict_point(x1,l_point,r_point))
	x2 = w-2
	y2 = int(predict_point(x2,l_point,r_point))
	cv2.line(image,(2,y1),(w-2,y2),(255,0,0),2,cv2.LINE_AA)
	y3 = int(predict_point(w//2,l_point,r_point))
	a,b   = linear_equation(x1,y1,x2,y2)
	cv2.putText(image, f"y = {round(a,2)}x+{round(b,2)}", (int(w//2), int(y3)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,69,0), 2)

	return Image.fromarray(image)


import cv2
from utils import *
def license_plate_recognition_final(img):

	results = yolov11(img)[0] 

	results =  results.boxes.xyxy.tolist()

	for plate in results:
		flag = 0
		x = int(plate[0]) # xmin
		y = int(plate[1]) # ymin
		w = int(plate[2] - plate[0]) # xmax - xmin
		h = int(plate[3] - plate[1]) # ymax - ymin  
		crop_img = img[y:y+h, x:x+w]
		cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
	  
		lp = ""
		for cc in range(0,2):
			for ct in range(0,2):
				lp = read_plate(license_plate_recognition_model, deskew(crop_img, cc, ct))
				if lp != "unknown":
					
					cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
					flag = 1
					break
			if flag == 1:
					break
			
	return Image.fromarray(img)



def license_plate_recognition_text(img,return_plate = True):

	results = yolov11(img,verbose = False,conf = 0.35)[0] 
	if len(results) == 0:
		if not return_plate:
			'unknow'
		return 'unknown',None
	
	boxes = results.boxes
	xyxy =  boxes.xyxy.tolist()
	conf = boxes.conf
	index  = torch.argmax(conf).item()
	plate  = xyxy[index]
	plate = padding(plate,img,p = 0.08)
	flag = 0
	x = int(plate[0]) # xmin
	y = int(plate[1]) # ymin
	w = int(plate[2] - plate[0]) # xmax - xmin
	h = int(plate[3] - plate[1]) # ymax - ymin  
	
	crop_img = img[y:y+h, x:x+w]
	cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 3)
	cv2.putText(img, "predict", (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,225), 2)
	lp = "unknown"
	for cc in range(0,2):
		for ct in range(0,2):
			lp = read_plate(license_plate_recognition_model, deskew(crop_img, cc, ct))
			if lp != "unknown":
				
				# cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
				flag = 1
				break
		if flag == 1:
			break
	if lp == "unknown":
		if not return_plate:
			return 'unknown'
		return lp, None
	if not return_plate:
		return lp
	return lp,plate





# Launch the app
if __name__ == "__main__":
	interface1 = gr.Interface(
		fn=predict_yolo,
		inputs=[gr.Image(type="numpy", label="Upload an Image"),
				gr.Dropdown(choices=model_name, label="Yolo model type", value=model_name[0], interactive=True)
		],
		outputs=gr.Image(type="pil", label="Predicted Image"),
		title="YOLO License Plate Detection",
		description="Upload an image to detect license plates using the trained YOLO model."
	)
	interface2 = gr.Interface(
		fn=license_plate_recognition,
		inputs=gr.Image(type="numpy", label="Upload an Image"),
		outputs=gr.Image(type="pil", label="Predicted Image"),
		title="YOLO License Plate Detection",
		description="Upload an image to detect license plates using the trained YOLO model."
	)
	interface3 = gr.Interface(
		fn=license_plate_recognition_final,
		inputs=gr.Image(type="numpy", label="Upload an Image"),
		outputs=gr.Image(type="pil", label="Predicted Image"),
		title="YOLO License Plate Detection",
		description="Upload an image to detect license plates using the trained YOLO model."
	)
	tabbel_interface = gr.TabbedInterface([interface1, interface2,interface3])
	tabbel_interface.launch()
