import os 
import sys 
from collections import defaultdict
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import random
from models import get_model
import torch
import os
import torch
from PIL import Image
from dataset.PLATEOCR_dataset import PLATEOCR
from dataset.vocab import Vocab
from tqdm import tqdm
import shutil
import pandas as pd
from inference.utils_2 import extract_real_name
from inference.test_gradio import top1_plate,convert_float_to_int,crop_img
import cv2
from ultralytics import  YOLO
from tqdm import tqdm
from utils.resize_image import ProcessImageV2
size_plate = 1200
resizer = ProcessImageV2(size_plate)
class Infer(object):
	def __init__(self, config):
		
		self.config = config
		self.model = get_model(config)
		self.model.eval()
		print(f"Device: {self.config['device']}")
		
		self.load_pretrained_model()
		self.default_transform = PLATEOCR.get_default_transform()
		self.default_visualize = PLATEOCR.get_default_visualize()
  
		self.vocab = Vocab(self.config['vocab'])
		self.device = self.config['device']
		self.vehicle_predict = YOLO('/work/21013187/phuoc/Image_Captionning_Transformer/weights/vehicle.pt').to(self.device)
		self.count_save_image = defaultdict(int)
	def load_pretrained_model(self):
		pretrained_path = self.config['model']['pretrained']
		state_dict = torch.load(pretrained_path, map_location=self.config['device'])
		model_state_dict = state_dict['model']
		self.model.load_state_dict(state_dict=model_state_dict)
		print(f"Load pretrained model from {pretrained_path}")
		self.best_score = state_dict['best_loss']
		print(f"Best score: {self.best_score}")
	def load_image_from_path(self,path,is_plate = True):
		
		if is_plate:
			pil_img = Image.open(path)
			
		else:
			img = cv2.imread(path)
			plate_location,img  = top1_plate(img)
			if plate_location == None:
				self.plate = None
				self.tensor = None
				return 
			plate_location = convert_float_to_int(plate_location)
			plate = crop_img(img,plate_location)
			plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(plate)
   
		# self.plate = self.default_visualize(pil_img)
		self.plate = pil_img
		tensor = self.default_transform(pil_img)
		self.tensor = tensor.unsqueeze(0).to(self.device)
	def load_images_array(self,images_list,save):
		all_plate_location = []
		all_locataion = []
  
		for img in images_list:
			plate_location,img  = top1_plate(img)
			if plate_location == None:
				all_plate_location.append(None)
				continue
			
			plate_location = convert_float_to_int(plate_location)
			plate = crop_img(img,plate_location)
			# plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
			if isinstance(plate,Image.Image):
				pil_img = plate
			else:
				pil_img = Image.fromarray(plate)
			all_plate_location.append(pil_img)
			all_locataion.append(plate_location)
   
		all_text = []
		for plate in all_plate_location:
			if plate is None:
				all_text.append(None)
				continue
			tensor = self.default_transform(plate)
			self.tensor = tensor.unsqueeze(0).to(self.device)
			text = self.inference()
			all_text.append(text)
			if save:
				
				if self.count_save_image[text] > 10:
					continue
				self.count_save_image[text] += 1
    
				save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_original_ver33"
				os.makedirs(save_folder,exist_ok=True)
				img_name = f"{text}_{random.randint(0,999999)}_{random.randint(0,9999)}.jpg"
				img_path = os.path.join(save_folder,img_name)
				
				plate.save(img_path)
			
		self.all_text = all_text
		self.all_location = all_locataion
		
		
   
	def inference(self):
		if self.tensor == None:
			return None
		outputs,prob = self.model.translate(self.tensor)

		index = outputs
		
		old_index = index[0].tolist()
		old_prob = prob[0].tolist()
		index = old_index
		# index = []
		# for i,prob in zip(old_index,old_prob):
		# 	if prob > 0.7:	
		# 		index.append(i)

		text = self.vocab.decode(index)
		return text
  
	def crop_and_save_plate(self,folder_path,save_folder,is_plate = False):
		
		if os.path.exists(save_folder):
			shutil.rmtree(save_folder)
		
		image_names = os.listdir(folder_path)
  
		for image_name in tqdm(image_names):
	  
			id = os.path.splitext(image_name)[0]
			image_path = os.path.join(folder_path,image_name)
			
			self.load_image_from_path(image_path,is_plate=is_plate)
			text = self.inference()
			if text is None:
				continue
			os.makedirs(save_folder,exist_ok=True)
			image_name = f"{text}_{id}.jpg"
			save_path = os.path.join(save_folder,image_name)
			
			self.plate.save(save_path)
	def infer_video(self, video_path, save_output_path,save = True):
		cap = cv2.VideoCapture(video_path)
	
		if not cap.isOpened():
			print("Error: Cannot open video file.")
			return
		
		# # Get video properties
		frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		
		# Define codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # You can change codec as needed
		out = cv2.VideoWriter(save_output_path, fourcc, fps // 4, (frame_width, frame_height))
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		pbar = tqdm(range(total_frames),total = total_frames)
		count = 0
		while True:
			pbar.update(1)
			ret, frame = cap.read()
			# frame  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			# if count > 3000:
			# 	break
			if not ret:
				break
			count += 1
			if count % 4 != 0:  # Process every third frame
				continue
			
			# frame = cv2.resize(frame, (1020, 500))
			
			# Predict objects in the frame using YOLO model
			with torch.no_grad():
				results = self.vehicle_predict.predict(frame, conf=0.35, verbose=False)
			
			detections = results[0].boxes
   
			all_vehicle_images = []
			all_new_w,all_new_h,all_pad,all_original_w,all_original_h = [],[],[],[],[]
			all_original_location = []
			for xyxy in detections.xyxy:
				x1, y1, x2, y2 = map(int, xyxy.cpu().tolist())
				all_vehicle_images.append(resizer(Image.fromarray(frame[y1:y2,x1:x2,:])))
    
				all_new_w.append(resizer.new_w)
				all_new_h.append(resizer.new_h)
				all_pad.append(resizer.pad)
				all_original_w.append(x2-x1)
				all_original_h.append(y2-y1)
				all_original_location.append([x1,y1])
				
	
			
			self.load_images_array(all_vehicle_images,save = save)

			
			for i, xyxy in enumerate(detections.xyxy):
				
				text = self.all_text[i]  # Get the corresponding text label
				if text is None:
					continue
				x1, y1, x2, y2 = map(int, xyxy.cpu().tolist())

				# Draw bounding box
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  

				# Put text above the bounding box
				cv2.putText(
					frame, text, (x1, y1 - 10),  # Position text slightly above the box
					cv2.FONT_HERSHEY_SIMPLEX,  # Font type
					0.5,  # Font scale
					(0, 0, 255),  # Font color (Green)
					1,  # Thickness
					cv2.LINE_AA  # Anti-aliasing
				)
	
			all_x1y1x2y2 = self.all_location
			all_new_location = []
			for x1y1_location_plate,x1y1x2y2,new_w,new_h,pad,original_w,original_h in zip(all_original_location,all_x1y1x2y2,all_new_w,all_new_h,all_pad,all_original_w,all_original_h):
				new_x1, new_y1, new_x2, new_y2 = resizer.caculate_reletive_location(x1y1x2y2,new_w,new_h,pad,original_w,original_h)
				x1,y1 = x1y1_location_plate
				all_new_location.append((new_x1+x1, new_y1+y1, new_x2+x1, new_y2+y1))
				

			for i, xyxy in enumerate(all_new_location):
				x1, y1, x2, y2 = map(int, xyxy)
    
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
			self.all_location = None
	
			# Draw bounding box
			out.write(frame)  # Write processed frame to video

		cap.release()
		out.release()
		print(f"Output video saved at {save_output_path}")


if __name__ == '__main__':
	config_file = r"/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/plate_ocr_infer.yaml"
 
	from utils import load_config
 
	config = load_config(config_file)
	infer = Infer(config=config)
	# infer.infer_video("/work/21013187/phuoc/Image_Captionning_Transformer/data/IMG_1890.MOV","output2.mp4")
	infer.infer_video("/work/21013187/phuoc/Image_Captionning_Transformer/IMG_2644.MOV","output3.mp4")
