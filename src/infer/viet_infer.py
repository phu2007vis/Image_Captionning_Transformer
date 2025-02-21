import os 
import sys 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

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
   
		self.plate = self.default_visualize(pil_img.copy())
		# self.plate = pil_img
		tensor = self.default_transform(pil_img)
		self.tensor = tensor.unsqueeze(0).to(self.device)
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
	def evaluate_folder(self,folder_path,save_folder):
		
		if os.path.exists(save_folder):
			shutil.rmtree(save_folder)
		count  = 0
		dung =  0 
		image_names = os.listdir(folder_path)
		for image_name in tqdm(image_names):
			label = image_name.split('_')[3]
			image_path = os.path.join(folder_path,image_name)
			self.load_image_from_path(image_path)
			text = self.inference()
			if text == label:
				sub_save_folder = os.path.join(save_folder,'dung')
				dung+=1
			else:
				sub_save_folder = os.path.join(save_folder,'sai')
	
			os.makedirs(sub_save_folder,exist_ok=True)
			image_name = f"{text}_{count}.jpg"
			save_path = os.path.join(sub_save_folder,image_name)
			# shutil.copy(image_path,save_path)
			self.plate.save(save_path)
			count+=1
		print(f"Accuracy: {dung/count}")
	def evaluate_folder_with_label_map(self,folder_path,save_folder,label_path,is_plate = False):
	 
		label_map = pd.read_csv(label_path)
		label_map.names = label_map.names.apply(extract_real_name)
		if os.path.exists(save_folder):
			shutil.rmtree(save_folder)
		count  = 0
		dung =  0 
		sai = 0
		image_names = os.listdir(folder_path)
	
		for image_name in tqdm(image_names):
			
   
			image_path = os.path.join(folder_path,image_name)
			self.load_image_from_path(image_path,is_plate=is_plate)
			text = self.inference()
			try:
				
				# label = label_map[label_map['names'] == os.path.splitext(extract_real_name(image_name))[0]]['plate'].values[0]
				label = label_map[label_map['names'] == extract_real_name(image_name)]['plate'].values[0]
			except:
			
				continue
			label = label.replace("-","")
			if text == label:
				sub_save_folder = os.path.join(save_folder,'dung')
				dung+=1
			else:
				sub_save_folder = os.path.join(save_folder,'sai')
				sai+=1 
				
	
			os.makedirs(sub_save_folder,exist_ok=True)
			image_name = f"{text}_{label}_{count}.jpg"
			save_path = os.path.join(sub_save_folder,image_name)
			# shutil.copy(image_path,save_path)
			self.plate.save(save_path)
			count+=1
		print(f"Dung {dung}, Sai: {sai}, Total: {count}")
		print(f"Accuracy: {dung/count}")
  
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
		
		


if __name__ == '__main__':
	config_file = r"/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/plate_ocr_infer.yaml"
 
	from utils import load_config
 
	config = load_config(config_file)
	infer = Infer(config=config)
 
	# infer.load_image_from_path(r"/work/21013187/phuoc/Image_Captionning_Transformer/data/image.png",is_plate=False)
	# print(infer.inference())

 
	# infer.evaluate_folder_with_label_map(folder_path="/work/21013187/phuoc/Image_Captionning_Transformer/data/test_dataset/images",
	# 								  save_folder=r"/work/21013187/phuoc/Image_Captionning_Transformer/results/infer_test",
	# 								  label_path="/work/21013187/phuoc/Image_Captionning_Transformer/data/test_dataset/labels_2.csv",
    #        										is_plate= False)
 
	infer.evaluate_folder_with_label_map("/work/21013187/phuoc/Image_Captionning_Transformer/data/license_plate_0-8/train/images",
                                      save_folder= "/work/21013187/phuoc/Image_Captionning_Transformer/results/valid_ver1",
                                       label_path="/work/21013187/phuoc/msi_license_plate/phuong_lp_map.csv",
                                       is_plate=True)
	
	# infer.evaluate_folder_with_label_map("/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number/images",
	# 										save_folder= "/work/21013187/phuoc/Image_Captionning_Transformer/results/valid_bien_xa_number",
    #                                    label_path="/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number/labels.csv",
    #                                    is_plate=True)
	infer.evaluate_folder_with_label_map("/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number_square/images",
											save_folder= "/work/21013187/phuoc/Image_Captionning_Transformer/results/valid_bien_xa_number",
                                       label_path="/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_xa_val_ver1_just_number/labels.csv",
                                       is_plate=True)
 
	# infer.crop_and_save_plate("/work/21013187/phuoc/Image_Captionning_Transformer/data/xemay",
    #                        	save_folder= "/work/21013187/phuoc/Image_Captionning_Transformer/data/xemay_plate_only")