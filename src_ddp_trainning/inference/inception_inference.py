import os 
import sys 
from PIL import Image
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import load_config
from models import get_model
from dataset_folder.CLASSIFIER_dataset import CLASSIFIER
import torch
import cv2

names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
names = sorted(names)
class Inference(object):
	def __init__(self, config):
		if isinstance(config, str):
			config = load_config(config)
   
		self.config = config
		self.model = get_model(config)
		print(f"Device: {self.config['device']}")
		self.device = self.config['device']
		self.load_pretrained_model()
		self.model.eval()
		self.setup_default_transform()
	def setup_default_transform(self):
		self.transform = CLASSIFIER.get_val_transform(self.config['image_size'])
		
	def load_pretrained_model(self):
		pretrained_path = self.config['model']['pretrained']
	  
		if os.path.exists(pretrained_path):
			state_dict = torch.load(pretrained_path, map_location=self.config['device'])
			model_state_dict = state_dict['model']
			self.model.load_state_dict(state_dict=model_state_dict)
			print(f"Load pretrained model from {pretrained_path}")
			score = state_dict['best_loss']
			print(f"Best score: {score}")
		else:
			print(f"No pretrained model found at {pretrained_path}")
	def load_image(self,image_path):
		return Image.open(image_path).convert('RGB')
	def preprocessing_image(self,image_pil):
		return self.transform(image_pil).unsqueeze(0)
	def predict(self, image_tensor, top_k):
		with torch.no_grad():
			image_tensor = image_tensor.to(self.device)
			predict = self.model(image_tensor).squeeze()
			predict = torch.softmax(predict, dim=-1)
			
		# Get the top-k predictions and their respective probabilities
		conf, prediction = torch.topk(predict, top_k)

		return conf, prediction
	def predict_from_image(self,image):
		image_tensor = self.preprocess_image_cv2(image)
		with torch.no_grad():
			conf,predict = self.predict(image_tensor,top_k=5)
		
		return conf.item(),names[predict.item()]

	def preprocess_image_cv2(self,image):
		image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).convert('RGB')
		image_tensor = self.preprocessing_image(image)
		image_tensor = image_tensor.to(self.device)
		return image_tensor
  
	def predict_from_list_image(self,image_list,top_k ):
		image_list = [self.preprocess_image_cv2(image) for image in image_list ]
		with torch.no_grad():
			conf,predict = self.predict(torch.cat(image_list, dim=0).to(self.device),top_k=top_k)
		conf = conf.squeeze()
		predict = predict.squeeze()
		return conf.tolist(), predict.tolist()
	def predict_image_from_path(self,image_path):
		image_pil = self.load_image(image_path)
		image_tensor = self.preprocessing_image(image_pil)
		image_tensor = image_tensor.to(self.device)
		conf,predict = self.predict(image_tensor,top_k= 1)
		return conf.squeeze().item(),names[predict.squeeze().item()]

runner = Inference('/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/number_classifier.yaml')

if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--config',default=r'/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/number_classifier.yaml')
	args = parser.parse_args()
	config = load_config(args.config)
	
	
	root_folder = "/work/21013187/phuoc/License_Plate/data/test_300_clean"
	save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/results/test_reuslt_real"
	# root_folder = "/work/21013187/phuoc/License_Plate/data/OCR_classification_split_300_clean/val"
	# save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/results/test_reuslt_300"
	import shutil
	if os.path.exists(save_folder):
		shutil.rmtree(save_folder)
	count = 0 
	for cls in os.listdir(root_folder):
		sub_root_folder = os.path.join(root_folder,cls)
		
		for image_name in os.listdir(sub_root_folder):
			image_path = os.path.join(sub_root_folder,image_name)
			conf, prediction = runner.predict_image_from_path(image_path)
			if prediction == cls:
				root_save_folder =  os.path.join(save_folder,'dung')
			else:
				root_save_folder =  os.path.join(save_folder,'khong_dung')
			os.makedirs(root_save_folder,exist_ok=True)
			save_path = os.path.join(root_save_folder,f"{cls}->{prediction}->{image_name}")
			shutil.copy(image_path,save_path)

			if ((prediction.isdigit() and cls.isdigit()) or (not cls.isdigit() and not prediction.isdigit())) and prediction!= cls:
			
				count +=1
				root_save_folder =  os.path.join(save_folder,'khong_dung_possition')
				os.makedirs(root_save_folder,exist_ok=True)
				save_path = os.path.join(root_save_folder,f"{cls}->{prediction}->{image_name}")
				shutil.copy(image_path,save_path)
 
	total_dung = len(os.listdir(os.path.join(save_folder,'dung')))
	total_sai = len(os.listdir(os.path.join(save_folder,'khong_dung')))
	print(f"Total dung {total_dung}, total_sai {total_sai}, accuracy {total_dung/(total_dung+total_sai)}")
	print(f"Total sai khi them possition: {count}, accuracy: {100*(1-count/(total_dung+total_sai))}")