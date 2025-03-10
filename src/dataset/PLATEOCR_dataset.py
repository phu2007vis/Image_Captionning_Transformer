
import os
from PIL import Image
import numpy as np
import torch
from dataset.vocab import Vocab
from utils.resize_image import process_image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.viet_aug import ImgAugTransformV2,UpwardsShift,WrapAug

# from aug import *

class Collator(object):
	def __init__(self, masked_language_model=True):
		self.masked_language_model = masked_language_model

	def __call__(self, batch):
		filenames = []
		img = []
		target_weights = []
		tgt_input = []
		max_label_len = max(len(sample["word"]) for sample in batch)

		for sample in batch:
		
			img.append(sample["img"].unsqueeze(0))
			
			filenames.append(sample["img_path"])
			label = sample["word"]
			label_len = len(label)

			tgt = np.concatenate(
				(label, np.zeros(max_label_len - label_len, dtype=np.int32))
			)
			tgt_input.append(tgt)

			one_mask_len = label_len - 1

			target_weights.append(
				np.concatenate(
					(
						np.ones(one_mask_len, dtype=np.float32),
						np.zeros(max_label_len - one_mask_len, dtype=np.float32),
					)
				)
			)

		img = torch.cat(img,dim = 0)

		tgt_input = np.array(tgt_input, dtype=np.int64).T
		tgt_output = np.roll(tgt_input, -1, 0).T
		tgt_output[:, -1] = 0

	

		tgt_padding_mask = np.array(target_weights) == 0

		rs = {
			"img": img,
			"tgt_input": torch.LongTensor(tgt_input),
			"tgt_output": torch.LongTensor(tgt_output),
			"tgt_padding_mask": torch.BoolTensor(tgt_padding_mask),
			"filenames": filenames,
		}

		return rs

class PLATEOCR(Dataset):
	def __init__(
		self,
		config
	):				

		self.config = config
  
		root_dir = self.config['root_dir']
		masked_language_model =   self.config['masked_language_model']
		image_height=self.config['image_height']
		image_min_width = self.config['image_min_width']
		image_max_width= self.config['image_max_width']
		transform= self.config.get('transform')
  
		self.root_dir = root_dir
		self.annotation_path = os.path.join(root_dir, "labels")
		self.image_dir = os.path.join(root_dir, "images")
		self.annotations = os.listdir(self.annotation_path)
		self.vocab = Vocab(self.config['vocab'])
		self.collate_fn = Collator(masked_language_model=masked_language_model)
		self.numpy_transform = None
		if self.config['phase'] == 'train':
			
			self.transform = transforms.Compose([
				WrapAug(p =[0.7,0.7,0]),
				ImgAugTransformV2(),
    			# transforms.RandomApply([transforms.RandomRotation(degrees=10,fill= (255,255,255))]),
			
				transforms.ToTensor(),
			])
		elif self.config['phase'] == 'val':
		
			self.transform = transforms.Compose([
				
				transforms.ToTensor(),
				
			])
		else:
			print(f"Phase {self.config['phase']} not found in ['tran','val]")
			
		self.image_height = image_height
		self.image_min_width = image_min_width
		self.image_max_width = image_max_width
		print(f"Number samples: {len(self)}")

	

	def __getitem__(self, idx):
	 
		name = self.annotations[idx]
		txt_path = os.path.join(self.annotation_path,name)
  
		id = os.path.splitext(name)[0]
		image_name = id+".jpg"
		image_path = os.path.join(self.image_dir,image_name)
  
		with open(txt_path,'r') as f:
			word = f.read().strip()
		word = self.vocab.encode(word)
		pil_image = Image.open(image_path)
		pil_image = process_image(pil_image,self.image_height,self.image_min_width,self.image_max_width)
		if self.numpy_transform is not None:
			pil_image = Image.fromarray(self.numpy_transform(image=np.asarray(pil_image))["image"])
		tensor_img = self.transform(pil_image)
		# print(tensor_img.shape)
		return {
			'img': tensor_img,
			'word': word,
			'img_path': image_path
		}

	def __len__(self):
		return len(self.annotations)	
	@staticmethod
	def get_default_transform(image_size):
		global totensor 
		totensor = transforms.Compose([
			
				transforms.ToTensor(),
			])
		
		def default_transform(image):
			global totensor
			
			
			image_height = image_size
			image_max_width = 10000
			image_min_width = 120
			
			processed_image = process_image(image,image_height, image_min_width, image_max_width)
			
			return totensor(processed_image)
		return default_transform
	@staticmethod
	def get_default_visualize(image_size):
		global visulize
	
		def default_visualize(image):
			global visulize
			image_height = image_size
			image_min_width = 112
			image_max_width = 10000
			processed_image = process_image(image,image_height, image_min_width, image_max_width)
		
			return processed_image
		return default_visualize
	

