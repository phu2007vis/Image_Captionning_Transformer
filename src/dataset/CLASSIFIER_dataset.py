
from torch.utils.data import Dataset
from torchvision import transforms ,datasets
import numpy as np
import cv2
from PIL import Image
from utils.resize_image import ProcessImageV2
from utils.image_transforms import RandomErasing
class GaussianBlur():
	def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
		self.sigma_min = sigma_min
		self.sigma_max = sigma_max
		self.kernel_size = kernel_size

	def __call__(self, img):
		sigma = np.random.uniform(self.sigma_min, self.sigma_max)
		img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
		return Image.fromarray(img.astype(np.uint8))

class CLASSIFIER(Dataset):
	def __init__(self,config):
		self.config = config
		self.collate_fn = None
		assert config['phase'] in ['train', 'test'] , f"MNIST Not support this phase: {config['phase']}! train or test !"
		train = True if config['phase'] == 'train' else False
		self.train = train
		self.image_size = int(config['image_size'])
  
		list_posts_transform = [transforms.ToTensor()]
		if train:
			print(f"Phases: {config['phase']} with train transform")
			transform = transforms.Compose([
										# transforms.Pad(padding = 15,fill = 255),
										transforms.Grayscale(num_output_channels=3),
										transforms.RandomRotation(config['aug_rotate'],fill = 255),
										ProcessImageV2(180),
										transforms.RandomApply([GaussianBlur(kernel_size=21)], p=1),
					  					transforms.ColorJitter(brightness = 0.3,contrast=0.2),
										# transforms.ToTensor(),
		  								# transforms.RandomErasing(p = 0.5,scale=(0.02, 0.1), ratio=(0.3, 3.3),value=1.0),
										# RandomErasing(p = 0.4,r1 = 0.2)
				  						])
			list_posts_transform.append(RandomErasing(p = 0.4,r1 = 0.2))
			self.train_transform = {}
			self.train_transform[6] = transforms.Compose([transforms.ToTensor()])
			self.train_transform[8] = transforms.Compose([transforms.ToTensor()])
		else:
			print(f"Phases: {config['phase']} with default transform")
			transform = transforms.Compose([
       
										transforms.Grayscale(num_output_channels=3),
	   									ProcessImageV2(180),
										# transforms.ToTensor()
		  									])
		
		self.post_transform = transforms.Compose(list_posts_transform)
		data_dir = config['data_dir']
		self.main_dataset = datasets.ImageFolder(data_dir, transform=transform)
	def __len__(self):
		return len(self.main_dataset)
	def __getitem__(self,index):
		image,label = self.main_dataset[index]
	
		if self.train and (self.train_transform.get(label) is not None):
			image = self.train_transform[label](image)
		else:
			image = self.post_transform(image)
		
		return image,label
	@staticmethod
	def get_val_transform(image_size):
		return  transforms.Compose([ProcessImageV2(image_size),
										transforms.ToTensor()])