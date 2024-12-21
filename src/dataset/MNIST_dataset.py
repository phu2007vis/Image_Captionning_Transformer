
import torch
from torch.utils.data import Dataset
from torchvision import transforms ,datasets

class MNIST(Dataset):
	def __init__(self,config):
		self.config = config
		self.collate_fn = None
		assert config['phase'] in ['train', 'test'] , f"MNIST Not support this phase: {config['phase']}! train or test !"
		train = True if config['phase'] == 'train' else False
		self.image_size = int(config['image_size'])
		transform=transforms.Compose([
		transforms.Resize((self.image_size,self.image_size)),
        transforms.ToTensor(),
		 transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ])
		self.main_dataset = datasets.MNIST('../data', train=train, download=True,
                       transform=transform)
	def __len__(self):
		return len(self.main_dataset)
	def __getitem__(self,index):
		return self.main_dataset[index]