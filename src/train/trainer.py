import os 
import sys 
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset import get_dataloader
from models import get_model
from tqdm import tqdm

class Trainer(object):
	def __init__(self,config):
		self.model = get_model(config)
		self.config = config
		self.save_path = self.config.get('save_folder')
		self.setup_dataset()
		self.train()
	def setup_dataset(self):
		dataset_config = self.config.get('dataset')
		#init dataset skateholder
		self.dataloader_register = {}

		for phase in dataset_config.keys():
			assert phase in ['train','val','test'], f"Can't recognize phase {phase} , ensure it in [ train, val , test]"
			phase_config = dataset_config[phase]  
			self.dataloader_register[phase] = get_dataloader(phase_config)

			# for _ in  tqdm(self.dataloader_register[phase],desc = f'Testing {phase}dataset '):
			# 	pass
				
	def train_one_epoch(self):
		self.model.train()
		self.pbar = tqdm(enumerate(self.dataloader_register['train']),total= len(self.dataloader_register['train']))
		for i,data in self.pbar:
			self.model.fetch_data(data)
			self.model.phuoc_forward()
			self.model.phuoc_optimize()
			loss = self.model.get_loss()
			# print(self.model.losses)
			if (i+1) %self.config['train']['print_frequency'] == 0:
				self.pbar.set_description(f"Epoch {self.epoch+1}/{self.config['train']['epochs']}, iter {i+1}, Train loss: {loss}")
		
	def train(self ):
		self.model.setup_optimizer()
		self.model.setup_loss_fn()
		self.train_iter = 0
		for epoch in range(self.config['train']['epochs']):
			self.epoch = epoch
			self.train_one_epoch()
			
