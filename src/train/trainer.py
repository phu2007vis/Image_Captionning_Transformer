import os 
import sys 
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset import get_dataset,get_dataloader
from models import get_model

class Trainer(object):
	def __init__(self,config):
		self.model = get_model(config)
		self.config = config
		self.setup_dataset()
	def setup_dataset(self):
		dataset_config = self.config.get('dataset')
		#init dataset skateholder
		self.dataset_register = {}

		for phase in dataset_config.keys():
			assert phase in ['train','val','test'], f"Can't recognize phase {phase} , ensure it in [ train, val , test]"
			phase_config = dataset_config[phase]  
			self.dataset_register[phase] = get_dataloader(phase_config)
			
	def train_one_epoch(self):
		pass
	def train(self,epochs ):
		
		for epoch in epochs:
			self.train_one_epoch()
