import os 
import sys 
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models import get_model

class Trainer(object):
	def __init__(self,config):
		self.model = get_model(config)
	def train_one_epoch(self):
		pass
	def train(self,epochs ):
		
		for epoch in epochs:
			self.train_one_epoch()
