
import sys 
import os 

folder_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(folder_path)

from models.transformers import ViTransformers

def get_model(config):
	model = ViTransformers(config)
	model.to(config.get('device'))
	return model