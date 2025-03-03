
from utils.image_transforms import *

def load_config(file_path):
	from yaml import safe_load
	with open(file_path,'r') as f:
		config = safe_load(f)
	return config
