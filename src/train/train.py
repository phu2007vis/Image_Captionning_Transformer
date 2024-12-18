import argparse
import os
import sys

# Add the parent directory to the system path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Absolute imports based on the package hierarchy
from trainer import Trainer
from models.utils import load_config



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config',default=r'C:\Users\9999\phuoc\transformer\src\configs\config.yaml')
	
	args = parser.parse_args()
	config = load_config(args.config)
	trainer = Trainer(config)

	import pdb;pdb.set_trace()