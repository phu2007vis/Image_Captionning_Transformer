import argparse
import os
import sys

# Add the parent directory to the system path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Absolute imports based on the package hierarchy
from trainer import Trainer
from utils import load_config



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config',default=r'/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/plate_ocr_hw.yaml')
	# parser.add_argument('--config',default=r'/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/plate_ocr.yaml')
	args = parser.parse_args()
	config = load_config(args.config)
	trainer = Trainer(config)
