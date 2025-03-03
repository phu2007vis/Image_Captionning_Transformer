import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models import get_model
from dataset import get_dataloader

def ddp_setup(rank, world_size):
	"""
	Args:
		rank: Unique identifier of each process
		world_size: Total number of processes
		gpu_ids: List of GPU indices to use
	"""

	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "65532"
	
	init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
	def __init__(self, gpu_id: int, config):
		self.config = config
		self.gpu_id = gpu_id  # This is now an index into gpu_ids
		self.max_epochs = self.config['train']['epochs']
		self.save_every = self.config['train'].get('save_every', 1)
		
		model = get_model(config).to(self.gpu_id)
		model.device_rank = "cuda:"+str(self.gpu_id)
		self.model = DDP(model, device_ids=[self.gpu_id])
		self._setup_dataset()

	def _setup_dataset(self):
		dataset_config = self.config.get('dataset')
		self.dataloader_register = {}
		for phase in dataset_config.keys():
			assert phase in ['train', 'val', 'test'], f"Unrecognized phase {phase}"
			phase_config = dataset_config[phase]
			self.dataloader_register[phase] = get_dataloader(phase_config)

	def _run_epoch(self, epoch):
		self.train_data = self.dataloader_register['train']
		self.train_data.sampler.set_epoch(epoch)
		data_sample = next(iter(self.train_data))
		b_sz = len(data_sample[next(iter(data_sample))])
		print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

		for data in self.train_data:
			self.model.module.clear_gradient()
			self.model.module.fetch_data(data)
			self.model.module.phuoc_forward()
			self.model.module.phuoc_optimizer_step()
			loss = self.model.module.get_loss()
			print(f"[GPU{self.gpu_id}], loss: {loss}")

	def _save_checkpoint(self, epoch):
		ckp = self.model.module.state_dict()
		PATH = f"checkpoint_epoch_{epoch}.pt"
		torch.save(ckp, PATH)
		print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

	def train(self):
		for epoch in range(self.max_epochs):
			self._run_epoch(epoch)
			if self.gpu_id == 0 and epoch % self.save_every == 0:  # Save only on rank 0
				self._save_checkpoint(epoch)

def main(rank: int, world_size: int, config):
	ddp_setup(rank, world_size)
	trainer = Trainer(config=config, gpu_id=rank)
	trainer.train()
	destroy_process_group()

if __name__ == "__main__":
	import argparse
	from utils import load_config
	
	# Specify the GPUs you want to use (e.g., "0,1" for GPUs 0 and 1)
	selected_gpus = "1,2,3"  # Change this to your desired GPUs (e.g., "1,3")
	os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpus
	world_size = torch.cuda.device_count()
	
	print(f"Using GPUs: {selected_gpus}")
	print(f"Number of GPUs: {torch.cuda.device_count()}")  # Should match len(gpu_ids)
	
	parser = argparse.ArgumentParser(description='Simple distributed training job')
	parser.add_argument('--config', default=r'/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/plate_ocr_hw.yaml')
	args = parser.parse_args()
	
	config = load_config(args.config)
	
	mp.spawn(main, args=(world_size, config), nprocs=world_size)
	