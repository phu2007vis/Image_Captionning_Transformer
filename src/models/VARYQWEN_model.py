
import torch.nn as nn
import torch
from torch.optim import Adam
# just for testing purposes
if __name__ == '__main__':
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
from layer.vit_b import _build_vary
  
class VARYQWEN(nn.Module):
	def __init__(self,config):
		super().__init__()

		self.model_config = config.get('model')
		self.encoder_config = self.model_config.get('encoder')
		self.image_encoder = _build_vary(**self.encoder_config)
		self.to(config.get('device'))
		self.device = config.get('device')
	def get_init_best_loss(self):
		return 9999
	def compare_best_loss(self,current_loss,best_loss):
		return current_loss < best_loss
	
	def setup_optimizer(self):
		optim_config = self.all_config['optim']
		self.optimizer = Adam(self.parameters(),**optim_config)
	def setup_loss_fn(self):
		self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)
	def fetch_data(self,data):
			

		self.x = data.get("images").to(self.device)
		self.labels = data.get("encodings").view(-1).to(self.device)
	
		self.text = data.get("captions")
		# import pdb;pdb.set_trace()
		# self.labels= nn.functional.one_hot(self.labels ,self.config.get('num_classes')).float()
		self.attention_mask = data.get("attention_mask")
  
	def get_output(self):
		
		outputs = self.outputs.detach().clone().cpu()

		indices = outputs.max(-1)[1].view(-1)
		return indices.numpy()

	def get_label(self):
		labels = self.labels.detach().clone().cpu()
		
		return labels.numpy()
	def get_loss(self):
		return self.losses.detach().clone().cpu().item()

	def forward(self,x):
		
		x = self.image_encoder(x)
		
		return x
	def phuoc_forward(self):
		x  = self.x
		batch_size = x.shape[0]

		x = self.image_tokenizer(x)
	
		x = self.embbeding(x)
		
		for encoder in self.encoder:
			x = encoder(x)
		x =  self.mlp_head(x)

		self.outputs = x
	def clear_gradient(self):
		self.optimizer.zero_grad()
	def phuoc_optimizer_step(self):
		self.losses.backward()
		self.optimizer.step()
	def do_loss(self):
		outputs = self.outputs.view(-1,self.outputs.size(-1))
		self.losses = self.loss_fn(outputs,self.labels)
		
	def phuoc_optimize(self):

		self.do_loss()
		self.clear_gradient()
		self.phuoc_optimizer_step()
		

if __name__ == "__main__":
	# import yaml
	import os 
	import sys
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(os.path.dirname(SCRIPT_DIR))
	from utils import load_config
	config = load_config("src/configs/vary_qwen.yaml")
	
	image_size = config['model'].get('encoder').get('image_size')
	x = torch.rand(2,3,image_size,image_size)
	model  = VARYQWEN(config)
	# print(model)
	print(model(x).shape)







