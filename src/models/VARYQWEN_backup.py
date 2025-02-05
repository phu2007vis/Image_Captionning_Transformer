
import torch.nn as nn
import torch
from torch.optim import Adam
from transformers import Qwen2Config,Qwen2ForCausalLM
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

		self.decoder_config = self.model_config.get('decoder')
		qwen_config = Qwen2Config(**self.decoder_config)
		self.qwen = Qwen2ForCausalLM(qwen_config)
		self.mm_projector_vary =  nn.Linear(1024, 1024)
		self.pose_init()
		self.to(config.get('device'))
		self.device = config.get('device')
	def pose_init(self):	
	 
		from utils.dowload_weight import download
		
		from safetensors import safe_open
  
		checkpoint_link = r"https://huggingface.co/stepfun-ai/GOT-OCR2_0/resolve/main/model.safetensors"
		path  = download(checkpoint_link,name="model.safetensors")
		tensors = {}
		with safe_open(path, framework="pt", device="cpu") as f:
			
			for key in f.keys():
				tensors[key] = f.get_tensor(key)
	
		verbose = self.model_config['decoder']['verbose']
		self.custom_load_state(tensors,verbose= verbose)
		if verbose:
			total_params = 0
			for param in self.qwen.parameters():
				total_params += param.numel()
			print(f"Total qwen param: {total_params}")
   
	def custom_load_state(self, state, prefix='', verbose=False):
		from copy import deepcopy
		
		# Get current model's state dictionary
		current_state = self.qwen.state_dict()
		

		updated_state = {}
		unmatched_keys = []
		

		for key, value in state.items():
	  
			if "vision" in key :
				continue
			if key.startswith(prefix):
				
				new_key = key.replace(prefix, '')
				if new_key in current_state:
					old_shape = current_state[new_key].shape
					new_shape = value.shape
					if old_shape != new_shape:
						print(f"Skipping unmatched shape: {new_key}")
					else:
						updated_state[new_key] = deepcopy(value)
						if verbose:
							print(f"Mapped: {key} -> {new_key}")
				else:
					unmatched_keys.append(new_key)
			elif verbose:
				print(f"Skipping unmatched key: {key}")

		# Update the current state dictionary
		current_state.update(updated_state)
  
		current_state_projection = self.mm_projector_vary.state_dict()
		bias,weights = state['model.mm_projector_vary.bias'],state['model.mm_projector_vary.weight']
		if weights.shape == current_state_projection['weight']:
			if verbose:
				print("Update projection weights")
			current_state_projection['weight'] = deepcopy(weights)
			current_state_projection['bias'] = deepcopy(bias)
   
		
		# Load the updated state dictionary into the model
		self.qwen.load_state_dict(current_state, strict=False)

		if verbose:
			print(f"Updated {len(updated_state)} keys.")
			if unmatched_keys:
				print(f"Unmatched keys (not found in model): {unmatched_keys}")
	
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
		
		# b, C, W,H
		x = self.image_encoder(x)
		#b ,t ,d
		x = x.flatten(2).transpose(1,2)
  
		x = self.qwen(input_ids = None,
						attention_mask = None,
				  		inputs_embeds = x,
						use_cache = False)
  
		return x[0]
	def phuoc_forward(self):
		x  = self.x
		

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
	x = torch.rand(2,3,image_size,image_size).to('cuda')
	model  = VARYQWEN(config).to('cuda')
	from utils.gpu_watch import get_gpu_memory_usage_message
	optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
	loss_fn = nn.CrossEntropyLoss()
	with torch.no_grad():
			label = torch.rand(2*49,40).max(1)[1].to('cuda')
	for i in range(1000):
		output = model(x)
		
		output = output.view(2*49,40)
		loss = loss_fn(output,label)
		
		loss.backward()
		
		optimizer.step()
		optimizer.zero_grad()
		print(loss.item())
	
	print(get_gpu_memory_usage_message())
	del x
	del model
	
