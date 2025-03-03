from transformers import Qwen2Model, Qwen2ForCausalLM,Qwen2Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from losses.L1Smoothing import LabelSmoothingLoss

# just for testing
if __name__ == '__main__':
	import os
	import sys
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(os.path.dirname(SCRIPT_DIR))
 
from layer.vit_b import build_vary_vit_b


class GOTConfig(Qwen2Config):
	model_type = "GOT"


class GOTQwenModel(Qwen2Model):
	config_class = GOTConfig

	def __init__(self, config: Qwen2Config):
		super(GOTQwenModel, self).__init__(config)

		self.vision_tower_high = build_vary_vit_b()

		self.mm_projector_vary =  nn.Linear(1024, 1024)


	def forward(
		self,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		images: Optional[torch.FloatTensor] = None,
  
	) -> Union[Tuple, BaseModelOutputWithPast]:

		vision_tower_high = getattr(self, 'vision_tower_high', None)
		
		with torch.set_grad_enabled(False):
			
			cnn_feature = vision_tower_high(images)
			cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # 256*1024
		#B,
		image_features  = self.mm_projector_vary(cnn_feature)
		
		return super(GOTQwenModel, self).forward(input_ids=None, attention_mask=attention_mask,inputs_embeds=image_features, use_cache=None, position_ids = position_ids)



class GOTQwenForCausalLM(Qwen2ForCausalLM):
	config_class = GOTConfig
	def __init__(self, config):
		super(Qwen2ForCausalLM, self).__init__(config)
	
		self.model = GOTQwenModel(config)
		self.loss_fct = CrossEntropyLoss()
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		self.post_init()
		self.load_pretrained_weights()
  
	def load_pretrained_weights(self):
		from utils.dowload_weight import download
		from safetensors import safe_open
  
		checkpoint_link = r"https://huggingface.co/stepfun-ai/GOT-OCR2_0/resolve/main/model.safetensors"
		path  = download(checkpoint_link,name="model.safetensors")
		tensors = {}
		with safe_open(path, framework="pt", device="cpu") as f:
			
			for key in f.keys():
				tensors[key] = f.get_tensor(key)
	
		verbose = self.model.config.verbose
		self.custom_load_state(tensors,verbose= verbose)
		if verbose:
			total_params = 0
			for param in self.parameters():
				total_params += param.numel()
			print(f"Total qwen param: {total_params}")
   
	def custom_load_state(self, state, prefix='', verbose=False):
		from copy import deepcopy
		
		# Get current model's state dictionary
		current_state = self.state_dict()

		updated_state = {}
		unmatched_keys = []
		
		for key, value in state.items():
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
  
	
		self.load_state_dict(current_state, strict=False)

		if verbose:
			print(f"Updated {len(updated_state)} keys.")
			if unmatched_keys:
				print(f"Unmatched keys (not found in model): {unmatched_keys}")
	def get_model(self):
		return self.model

	def forward(
		self,
		attention_mask: Optional[torch.Tensor] = None,
		images: Optional[torch.FloatTensor] = None,
	):
		
		outputs  = self.model(
			attention_mask=attention_mask,
			images=images,
			
		)
		hidden_states = outputs[0]
		logits = self.lm_head(hidden_states)
		logits = logits.float()

		return logits


	def prepare_inputs_for_generation(
		self, attention_mask
	):
	
		position_ids = attention_mask.long().cumsum(-1) - 1
		position_ids.masked_fill_(attention_mask == 0, 1)
	
		return position_ids

class GOTOCR(GOTQwenForCausalLM):
	def __init__(self, config):
		self.all_config = config
		self.model_config = config['model']
		GOTConfig  = Qwen2Config(**config['model']['decoder'])
		super().__init__(GOTConfig)
	def get_init_best_loss(self):
		return 9999
	def compare_best_loss(self,current_loss,best_loss):
		return current_loss < best_loss
	
	def setup_optimizer(self):
		optim_config = self.all_config['optim']
		self.optimizer = AdamW(self.parameters(),**optim_config)
  
	def setup_loss_fn(self):
		self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
		# self.loss_fn = LabelSmoothingLoss(
		# 			self.model_config['decoder']['vocab_size'], padding_idx=0, smoothing=0.05
		# 		)

	def fetch_data(self,data):
		self.img,self.tgt_input,self.tgt_key_padding_mask,self.labels   = data['img'].to(self.device),data['tgt_input'].to(self.device),data['tgt_padding_mask'].to(self.device),data['tgt_output']
		self.labels = self.labels.to(self.device).view(-1)
	def get_output(self):
		
		outputs = self.outputs.detach().clone().cpu()

		indices = outputs.max(-1)[1].view(-1)
		return indices.numpy()
	def phuoc_forward(self):
		
		self.outputs = super(GOTOCR,self).forward(images=self.img)
	def get_label(self):
		labels = self.labels.detach().clone().cpu()
		
		return labels.numpy()
	def get_output_label(self):
		# Retrieve labels and outputs
		labels = self.get_label()
		outputs = self.get_output()

		# Ensure labels and outputs have the same length
		assert len(labels) == len(outputs), "Labels and outputs must have the same length"

		# Filter out labels and outputs where labels are 0
		filtered_indices = labels != 0  # Create a boolean mask for labels != 0
		filtered_labels = labels[filtered_indices]
		filtered_outputs = outputs[filtered_indices]
	
		return filtered_outputs,filtered_labels
	def get_loss(self):
		return self.losses.detach().clone().cpu().item()

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
	
	from transformers import Qwen2Config
	model = GOTOCR(config).to('cuda')

	
	from utils.gpu_watch import get_gpu_memory_usage_message
	optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
	loss_fn = nn.CrossEntropyLoss()
	x = torch.randn(2,3,1024,1024).to('cuda')
	with torch.no_grad():
			label = torch.rand(2*256,40).max(1)[1].to('cuda')
	for i in range(1000):
		output = model(images = x)
		
		output = output.view(2*256,40)
		loss = loss_fn(output,label)
		
		loss.backward()
		
		optimizer.step()
		optimizer.zero_grad()
		print(loss.item())
	
	print(get_gpu_memory_usage_message())
	del x
	del model
	