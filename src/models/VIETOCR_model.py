from layer.resnet import Resnet50
from layer.vgg import vgg19_bn
from torch import nn
from layer.viet_transformer import LanguageTransformer
from utils.dowload_weight import download_weights
import torch
import os
from copy import deepcopy
from torch.optim import Adam,AdamW

class VIETOCR(nn.Module):
	def __init__(
		self,
	   config
	):

		super(VIETOCR, self).__init__()
		self.all_config = config
		self.model_config = config['model']
		self.device = self.all_config['device']
		hidden_dim = self.model_config['hidden_dim']

		if self.model_config['backbone'] == 'resnet':
			self.cnn = Resnet50(hidden_dim)
		else:
			print('VGG19 backbone')
			self.cnn = vgg19_bn(**self.model_config['cnn'])
		
		max_length_pe = self.model_config['transformers'].pop('max_seq_length')
		self.model_config['transformers']['max_seq_length'] = 1024
		self.transformer =  LanguageTransformer(**self.model_config['transformers'])
		self.load_transformer_weight()
  
		dropout = self.model_config.get('transformers').get("pos_dropout")
		d_model = self.model_config.get('transformers').get("d_model")
		self.transformer.replace_pe(d_model,max_seq_length = max_length_pe,dropout = dropout)
	# def move(self):
	# 	self.cnn = 
	def phuoc_forward(self):
		"""
		Shape:
			- img: (N, C, H, W)
			- tgt_input: (T, N)
			- tgt_key_padding_mask: (N, T)
			- output: b t v
		"""
		img, tgt_input, tgt_key_padding_mask = self.img,self.tgt_input,self.tgt_key_padding_mask
		src = self.cnn(img)
		
		self.outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
	
	def get_init_best_loss(self):
		return 9999
	def compare_best_loss(self,current_loss,best_loss):
		return current_loss < best_loss
	
	def setup_optimizer(self):
		optim_config = self.all_config['optim']
		self.optimizer = AdamW(self.parameters(),**optim_config)
  
	def setup_loss_fn(self):
		self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	def fetch_data(self,data):
		self.img,self.tgt_input,self.tgt_key_padding_mask,self.labels   = data['img'].to(self.device),data['tgt_input'].to(self.device),data['tgt_padding_mask'].to(self.device),data['tgt_output']
		self.labels = self.labels.to(self.device).view(-1)
	def get_output(self):
		
		outputs = self.outputs.detach().clone().cpu()

		indices = outputs.max(-1)[1].view(-1)
		return indices.numpy()

	def get_label(self):
		labels = self.labels.detach().clone().cpu()
		
		return labels.numpy()
	def get_loss(self):
		return self.losses.detach().clone().cpu().item()

	def forwar(self):
		self.phuoc_forward()
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
	def load_transformer_weight(self,transformer_weight = "https://vocr.vn/data/vietocr/vgg_transformer.pth"):
		print("Load transformer weights from {}".format(transformer_weight))
		weight_file = download_weights(transformer_weight)
		state_dict = torch.load(weight_file,map_location='cpu')
	   
		
		current_model_state = self.state_dict()
		for key in state_dict.keys():
			
			if key in ['transformer.embed_tgt.weight','transformer.fc.weight','transformer.fc.bias']:
				continue
			if key.startswith('cnn') :
				new_key = key.replace('model.','')
			else:
				new_key = key
			
			if new_key in current_model_state.keys():
				print(new_key)
				current_model_state[new_key] = deepcopy(state_dict[key])
		
		# self.transformer.load_state_dict(transformer_weights,strict=False)