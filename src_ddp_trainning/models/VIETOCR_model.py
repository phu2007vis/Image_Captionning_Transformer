from layer.resnet import Resnet50
from layer.vgg import vgg19_bn
from layer.viet_transformer import LanguageTransformer
from layer.efficient import Efficient
from torch import nn
from torch.optim import AdamW
from torch.nn.functional import softmax
import torch
import numpy as np
from copy import deepcopy

class VIETOCR(nn.Module):
	def __init__(self, config, gpu_id=None):
		super(VIETOCR, self).__init__()
		self.all_config = config
		self.model_config = config['model']
	 
		
		hidden_dim = self.model_config['hidden_dim']
		
		# CNN Backbone
		if self.model_config['backbone'] == 'resnet':
			self.cnn = Resnet50(hidden_dim)
		elif self.model_config['backbone'] == 'vgg19_bn':
			print('VGG19 backbone')
			self.cnn = vgg19_bn(**self.model_config['cnn'])
		else:
			self.cnn = Efficient(hidden_dim)
		
		# Transformer Setup
		transformer_config = self.model_config['transformers'].copy()
		max_length_pe = transformer_config.pop('max_seq_length')
		transformer_config['max_seq_length'] = 1024
		self.transformer = LanguageTransformer(**transformer_config)
		dropout = transformer_config.get("pos_dropout")
		d_model = transformer_config.get("d_model")
		self.transformer.replace_pe(d_model, max_seq_length=max_length_pe, dropout=dropout)
		
		# Load pretrained weights
		self.load_transformer_weight()
		
		# Freeze transformer if specified
		if not self.model_config.get('transformer_fine_tune', True):
			print("Turning off transformer fine-tuning!")
			for param in self.transformer.parameters():
				param.requires_grad = False
 
	
		# Parameter counts
		print(f"CNN parameters: {sum(p.numel() for p in self.cnn.parameters())}")
		print(f"Transformer parameters: {sum(p.numel() for p in self.transformer.parameters())}")

	def forward(self,img,tgt_input,tgt_padding_mask,**kwargs):
		"""
		Shape:
			- img: (N, C, H, W)
			- tgt_input: (T, N)
			- tgt_key_padding_mask: (N, T)
			- output: (B, T, V)
		"""

		# img, tgt_input, tgt_key_padding_mask = self.img, self.tgt_input, self.tgt_key_padding_mask
		src = self.cnn(img)
		outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
		return outputs
	def proprocessing_output(self,output):
		B,T,C = output.shape
		return output.view(B*T,C)
	def translate(self, img, max_seq_length=10, sos_token=1, eos_token=2):
		self.eval()
		with torch.no_grad():
			src = self.cnn(img)
			memory = self.transformer.forward_encoder(src)
			translated_sentence = [[sos_token] * len(img)]
			char_probs = [[1] * len(img)]
			max_length = 0

			while max_length <= max_seq_length and not all(
				np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
			):
				tgt_inp = torch.LongTensor(translated_sentence).to(self.device)
				output, memory = self.transformer.forward_decoder(tgt_inp, memory)
				output = softmax(output, dim=-1)
				values, indices = torch.topk(output, 1)
				indices = indices[:, -1, 0].tolist()
				values = values[:, -1, 0].tolist()
				char_probs.append(values)
				translated_sentence.append(indices)
				max_length += 1

			translated_sentence = np.asarray(translated_sentence).T
			char_probs = np.asarray(char_probs).T
			return translated_sentence, char_probs

	def get_label(self,data):
		return data['tgt_output'].view(-1)

	def load_transformer_weight(self):
		weight_file = self.model_config.get('transformer_pretrained') or self.model_config.get('weight_pretrained')
		if not weight_file:
			print("No pretrained weight file specified, skipping!")
			return
		
		try:
			print(f"Loading pretrained weights from {weight_file}")
			checkpoint = torch.load(weight_file, map_location='cpu')
			state_dict = checkpoint.get('model', checkpoint)  # Fallback to full checkpoint
			unmatched = self.load_state_dict(state_dict, strict=False)
			if unmatched.missing_keys:
				print(f"Missing keys: {unmatched.missing_keys}")
			if unmatched.unexpected_keys:
				print(f"Unexpected keys: {unmatched.unexpected_keys}")
			print("Pretrained weights loaded successfully")
		except Exception as e:
			print(f"Error loading weights: {e}")
			raise

def get_model(config, gpu_id=None):
	return VIETOCR(config, gpu_id)