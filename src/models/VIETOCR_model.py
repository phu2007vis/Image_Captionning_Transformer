from layer.resnet import Resnet50
from layer.vgg import vgg19_bn
from torch import nn
from layer.viet_transformer import LanguageTransformer
from utils.dowload_weight import download_weights
import torch
import os
from copy import deepcopy
from torch.optim import AdamW
from losses.L1Smoothing import LabelSmoothingLoss
from torch.nn.functional import softmax,log_softmax
import numpy as np
# from utils.beam import Beam

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
		sum = 0 
		for param in self.cnn.parameters():
			# param.requires_grad = False
			sum += param.numel()
		print(f'Cnn parameters: {sum}')
		sum = 0 
		for param in self.transformer.parameters():
			param.requires_grad = False
			sum += param.numel()
		print(f'Transformer parameters: {sum}')

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
		# #N ,B, D -> B,D,N
		# src = src.permute(1,2,0)
		# src = self.proj(src)
		# #N,B,D
		# src = src.permute(2,0,1)
		
		self.outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
	def translate(self,img, max_seq_length=10, sos_token=1, eos_token=2):
		model = self
		model.eval()
		device = img.device

		with torch.no_grad():
			src = model.cnn(img)
			memory = model.transformer.forward_encoder(src)

			translated_sentence = [[sos_token] * len(img)]
			char_probs = [[1] * len(img)]

			max_length = 0

			while max_length <= max_seq_length and not all(
				np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
			):

				tgt_inp = torch.LongTensor(translated_sentence).to(device)
				output, memory = model.transformer.forward_decoder(tgt_inp, memory)
				output = softmax(output, dim=-1)
				output = output.to("cpu")
				
				values, indices = torch.topk(output, 1)

				indices = indices[:, -1, 0]
				indices = indices.tolist()

				values = values[:, -1, 0]
				values = values.tolist()
				char_probs.append(values)

				translated_sentence.append(indices)
				max_length += 1

				del output

			translated_sentence = np.asarray(translated_sentence).T

			char_probs = np.asarray(char_probs).T
			char_probs = np.multiply(char_probs, translated_sentence > 3)
			char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

			return translated_sentence
		
	def translate_beam_search(
    self,img, beam_size=4, candidates=1, max_seq_length=10, sos_token=1, eos_token=2
	):
		model = self
		# img: 1xCxHxW
		model.eval()
		device = self.device

		with torch.no_grad():
			src = model.cnn(img)
			memory = model.transformer.forward_encoder(src)  # TxNxE
			sent = self.beamsearch(
				memory,
				model,
				device,
				beam_size,
				candidates,
				max_seq_length,
				sos_token,
				eos_token,
			)

		return sent


	def beamsearch(
		self,
		memory,
		model,
		device,
		beam_size=4,
		candidates=1,
		max_seq_length=128,
		sos_token=1,
		eos_token=2,
	):
		# memory: Tx1xE
		model.eval()

		beam = Beam(
			beam_size=beam_size,
			min_length=0,
			n_top=candidates,
			ranker=None,
			start_token_id=sos_token,
			end_token_id=eos_token,
		)

		with torch.no_grad():
			#        memory = memory.repeat(1, beam_size, 1) # TxNxE
			memory = model.transformer.expand_memory(memory, beam_size)

			for _ in range(max_seq_length):

				tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
				decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

				log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
				beam.advance(log_prob.cpu())

				if beam.done():
					break

			scores, ks = beam.sort_finished(minimum=1)

			hypothesises = []
			for i, (times, k) in enumerate(ks[:candidates]):
				hypothesis = beam.get_hypothesis(times, k)
				hypothesises.append(hypothesis)

		return [1] + [int(i) for i in hypothesises[0][:-1]]
	def get_init_best_loss(self):
		return 0
	def compare_best_loss(self,current_loss,best_loss):
		return current_loss < best_loss
	
	def setup_optimizer(self):
		optim_config = self.all_config['optim']
		self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()),**optim_config)
  
	def setup_loss_fn(self):
		# self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
		self.loss_fn = LabelSmoothingLoss(
					self.model_config['transformers']['vocab_size'], padding_idx=0, smoothing=0.12
				)

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

	def forward(self):
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
				# print(new_key)
				current_model_state[new_key] = deepcopy(state_dict[key])
		
		# self.transformer.load_state_dict(transformer_weights,strict=False)