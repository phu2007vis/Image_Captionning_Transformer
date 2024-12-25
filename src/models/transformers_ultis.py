import torch.nn as nn
import torch


def get_activation(name):
	return getattr(nn,name)()
	# if name == 'ReLU':
	# 	return nn.ReLU()
	# elif name == 'GELU':
	# 	return nn.GELU()
	
class Embedding(nn.Module):
	def __init__(self,config):
		super(Embedding, self).__init__()

		self.config = config
		self.d_model = self.config.get('d_model')
		self.max_length = self.config.get('max_length')
		
		self.possition_embedding = nn.Embedding(self.max_length,self.d_model)

	def forward(self,x):
		length = x.shape[1]
		
		return x + self.possition_embedding(torch.arange(0,length,device  = x.device))
def get_mlp_head(infeatures,outfeatuers,activation = 'GELU',dropout = 0.1):
	return nn.Sequential(
		nn.Linear(infeatures,infeatures*2),
		get_activation(activation),
		nn.Dropout(dropout),
		nn.Linear(infeatures*2,outfeatuers)
	)
class FeedForward(nn.Module):
	def __init__(self, config):
		super(FeedForward, self).__init__()
		self.config = config

		# Extract model configurations with proper defaults
		d_model = self.config.get('d_model')  # Default to 512 if not specified
		hidden = self.config.get('d_hidden')  # Default to 2048 if not specified
		drop_prob = self.config.get('drop_prob')  # Default to 0.1 if not specified

		# Layers
		self.linear1 = nn.Linear(d_model, hidden)
		self.linear2 = nn.Linear(hidden, d_model)
		self.activation = get_activation(self.config.get('activation', 'relu'))  # Default to ReLU
		self.dropout = nn.Dropout(p=drop_prob)

	def forward(self, x):
		
		x = self.linear1(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.linear2(x)
		return x
	
	
# ver1
# class ImageTokenizer(nn.Module):
# 	def __init__(self,config):
# 		super().__init__()
# 		self.config = config['model']
# 		self.patch_size = self.config.get('patch_size')
# 		self.image_size = self.config.get('image_size')
# 		self.proj = nn.Linear(self.patch_size*self.patch_size*3,self.config['d_model'])
# 	def forward(self,x): 
# 		B,C,W,H = x.shape
# 		assert(W*H % self.patch_size*self.patch_size == 0)
# 		# B,W*H,C
# 		x = x.view(B,C,W*H).transpose(1,2)
# 		N = W*H//(self.patch_size*self.patch_size)

# 		x = x.view(B,N,self.patch_size*self.patch_size,C).reshape(B,N,self.patch_size*self.patch_size*C).contiguous()
# 		x = self.proj(x)
# 		# B , N , 
# 		return x

#ver2 
from math import floor
class ImageTokenizer(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config['model']
		self.patch_size = self.config.get('patch_size')
		self.stride = self.config.get('stride')
		self.image_size = self.config.get('image_size')
		self.conv_proj = nn.Conv2d(in_channels=3,out_channels=self.config['max_length'],kernel_size =self.patch_size,stride = self.patch_size,padding=0)

		num_seq = floor((self.image_size-self.patch_size)/self.patch_size)+1
		self.proj = nn.Linear(num_seq*num_seq,self.config['d_model'])
  
	def forward(self,x): 

		x = self.conv_proj(x)
		B,C,W,H = x.shape
		x = x.view(B,C,W*H)
		x = self.proj(x)
		return x