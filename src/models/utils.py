

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

		self.config = config.get('model')
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
        self.config = config['model']

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
	
class ImageTokenizer(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config['model']
		self.patch_size = self.config.get('patch_size')
		self.image_size = self.config.get('image_size')
		self.proj = nn.Linear(self.patch_size*self.patch_size*3,self.config['d_model'])
	def forward(self,x): 
		B,C,W,H = x.shape
		assert(W*H % self.patch_size*self.patch_size == 0)
		# B,W*H,C
		x = x.view(B,C,W*H).transpose(1,2)
		N = W*H//(self.patch_size*self.patch_size)

		x = x.view(B,N,self.patch_size*self.patch_size,C).reshape(B,N,self.patch_size*self.patch_size*C).contiguous()
		x = self.proj(x)
		# B , N , 
		return x

def load_config(file_path):
	from yaml import safe_load
	with open(file_path,'r') as f:
		config = safe_load(f)
	return config

if __name__ == "__main__":
	config = load_config(file_path="src/configs/config.yaml")
	# x = torch.arange(0,500).reshape(5,100)
	# embedding =Embedding(config)
	# embbed = embedding(x).float()

	# ffw = FeedForward(config)
	# ffw_x = ffw(embbed)
	# print(ffw_x.shape)
	img_tokenizer = ImageTokenizer(config)
	x = torch.rand(2,3,224,224)
	print(img_tokenizer(x).shape)

	import pdb;pdb.set_trace()