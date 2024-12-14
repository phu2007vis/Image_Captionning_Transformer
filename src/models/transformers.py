import torch.nn as nn
import torch
from attention import MultiHeadAttention
from utils import FeedForward,ImageTokenizer,get_mlp_head,Embedding
from torch.optim import Adam


class Encoder(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.config = config['model']
		self.multi_attention = MultiHeadAttention(config)
		self.norm1 = nn.LayerNorm(self.config['d_model'])
		self.ffc = FeedForward(config)
		self.norm2 = nn.LayerNorm(self.config['d_model'])
		self.dropout = nn.Dropout(self.config['dropout'])
	def forward(self,x):
		attetion_x = self.multi_attention(x,x,x)
		norm1_x = self.norm1(attetion_x + x)
		ffc_x = self.ffc(norm1_x)
		return self.norm2(ffc_x+norm1_x)


class ViTransformers(nn.Module):
	def __init__(self,config):
		super().__init__()
		self.all_config = config
		self.config = config['model']
		self.classifier_token = nn.Parameter(torch.randn(1,self.config['d_model']))
		self.embbeding = Embedding(config)
		self.encoder = nn.ModuleList([Encoder(config) for _ in range(self.config['num_layers'])])
		self.image_tokenizer = ImageTokenizer(config)
		self.mlp_head = get_mlp_head(self.config['d_model'],self.config['num_classes'])
		self.to(config.get('device'))
		self.device = config.get('device')
		
	def setup_optimizer(self):
		optim_config = self.all_config['optim']
		self.optimizer = Adam(self.parameters(),**optim_config)
	def setup_loss_fn(self):
		self.loss_fn = nn.CrossEntropyLoss()
	def fetch_data(self,data):
		x, labels =  data
		self.x = x.to(self.device)
		self.labels = labels.to(self.device)
		self.labels= nn.functional.one_hot(self.labels % self.config.get('num_classes')).float()
	
	def get_loss(self):
		return self.losses.detach().clone().cpu().item()

	def forward(self,x):
		
		batch_size = x.shape[0]

		x = self.image_tokenizer(x)
		
		x = torch.cat([self.classifier_token.repeat(batch_size,1,1),x],dim = 1)
	
		x = self.embbeding(x)
		
		for encoder in self.encoder:
			x = encoder(x)

		x = x[:,0,:]
		x =  self.mlp_head(x)

		return x
	def phuoc_forward(self):
		x  = self.x
		batch_size = x.shape[0]

		x = self.image_tokenizer(x)
		
		x = torch.cat([self.classifier_token.repeat(batch_size,1,1),x],dim = 1)
	
		x = self.embbeding(x)
		
		for encoder in self.encoder:
			x = encoder(x)

		x = x[:,0,:]
		x =  self.mlp_head(x)

		self.outputs = x
	def clear_gradient(self):
		self.optimizer.zero_grad()
	def phuoc_optimizer_step(self):
		self.losses.backward()
		self.optimizer.step()
	def phuoc_optimize(self):
		
		self.losses = self.loss_fn(self.outputs,self.labels)
		self.clear_gradient()
		self.phuoc_optimizer_step()
		

if __name__ == "__main__":
	# import yaml
	from utils import Embedding,load_config
	config = load_config("src/configs/config.yaml")
	
	x = torch.rand(2,3,224,224)
	# embedding =Embedding(config)
	# embbed = embedding(x).float()

	# encoder = Encoder(config)
	# print(encoder(embbed).shape)
	model  = ViTransformers(config)
	print(model)
	print(model(x).shape)