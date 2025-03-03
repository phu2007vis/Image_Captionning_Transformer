# -*- coding: utf-8 -*-

import torch
import torch.nn as nn



class INCEPTION(torch.nn.Module):

	def __init__(self, all_config):
		super(INCEPTION, self).__init__()
		self.all_config =  all_config
		self.model_config = all_config['model']
		self.num_classes = self.model_config['num_classes']
		self.device = self.all_config['device']
		#Stem
		self.conv_1a = BasicConv2(3, 32, kernel_size=3, stride=2)
		self.conv_2a = BasicConv2(32, 32, kernel_size=3, stride=1)
		self.conv_2b = BasicConv2(32, 64, kernel_size=3, stride=1, padding=1)
		self.maxpool_3a = nn.MaxPool2d(3, stride=2)
		self.conv_3b = BasicConv2(64, 80, kernel_size=1, stride=1)
		self.conv_3c = BasicConv2(80, 192, kernel_size=3, stride=1)
		self.maxpool_4a = nn.MaxPool2d(3, stride=2)
		self.branch_0 = BasicConv2(192, 96, kernel_size=1, stride=1)
		self.branch_1 = nn.Sequential(
			BasicConv2(192, 48, kernel_size=1, stride=1),
			BasicConv2(48, 64, kernel_size=5, stride=1, padding=2)
		)
		self.branch_2 = nn.Sequential(
			BasicConv2(192, 64, kernel_size=1, stride=1),
			BasicConv2(64, 96, kernel_size=3,  stride=1, padding=1),
			BasicConv2(96, 96, kernel_size=3,  stride=1, padding=1)
		)
		self.branch_3 = nn.Sequential(
			nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
			BasicConv2(192, 64, kernel_size=1, stride=1)
		)
		#Inception A
		self.inception_a = nn.Sequential(
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17),
			Inception_Resnet_A(320, scale=0.17)
		)
		self.reduction_a = Reduction_A(320, 256, 256, 384, 384)
		self.inception_b = nn.Sequential(
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10),
			Inception_Resnet_B(1088, scale=0.10)
		)
		self.reduction_b = Reduction_B(1088)
		self.inception_c = nn.Sequential(
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20),
			Inception_Resnet_C(2080, scale=0.20)
		)
		self.inception_c_last = Inception_Resnet_C(2080, scale=0.20, activation=True)
		self.conv = BasicConv2(2080, 1536, kernel_size=1, stride=1)
		self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
		self.liner = nn.Linear(1536, self.num_classes)
	
	def features(self, input):
		#Stem
		x = self.conv_1a(input)
		x = self.conv_2a(x)
		x = self.conv_2b(x)
		x = self.maxpool_3a(x)
		x = self.conv_3b(x)
		x = self.conv_3c(x)
		x = self.maxpool_4a(x)
		x0 = self.branch_0(x)
		x1 = self.branch_1(x)
		x2 = self.branch_2(x)
		x3 = self.branch_3(x)
		x = torch.cat((x0, x1, x2, x3), dim=1)
		
		#Inception A
		x = self.inception_a(x)
		
		#Reduction A
		x = self.reduction_a(x)
		
		#Inception B
		x = self.inception_b(x)
		
		#Reduction B
		x = self.reduction_b(x)
		
		#Inception C
		x = self.inception_c(x)
		x = self.inception_c_last(x)
		
		x = self.conv(x)
		return x
	
	
	def logits(self, features):
		x = self.global_average_pooling(features)
		x = x.view(x.size(0), -1)
		x = self.liner(x)
		return x

	def forward(self, input):
		x = self.features(input)
		x = self.logits(x)
		return x
	def get_init_best_loss(self):
		return 9999
	def compare_best_loss(self,current_loss,best_loss):
		return current_loss < best_loss
	
	def setup_optimizer(self):
		optim_config = self.all_config['optim']
		self.optimizer = torch.optim.Adam(self.parameters(),**optim_config)
	def setup_loss_fn(self):
		self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.12)
	def fetch_data(self,data):
		
		x, labels =  data
		self.x = x.to(self.device)
		self.labels = labels.to(self.device)
  
		self.labels= nn.functional.one_hot(self.labels ,self.model_config.get('num_classes')).float()
	def get_output(self):
		
		outputs = self.outputs.detach().clone().cpu()
		values,indices = outputs.max(1)
		return indices.numpy()
	def get_label(self):
		labels = self.labels.detach().clone().cpu()
		values , indices = labels.max(1)
		return indices.numpy()
	def get_loss(self):
		return self.losses.detach().clone().cpu().item()

	def phuoc_forward(self):
		self.outputs = self.forward(self.x)
  
	def clear_gradient(self):
		self.optimizer.zero_grad()
	def phuoc_optimizer_step(self):
		self.losses.backward()
		self.optimizer.step()
	def do_loss(self):
		self.losses = self.loss_fn(self.outputs,self.labels)
		
	def phuoc_optimize(self):

		self.do_loss()
		self.clear_gradient()
		self.phuoc_optimizer_step()

class BasicConv2(nn.Module):
	def __init__(self, in_size, out_size, kernel_size, stride, padding = 0):
		super(BasicConv2, self).__init__()
		self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, 
								stride=stride, padding=padding, bias=False)
		self.batch_norm = nn.BatchNorm2d(out_size)
		self.relu = nn.ReLU(inplace=False)
	
	def forward(self, x):
		x = self.conv(x)
		x = self.batch_norm(x)
		x = self.relu(x)
		return x



class Inception_Resnet_A(nn.Module):
	def __init__(self, in_size, scale=1.0):
		super(Inception_Resnet_A, self).__init__()
		self.scale = scale
		self.branch_0 = BasicConv2(in_size, 32, kernel_size=1, stride=1)
		self.branch_1 = nn.Sequential(
			BasicConv2(in_size, 32, kernel_size=1, stride=1),
			BasicConv2(32, 32, kernel_size=3, stride=1, padding=1)
		)
		self.branch_2 = nn.Sequential(
			BasicConv2(in_size, 32, kernel_size=1, stride=1),
			BasicConv2(32, 48, kernel_size=3, stride=1, padding=1),
			BasicConv2(48, 64, kernel_size=3, stride=1, padding=1)
		)
		self.conv = nn.Conv2d(128, 320, stride=1, kernel_size=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch_0(x)
		x1 = self.branch_1(x)
		x2 = self.branch_2(x)
		out = torch.cat((x0, x1, x2), dim=1)
		out = self.conv(out)
		return self.relu(x + self.scale * out)

class Reduction_A(nn.Module):
	def __init__(self, in_size, k, l, m, n):
		super(Reduction_A, self).__init__()
		self.branch_0 = BasicConv2(in_size, n, kernel_size=3, stride=2)
		self.branch_1 = nn.Sequential(
			BasicConv2(in_size, k,kernel_size=1, stride=1),
			BasicConv2(k, l, kernel_size=3, stride=1, padding=1),
			BasicConv2(l, m, kernel_size=3, stride=2)
		)
		self.branch_2 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch_0(x) 
		x1 = self.branch_1(x)    
		x2 = self.branch_2(x)    
		return torch.cat((x0, x1, x2), dim=1)

class Inception_Resnet_B(nn.Module):
	def __init__(self, in_size, scale=1.0):
		super(Inception_Resnet_B, self).__init__()
		self.scale = scale
		self.branch_0 = BasicConv2(in_size, 192, kernel_size=1, stride=1)
		self.branch_1 = nn.Sequential(
			BasicConv2(in_size, 128, kernel_size=1, stride=1),
			BasicConv2(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
			BasicConv2(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
		)
		self.conv = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)
	def forward(self, x):
		x0 = self.branch_0(x)
		x1 = self.branch_1(x)
		out = torch.cat((x0, x1), dim=1)
		out = self.conv(out)
		return self.relu(out * self.scale + x)

class Reduction_B(nn.Module):
	def __init__(self, in_size):
		super(Reduction_B, self).__init__()
		self.branch_0 = nn.Sequential(
			BasicConv2(in_size, 256, kernel_size=1, stride=1),
			BasicConv2(256, 384, kernel_size=3, stride=2)
		)
		self.branch_1 = nn.Sequential(
			BasicConv2(in_size, 256, kernel_size=1, stride=1),
			BasicConv2(256, 288, kernel_size=3, stride=2)
		)
		self.branch_2 = nn.Sequential(
			BasicConv2(in_size, 256, kernel_size=1, stride=1),
			BasicConv2(256, 288, kernel_size=3, stride=1, padding=1),
			BasicConv2(288, 320, kernel_size=3, stride=2)
		)
		self.branch_3 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch_0(x)
		x1 = self.branch_1(x)
		x2 = self.branch_2(x)
		x3 = self.branch_3(x)
		return torch.cat((x0, x1, x2, x3), dim=1)

class Inception_Resnet_C(nn.Module):
	def __init__(self, in_size, scale=1.0, activation=False):
		super(Inception_Resnet_C, self).__init__()
		self.scale = scale
		self.activation = activation
		self.branch_0 = BasicConv2(in_size, 192, kernel_size=1, stride=1)
		self.branch_1 = nn.Sequential(
			BasicConv2(in_size, 192, kernel_size=1, stride=1),
			BasicConv2(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
			BasicConv2(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
		)
		self.conv = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)
	def forward(self, x):
		x0 = self.branch_0(x)
		x1 = self.branch_1(x)
		out = torch.cat((x0, x1), dim=1)
		out = self.conv(out)
		if self.activation:
			return self.relu(out * self.scale + x)
		return out * self.scale + x
	
if __name__ == "__main__":
	x = torch.randn(2,3,224,224)
	model = INCEPTION(8).to(device)
	output  = model(x)
	import pdb;pdb.set_trace()
