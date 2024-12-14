from models import ViTransformers
from sklearn.metrics import accuracy_score
import torch


def accuracy_classification(model:ViTransformers,dataloader,labels = None, outputs = None):
	
	# if labels is not None and outputs is not None:
	# 	return accuracy_score(labels,outputs)
	labels = []
	outputs = []
	with torch.no_grad():
		for i,data in enumerate(dataloader):
			model.fetch_data(data)
			model.phuoc_forward()
			output = model.get_output().tolist()
			label = model.get_label().tolist()
			labels.extend(label)
			outputs.extend(output)
			# print(output)
			# print(label)
			# import pdb;pdb.set_trace()
	return accuracy_score(labels,outputs)
	
		
	

		