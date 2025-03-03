
import sys 
import os 
import importlib

folder_path = os.path.dirname(os.path.abspath(__file__))
folder_name = os.path.basename(folder_path)
base_folder = os.path.dirname(folder_path)

sys.path.append(base_folder)
model_register = {}
for file in os.listdir(folder_path):
	if file.endswith('_model.py'):
		# Remove the file extension to get the module name
		module_name = file.replace('.py', '')
	
		module = importlib.import_module(f'{folder_name}.{module_name}')
		class_name = module_name.replace("_model","")
		model_register[class_name] = getattr(module,class_name)
  

def get_model(config):
    model_name = config.get('model').get('model_name')
    model_initor = model_register[model_name]
    model = model_initor(config)
  
    return model
