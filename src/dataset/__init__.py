import os
import importlib
from torch.utils.data import DataLoader
# Get the current file path and directory
file_path = os.path.abspath(__file__)
dir_file = os.path.dirname(file_path)
dir_name = os.path.basename(dir_file)

# Dictionary to register datasets
dataset_register = {}

for dataset_name in os.listdir(dir_file):
    if dataset_name.endswith('_dataset.py'):
        # Remove the file extension to get the module name
        module_name = dataset_name.replace('.py', '')

        # Import the module from the 'dataset' subdirectory
        try:
            module = importlib.import_module(f"{dir_name}.{module_name}")
        except ModuleNotFoundError as e:
            print(f"Failed to import module {module_name}: {e}")
            continue

        # Map the dataset name by removing '_dataset' from the module name
        map_name = module_name.replace('_dataset', '')

        # Ensure the mapped name exists as an attribute in the module
        if hasattr(module, map_name):
            dataset_register[map_name] = getattr(module, map_name)
        else:
            print(f"Attribute '{map_name}' not found in module '{module_name}'")

def get_dataset(dataset_config):
    dataset_name = dataset_config.get('name')
    return dataset_register[dataset_name](dataset_config)
def get_dataloader(dataset_config):
    
     dataset = get_dataset(dataset_config)
     loader_config =dataset_config.get('loader_config')
     return DataLoader(dataset,**loader_config)