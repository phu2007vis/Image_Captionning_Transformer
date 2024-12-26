import os 
import sys 
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset import get_dataloader
from models import get_model
from tqdm import tqdm
import metrics
import torch

class Runner(object):
    def __init__(self, config):
        
        self.config = config
        self.model = get_model(config)
        print(f"Device: {self.config['device']}")
        
        self.load_pretrained_model()
        
        self.save_path = self.config.get('save_folder')
        
        self.setup_dataset()
        
    
    def setup_dataset(self):
        
        dataset_config = self.config.get('dataset')
        self.dataloader_register = {}

        for phase in dataset_config.keys():
            assert phase in ['train', 'val', 'test'], f"Can't recognize phase {phase} , ensure it in [ train, val , test]"
            phase_config = dataset_config[phase]  
            self.dataloader_register[phase] = get_dataloader(phase_config)
        
                
   
    def evaluate_main(self):
        
        self.model.setup_loss_fn()
        self.evaluate('val')

    def evaluate(self, phase):
        dataloader = self.dataloader_register[phase]
        self.model.eval()

        # labels = []
        # outputs = []
        # with torch.no_grad():
        #     self.eval_loss = 0
        #     for data in dataloader:
        #         self.model.fetch_data(data)
        #         self.model.phuoc_forward()
        #         self.model.do_loss()
        #         loss = self.model.get_loss()
        #         output = self.model.get_output().tolist()
        #         label = self.model.get_label().tolist()
        #         labels.extend(label)
        #         outputs.extend(output)
        #         self.eval_loss += loss

        self.eval_loss /= len(dataloader)
        
       
        
        # print("Eval loss: ", self.eval_loss)
        # for metric_name in self.config['evaluate'].get('metrics', []):
        #     if hasattr(metrics, metric_name):
        #         value = getattr(metrics, metric_name)(self.model, dataloader, labels, outputs)
        #         print(f"{metric_name}: {value}")
        #     else:
        #         print(f"Metric {metric_name} not found.")

    
    def load_pretrained_model(self):
        pretrained_path = self.config['model']['pretrained']
        
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=self.config['device'])
            model_state_dict = state_dict['model']
            self.model.load_state_dict(state_dict=model_state_dict)
            print(f"Load pretrained model from {pretrained_path}")
        else:
            print(f"No pretrained model found at {pretrained_path}")
            exit()

   
