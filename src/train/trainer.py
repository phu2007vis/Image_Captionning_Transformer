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
import os
import torch


class Trainer(object):
    def __init__(self, config):
        
        self.config = config
        self.model = get_model(config)
        print(f"Device: {self.config['device']}")
        
        self.load_pretrained_model()
        
        self.save_path = self.config.get('save_folder')
        
        self.setup_dataset()
        self.train()
    
    def setup_dataset(self):
        
        dataset_config = self.config.get('dataset')
        self.dataloader_register = {}

        for phase in dataset_config.keys():
            assert phase in ['train', 'val', 'test'], f"Can't recognize phase {phase} , ensure it in [ train, val , test]"
            phase_config = dataset_config[phase]  
            self.dataloader_register[phase] = get_dataloader(phase_config)

        self.visualize()
        # exit()

    def visualize(self):
        import math
        import matplotlib.pyplot as plt
        n = 0
        num_images = 16  # Total number of images to display
        grid_size = math.ceil(math.sqrt(num_images))  # Determine the size of the grid (square)

        fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.5, hspace=0.8)  # Adjust spacing between subplots

        for batch in self.dataloader_register['train']:
            for i in range(10):
                if n >= num_images:
                    break

                img = batch["img"][i].numpy().transpose(1, 2, 0)
                sent = self.dataloader_register['train'].dataset.vocab.decode(batch["tgt_input"].T[i].tolist())

                # Calculate row and column for the subplot
                row, col = divmod(n, grid_size)
                ax = axs[row, col]
                ax.imshow(img)
                ax.set_title(f"sent: {sent}", fontname="serif")
                ax.axis("off")

                n += 1

            if n >= num_images:
                break

        # Remove any unused subplots if grid is larger than required
        for i in range(n, grid_size * grid_size):
            row, col = divmod(i, grid_size)
            axs[row, col].axis("off")

        plt.savefig("dataset.png", bbox_inches="tight")  # Save the figure
    

    def train_one_epoch(self):
        
        self.model.train()
        self.pbar = tqdm(enumerate(self.dataloader_register['train']), total=len(self.dataloader_register['train']))
        
        for i, data in self.pbar:
            
            self.model.fetch_data(data)
         
            self.model.phuoc_forward()

            self.model.phuoc_optimize()
        
            loss = self.model.get_loss()
    
            self.train_iter += 1
            if self.train_iter % self.config['train']['print_frequency'] == 0:
                self.pbar.set_description(f"Epoch {self.epoch + 1}/{self.config['train']['epochs']}, iter {self.train_iter}, Train loss: {loss}")
            if self.train_iter % self.config['evaluate']['frequency'] == 0:
                self.evaluate(phase='val')
    
    def train(self):
        self.best_loss = self.model.get_init_best_loss()
        self.model.setup_optimizer()
        self.model.setup_loss_fn()

        # setup save folder
        if self.config['train'].get('save_folder'):
            save_folder = os.path.join(self.config['train']['save_folder'], self.config['model']['model_name'])
            index = self.get_index_save_folder(save_folder)
            self.save_folder = os.path.join(save_folder, str(index))
            os.makedirs(self.save_folder, exist_ok=True)
            print(f"Save model {self.config['model']['model_name']} at {os.path.abspath(self.save_folder)}")
        
        self.train_iter = 0
        for epoch in range(self.config['train']['epochs']):
            self.epoch = epoch
            self.train_one_epoch()

    def evaluate(self, phase):
        dataloader = self.dataloader_register[phase]
        self.model.eval()

        labels = []
        outputs = []
        with torch.no_grad():
            self.eval_loss = 0
            for data in dataloader:
                self.model.fetch_data(data)
                self.model.phuoc_forward()
                self.model.do_loss()
                loss = self.model.get_loss()
                output = self.model.get_output().tolist()
                label = self.model.get_label().tolist()
                output = self.model.get_output().tolist()


                filtered = [(o, l) for o, l in zip(output, label) if l != 0]
                outputs.extend([o for o, l in filtered])
                labels.extend([l for o, l in filtered])

                # labels.extend(label)
                # outputs.extend(output)
                self.eval_loss += loss
        self.eval_loss /= len(dataloader)
        print("eval_loss: ", self.eval_loss)
        
    
        for metric_name in self.config['evaluate'].get('metrics', []):
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)(self.model, dataloader, labels, outputs)
                print(f"{metric_name}: {value}")
                metric = value
            else:
                print(f"Metric {metric_name} not found.")
    
        if self.config['train'].get('save_folder'):
            if self.model.compare_best_loss(self.best_loss, metric):
                self.save_path = os.path.join(self.save_folder, "best.pt")
                self.best_loss = metric
                lstate = {}
                lstate['model'] = self.model.state_dict()
                lstate['best_loss'] = self.best_loss
                torch.save(lstate, self.save_path)
                
          
                print(f"Save a best model at {self.save_path} - Best accuracy: {self.best_loss}")

        
        
        

        self.model.train()
    
    def load_pretrained_model(self):
        pretrained_path = self.config['model']['pretrained']
        if not pretrained_path:
            print("Use default init model")
            return
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=self.config['device'])
            model_state_dict = state_dict['model']
            self.model.load_state_dict(state_dict=model_state_dict)
            print(f"Load pretrained model from {pretrained_path}")
            self.best_loss = state_dict['best_loss']
            print(f"Model best_loss: {self.best_loss}")
        else:
            print(f"No pretrained model found at {pretrained_path}")

    def get_index_save_folder(self, path):
        try:
            return max([int(i) for i in os.listdir(path) if i.isdigit()], default=0) + 1
        except:
            return 0
