import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.nn import functional as F
import torch.nn as nn
import os
from accelerate.utils import tqdm
import sys
import math
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models import get_model
from dataset_folder import get_dataloader
from utils import load_config

def evaluate(accelerator, model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_sen_cor = 0 
    total_sen_samples = 0

    with torch.no_grad():
        with tqdm(test_loader, total=len(test_loader), unit='batch',
                disable=not accelerator.is_main_process) as tepoch:
            for data in tepoch:
                output = model(**data)
                _, output = output.max(-1)
                label = data['tgt_output']
    
                sen_acc = 0
                sen_total = 0
                for l, o in zip(label, output):
                    valid_mask = (l != 0)
                    valid_predicted = o[valid_mask]
                    valid_labels = l[valid_mask]
                    if torch.all(valid_predicted == valid_labels):
                        sen_acc += 1
                    sen_total += 1
                sen_acc = torch.tensor(sen_acc, device=label.device)
                sen_total = torch.tensor(sen_total, device=label.device)
            
                gathered_correct_sen = accelerator.gather_for_metrics(sen_acc)
                gathered_total_sen_total = accelerator.gather_for_metrics(sen_total)
    
                total_sen_cor += gathered_correct_sen.sum().item()
                total_sen_samples += gathered_total_sen_total.sum().item()
  
                label = label.view(-1)
                output = output.view(-1)
                valid_mask_all = (label != 0)
                valid_predicted_all = output[valid_mask_all]
                valid_labels_all = label[valid_mask_all]
    
                correct = (valid_predicted_all == valid_labels_all).sum()
                batch_size = torch.tensor(valid_labels_all.size(0), device=label.device)
                
                gathered_correct = accelerator.gather_for_metrics(correct)
                gathered_total = accelerator.gather_for_metrics(batch_size)
                
                total_correct += gathered_correct.sum().item()
                total_samples += gathered_total.sum().item()

    accuracy_char = total_correct / total_samples if total_samples > 0 else 0
    accuracy_sen = total_sen_cor / total_sen_samples if total_sen_samples > 0 else 0
    
    accelerator.print(f'Batch Evaluation - Accuracy (char level): {round(accuracy_char*100,2)}%')
    accelerator.print(f'Batch Evaluation - Accuracy (sentence level): {round(accuracy_sen*100,2)}%')
    return accuracy_sen, accuracy_char

def setup_dataloader(config):
    dataset_config = config.get('dataset')
    dataloader_register = {}
    for phase in dataset_config.keys():
        assert phase in ['train', 'val'], f"Unrecognized phase {phase}"
        phase_config = dataset_config[phase]
        dataloader_register[phase] = get_dataloader(phase_config)
    return dataloader_register['train'], dataloader_register['val']

def load_weights(model, pretrained_path, accelerator):
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    accelerator.print(f"Load pretrained model from {pretrained_path}")
    accelerator.print(f"Model best_loss: {state_dict['best_loss']}")
    return model

def train_ddp_accelerate():
    config_path = os.environ['config_path']
    config = load_config(config_path)
 
    max_epochs = config['train']['epochs']
    mode = config['mode']
    eval_interval = config.get('evaluate', {}).get('frequency', 50)
   
    assert mode in ['train', 'val', 'visualize'], 'mode must be train, val, or visualize'
    
    accelerator = Accelerator(
        log_with="all",
        project_dir="result_ddp",
        gradient_accumulation_steps=config['num_mini_batches']
    )
    accelerator.print(f"eval_interval: {eval_interval}")
    accelerator.print(f"Mode: {mode}")
    accelerator.print(f"num_mini_batches : {config['num_mini_batches']}")
    
    model = get_model(config)
    pretrained_path = config['model']['pretrained']
    model = load_weights(model, pretrained_path, accelerator)

    if mode == 'train':
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.22, ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=7e-5)

        train_loader, test_loader = setup_dataloader(config=config)
        train_loader, test_loader, model, optimizer = accelerator.prepare(
            train_loader, test_loader, model, optimizer
        )
        
        # Training loop with batch-level evaluation
        global_batch_idx = 0
        best_accuracy_char = 0
        
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            with tqdm(train_loader, total=len(train_loader), unit='batch',
                     disable=not accelerator.is_main_process) as tepoch:
                for batch_idx, data in enumerate(tepoch):
                    with accelerator.accumulate(model):    
                        output = model(**data)
                        output = model.module.proprocessing_output(output)
                        label = model.module.get_label(data)
                        
                        loss = loss_fn(output, label)
                        total_loss += loss.item()
                        accelerator.backward(loss)
                        # if accelerator.sync_gradients:
                        #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad() 
                        tepoch.set_postfix(loss=loss.item())
                    
                    global_batch_idx += 1
                    
                    # Evaluate every eval_interval batches
                    if global_batch_idx % eval_interval == 0:
                        # Synchronize all processes before evaluation
                        accelerator.wait_for_everyone()
                        
                        accelerator.print(f'\nEvaluation after {global_batch_idx} batches (Epoch {epoch+1})')
                        accuracy_sen, accuracy_char = evaluate(accelerator, model, test_loader)
                        
                        # Save checkpoint if this is the best model
                        save_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/basic-trainer2"
                        os.makedirs(save_folder, exist_ok=True)
                        
                        if accuracy_char > best_accuracy_char:
                            best_accuracy_char = accuracy_char
                            save_path = os.path.join(save_folder, 
                                                  f"best_batch_{global_batch_idx}_e{epoch+1}_{round(accuracy_char,3)}_{round(accuracy_sen,3)}.pth")
                            
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.save({
                                "model": unwrapped_model.state_dict(),
                                'best_loss': round(accuracy_char,3)
                            }, save_path)
                            
                            accelerator.print(f'New best model saved to {save_path}')
                        
                        # Synchronize again after saving
                        accelerator.wait_for_everyone()
                        
                        # Return to training mode and ensure gradients are enabled
                        model.train()
            
            avg_loss = total_loss / len(train_loader)
            accelerator.print(f'Epoch {epoch+1}/{max_epochs} - Average Training Loss: {avg_loss:.4f}')

        # Final evaluation and save
        accelerator.wait_for_everyone()
        accelerator.print('\nFinal Evaluation:')
        accuracy_sen, accuracy_char = evaluate(accelerator, model, test_loader)
        
        save_path = os.path.join(save_folder, 
                               f"final_batch_{global_batch_idx}_e{max_epochs}_{round(accuracy_char,3)}_{round(accuracy_sen,3)}.pth")
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save({
            "model": unwrapped_model.state_dict(),
            "optimizer": optimizer.optimizer.state_dict(),
            'best_loss': round(accuracy_char,3)
        }, save_path)
        accelerator.print(f'Final model saved to {save_path}')

    elif mode == 'val':
        train_loader, test_loader = setup_dataloader(config=config)
        train_loader, test_loader, model = accelerator.prepare(
            train_loader, test_loader, model
        )
        accuracy_sen, accuracy_char = evaluate(accelerator, model, test_loader)
    
    elif mode == 'visualize':
        if accelerator.is_main_process:
            train_loader, _ = setup_dataloader(config=config)
            num_images = config.get('visualize', {}).get('num_images', 64)
            output_file = config.get('visualize', {}).get('output_file', 'dataset.png')
            grid_size = math.ceil(math.sqrt(num_images))
            
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(24, 24))
            plt.subplots_adjust(wspace=0.3, hspace=0.5)
            
            n = 0
            for batch in train_loader:
                batch_size = batch["img"].shape[0]
                for i in range(min(batch_size, num_images - n)):
                    img = batch["img"][i].cpu().numpy().transpose(1, 2, 0)
                    sent = train_loader.dataset.vocab.decode(batch["tgt_input"].T[i].cpu().tolist())
                    row, col = divmod(n, grid_size)
                    ax = axs[row, col]
                    ax.imshow(img)
                    ax.set_title(sent[:50] + '...' if len(sent) > 50 else sent, fontname="serif")
                    ax.axis("off")
                    n += 1
                    if n >= num_images:
                        break
                if n >= num_images:
                    break
 
            for i in range(n, grid_size * grid_size):
                row, col = divmod(i, grid_size)
                axs[row, col].axis("off")
            
            plt.savefig(output_file, bbox_inches="tight")
            plt.close(fig)
            print(f"Visualization saved to {output_file}")

if __name__ == '__main__':
    from accelerate import notebook_launcher
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,1"
    os.environ['config_path'] = '/work/21013187/phuoc/Image_Captionning_Transformer/src_ddp_trainning/configs/plate_ocr_hw.yaml'
    notebook_launcher(train_ddp_accelerate, args=(), num_processes=torch.cuda.device_count())