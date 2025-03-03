
import os
import sys
import torch.nn as nn
import evaluate
import numpy as np
import torch 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models import get_model
from dataset_folder import get_dataloader
from utils import load_config

from transformers import (
	EvalPrediction,
	Trainer,
	TrainingArguments,
)
def collate(batch):

	filenames = []
	img = []
	target_weights = []
	tgt_input = []
	max_label_len = max(len(sample["word"]) for sample in batch)

	for sample in batch:
	
		img.append(sample["img"].unsqueeze(0))
		
		filenames.append(sample["img_path"])
		label = sample["word"]
		label_len = len(label)

		tgt = np.concatenate(
			(label, np.zeros(max_label_len - label_len, dtype=np.int32))
		)
		tgt_input.append(tgt)

		one_mask_len = label_len - 1

		target_weights.append(
			np.concatenate(
				(
					np.ones(one_mask_len, dtype=np.float32),
					np.zeros(max_label_len - one_mask_len, dtype=np.float32),
				)
			)
		)

	img = torch.cat(img,dim = 0)

	tgt_input = np.array(tgt_input, dtype=np.int64).T
	tgt_output = np.roll(tgt_input, -1, 0).T
	tgt_output[:, -1] = 0



	tgt_padding_mask = np.array(target_weights) == 0

	rs = {
		"img": img,
		"tgt_input": torch.LongTensor(tgt_input),
		"tgt_output": torch.LongTensor(tgt_output),
		"tgt_padding_mask": torch.BoolTensor(tgt_padding_mask),
		"filenames": filenames,
	}


def setup_dataloader(config):
	dataset_config = config.get('dataset')
	dataloader_register = {}
	for phase in dataset_config.keys():
		assert phase in ['train', 'val'], f"Unrecognized phase {phase}"
		phase_config = dataset_config[phase]
		dataloader_register[phase] = get_dataloader(phase_config)
	return dataloader_register['train'].dataset, dataloader_register['val'].dataset

training_args = TrainingArguments(
    "basic-trainer2",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=15,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

metric = evaluate.load("/work/21013187/phuoc/Image_Captionning_Transformer/src_ddp_trainning/evaluate/metrics/accuracy/accuracy.py")
def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		result = metric.compute(predictions=preds, references=p.label_ids)
		if len(result) > 1:
			result["combined_score"] = np.mean(list(result.values())).item()
		return result

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        
        outputs = model(**inputs)
        target = inputs["tgt_output"].view(-1)
        loss = nn.CrossEntropyLoss(outputs, target)
        return (loss, outputs) if return_outputs else loss
    

def main():
    
	config_path = os.environ['config_path']
	config = load_config(config_path)
 
	train_dataset,test_dataset = setup_dataloader(config)
	model = get_model(config)
 

	trainer = MyTrainer(
		model,
		training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		data_collator=test_dataset.collate_fn,
  		compute_metrics=compute_metrics,
		)
	trainer.train()

	

if __name__ == "__main__":
    
	os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
	os.environ['config_path'] = '/work/21013187/phuoc/Image_Captionning_Transformer/src_ddp_trainning/configs/plate_ocr_hw.yaml'
	os.environ['WANDB_DISABLED'] = 'true'
	main()