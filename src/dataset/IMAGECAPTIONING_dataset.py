
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from torchvision import transforms 
from utils.text_cleaner import tien_xu_li
from PIL import Image
from transformers import AutoTokenizer

def collate_fn(batch):
    captions = [example['text'] for example in batch]
    images = [example['image'] for example in batch]
    encodings = [example['encoding'] for example in batch]
    attention_mask = [example['attention_mask'] for example in batch]
    
    return {'images': torch.cat(images,dim = 0),
        'captions':  captions,
        'encodings': torch.cat(encodings,dim = 0),
        'attention_mask': attention_mask
           }

class IMAGECAPTIONING(Dataset):
    def __init__(self, config ):
        self.config = config
        self.img_folder = self.config['img_folder']
        self.json_file = self.config['json_file']
        self.max_len = self.config['max_len']
        self.collate_fn = collate_fn
        
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        
        self.images = pd.DataFrame(dict(data)['images'])
        self.annotation = dict(data)['annotations']
    
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
       
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ]
        )
       
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()
   
        annotation = self.annotation[index]
        
        image_id  = annotation['image_id']
        caption  = annotation['caption']
        
        caption = tien_xu_li(caption)
        image_name = str(self.images[self.images.id == image_id].filename.values[0])
        image_path = os.path.join(self.img_folder,image_name)
        pil_image = Image.open(image_path)
        tensor_image = self.transform(pil_image).unsqueeze(0)
        encoding = self.tokenizer.encode_plus(caption,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        return {
            'text': caption,
            'encoding':encoding['input_ids'],
            'image': tensor_image,
            "attention_mask" : encoding['attention_mask']
        }




if __name__ == "__main__":
    
    import os
    import sys
    folfder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(folfder_dir)
    from utils import load_config
    config_path = "/work/21013187/phuoc/Image_Captionning_Transformer/src/configs/image_captioning.yaml"
    data_path = "/home/21013187/.cache/kagglehub/datasets/phuocnguyenxuan/image-captioning/versions/1"
    config = load_config(config_path)
    dataset_config = config['dataset']['train']
    dataset = IMAGECAPTIONING(dataset_config)
    import pdb;pdb.set_trace()