import os 
from os.path import join
from os import listdir


root_folder = "/work/21013187/phuoc/Image_Captionning_Transformer/data/ted2_doiten_split_manual_splited/val"
img_folder = join(root_folder,"images") 
txt_folder = join(root_folder,"labels")
count = {
	'bon': 0,
 'nam' : 0 
}
for img_name in os.listdir(img_folder):
    id = os.path.splitext(img_name)[0]
    txt_name = f"{id}.txt"
    txt_path = join(txt_folder,txt_name)
    with open(txt_path,'r') as f:
        data = f.readline().strip()
    if len(data) == 8:
        phase = 'bon' 
    elif len(data)== 9:
        phase = 'nam'
    else:
        continue
    count[phase] += 1


for key, val in count.items():
    print(f"{key}: {val}")