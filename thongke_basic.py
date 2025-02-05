import os

root_folder = r"/work/21013187/phuoc/Image_Captionning_Transformer/results/test_reuslt"

from collections import defaultdict
count_map = defaultdict(int)
for folder in os.listdir(root_folder):
    count_map[folder] = len(os.listdir(os.path.join(root_folder,folder)))

for key,value in count_map.items():
    print(f"{key} : {value}")
