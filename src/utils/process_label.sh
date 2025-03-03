#!/bin/bash
#SBATCH --job-name=Image_captioning
#SBATCH --partition=dgx-small
#SBATCH --time=23:00:00
#SBATCH --account=ddt_acc23

#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err

squeue --me
cd /work/21013187/phuoc/Image_Captionning_Transformer
module load python
python /work/21013187/phuoc/TextRecognitionDataGenerator/trdg_phuoc/generators/from_strings.py --output_dir=/work/21013187/phuoc/Image_Captionning_Transformer/data2/out --number_each=150 --aug
# Xóa thư mục images nếu tồn tại
if [ -d "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/images" ]; then
    rm -r "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/images"
fi

# Xóa thư mục labels nếu tồn tại
if [ -d "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/labels" ]; then
    rm -r "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/labels"
fi

# Xóa tệp labels.csv nếu tồn tại
if [ -e "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/labels.csv" ]; then
    rm "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/labels.csv"
fi

cp -r /work/21013187/phuoc/Image_Captionning_Transformer/data2/out /work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val/images
python /work/21013187/phuoc/Image_Captionning_Transformer/src/utils/generate_txt_label.py --root_folder=/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val
python /work/21013187/phuoc/Image_Captionning_Transformer/src/utils/generate_csv_label.py --root_folder=/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/val


python /work/21013187/phuoc/TextRecognitionDataGenerator/trdg_phuoc/generators/from_strings.py --output_dir=/work/21013187/phuoc/Image_Captionning_Transformer/data2/out --number_each=1000 
# Xóa thư mục images nếu tồn tại
if [ -d "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/images" ]; then
    rm -r "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/images"
fi

# Xóa thư mục labels nếu tồn tại
if [ -d "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/labels" ]; then
    rm -r "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/labels"
fi

# Xóa tệp labels.csv nếu tồn tại
if [ -e "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/labels.csv" ]; then
    rm "/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/labels.csv"
fi


cp -r /work/21013187/phuoc/Image_Captionning_Transformer/data2/out /work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train/images
python /work/21013187/phuoc/Image_Captionning_Transformer/src/utils/generate_txt_label.py --root_folder=/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train
python /work/21013187/phuoc/Image_Captionning_Transformer/src/utils/generate_csv_label.py --root_folder=/work/21013187/phuoc/Image_Captionning_Transformer/data2/systhetic_data_v1/train
