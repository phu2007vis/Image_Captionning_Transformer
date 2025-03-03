#!/bin/bash
#SBATCH --job-name=Image_captioning
#SBATCH --partition=dgx-small
#SBATCH --time=69:00:00
#SBATCH --account=ddt_acc23
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err

squeue --me
cd /work/21013187/phuoc/Image_Captionning_Transformer
module load python
module load cuda
nvidia-smi
python /work/21013187/phuoc/Image_Captionning_Transformer/src_ddp_trainning/train/acceslator.py