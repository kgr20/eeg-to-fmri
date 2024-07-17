#!/bin/bash

#$ -j y
#$ -cwd

### Load necessary modules to run the code
source ~/.bashrc
module load cuda/11.2 nccl/2.8
# module load cudnn/8.6  # Commenting this out to use the conda environment's cuDNN

### Activate the conda environment
source /home/aca10131kr/anaconda3/etc/profile.d/conda.sh
conda activate py38

### Execute your code
python train_kris.py
# python 20240716_Diffusion.py