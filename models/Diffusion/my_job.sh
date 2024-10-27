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
python 202410_Diffusion.py
# python 20240730_Diffusion.py
# python 20240728_Diffusion.py
# python 2024.py