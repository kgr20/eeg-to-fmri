#!/bin/bash

# Load the bash configuration
source ~/.bashrc

# Initialize Conda and activate the environment
source /home/quan/anaconda3/etc/profile.d/conda.sh
conda activate py38

# Run the Python script
python 20241226_diffusion_v2.py