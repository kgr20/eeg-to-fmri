import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from models.networks import define_G, DeeperWiderConvAutoencoder2D
from data.datasets import EEGfMRIDataset
from models.losses import SSIMLoss
import torchmetrics

import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from typing import Tuple, List
from io import BytesIO
from PIL import Image

import wandb
from tqdm import tqdm
from datetime import datetime
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="EEG to fMRI Autoencoder Training Script")
parser.add_argument('--dataset_name', type=str, default="01", help="Dataset identifier")
parser.add_argument('--data_root', type=str, default="/groups/1/gca50041/quan/Datasets/EEG2fMRI/h5_data/NODDI", 
                    help="Path to the dataset directory in h5 format")
parser.add_argument('--work_dir', type=str, default="/scratch/1/acb11155on/WorkSpace/eeg2fmri", help="Path to save experiments")

parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for optimizer")

args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # W&B
# wandb.login()

def plot_comparison(labels, outputs, slice_idx=16):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    label_slice = labels[0, slice_idx, :, :]
    output_slice = outputs[0, slice_idx, :, :]
    diff_slice = np.abs(label_slice - output_slice)

    axes[0].imshow(label_slice, cmap='gray')
    axes[0].set_title('Ground Truth')

    axes[1].imshow(output_slice, cmap='gray')
    axes[1].set_title('Generated Output')

    axes[2].imshow(diff_slice, cmap='gray')
    axes[2].set_title('Difference')
    
    buff = BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    # Convert the BytesIO object to an Image object
    image = Image.open(buff)

    plt.close()

    return image

# Normalize data
def normalize_data(data: np.ndarray, scale_range: Tuple=None):
    """Normalize data to range [a, b]"""
    # scale to range [0, 1] first
    new_data = (data - data.min())/(data.max() - data.min())
    # scale to range [a, b]
    if scale_range is not None:
        a, b = scale_range
        assert a<=b, f'Invalid range: {scale_range}'
        new_data = (b-a)*new_data + a

    return new_data

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def load_h5_from_list(data_root: str, individual_list: List):
    eeg_data = None
    fmri_data = None

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Individual {individual_name}')
        with h5py.File(Path(data_root)/f'{individual_name}.h5', 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            eeg_data = eeg_indv if eeg_data is None else np.concatenate([eeg_data, eeg_indv], axis=0)
            fmri_data = fmri_indv if fmri_data is None else np.concatenate([fmri_data, fmri_indv], axis=0)
    
    return eeg_data, fmri_data

"""Load the data
Data is already in zero-mean and unit-std
"""
# NOTE: config list of train/test
# # Train/test from David
# train_list = ['32', '35', '36', '37', '38', '39', '40', '42']
# test_list = ['43', '44']

# Train/test Quan-Kris
test_list = ['43', '44']
train_list = [Path(indv).stem for indv in os.listdir(args.data_root) if Path(indv).stem not in test_list]

print(sorted(train_list))

# each data has: [N_sample, H, W, C]
print(f'Loading train data ...')
eeg_train, fmri_train = load_h5_from_list(args.data_root, individual_list=train_list)
print(f'Loading test data ...')
eeg_test, fmri_test = load_h5_from_list(args.data_root, individual_list=test_list)

# In PyTorch, 1 data sample is represented as [C, H, W]
eeg_train = eeg_train.transpose(0, 3, 1, 2)
fmri_train = fmri_train.transpose(0, 3, 1, 2)

eeg_test = eeg_test.transpose(0, 3, 1, 2)
fmri_test = fmri_test.transpose(0, 3, 1, 2)

# Normalize the data to range [a, b]
# NOTE: this is optional, we ONLY can choose either zero-mean unit-std scale OR [a, b] scale
eeg_train = normalize_data(eeg_train, scale_range=(0, 1))
fmri_train = normalize_data(fmri_train, scale_range=(0, 1))

eeg_test = normalize_data(eeg_test, scale_range=(0, 1))
fmri_test = normalize_data(fmri_test, scale_range=(0, 1))

print("EEG Train Shape:", eeg_train.shape)
print("fMRI Train Shape:", fmri_train.shape)
print("EEG Test Shape:", eeg_test.shape)
print("fMRI Test Shape:", fmri_test.shape)

train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train, dtype=torch.float32), 
                               fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test, dtype=torch.float32), 
                              fmri_data=torch.tensor(fmri_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize a new W&B run with the current timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
exp_name = f"dataset_{args.dataset_name}_run_{timestamp}"

exp_dir = os.path.join(args.work_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

run = wandb.init(project="eeg_fmri_project", name=exp_name)

# Initialize the model, eeg: [H, W, 10], fmri: [H, W, 30]
# model = define_G(input_nc=10, output_nc=30, ngf=64, netG='resnet_3blocks', output_size=64).to(device)
model = DeeperWiderConvAutoencoder2D(input_nc=10, output_nc=30, output_size=64).to(device)

# # Kris's original code
# model = DeeperWiderConvAutoencoder3D().to(device)

# Define the loss function
criterion = SSIMLoss().to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop with timing and SSIM tracking
total_training_time = 0.0
best_ssim = -1.0
best_psnr = -1.0
best_model_weights = None
best_save_path = None

pbar = tqdm(range(args.num_epochs), leave=True)

calculate_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

for epoch in pbar:
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # labels = labels.permute(0, 4, 1, 2, 3)[:, :, :, :, :28]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    scheduler.step(epoch_loss)

    # track the latest lr
    last_lr = optimizer.param_groups[0]["lr"]

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_training_time += epoch_duration

    # Evaluation
    model.eval()
    ssim_score = 0.0
    psnr_score = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.permute(0, 4, 1, 2, 3)[:, :, :, :, :28]

            outputs = model(inputs)
            ssim_score += calculate_ssim(outputs, labels).item()
            psnr_score += calculate_psnr(outputs, labels)

    ssim_score /= len(test_loader)
    psnr_score /= len(test_loader)

    # visualize the last batch
    labels_np = labels.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    image = plot_comparison(labels_np, outputs_np, slice_idx=16)

    run.log({
        "lr": last_lr,
        "loss": epoch_loss,
        "ssim": ssim_score,
        "psnr": psnr_score,
        "image": wandb.Image(image),
    })

    if ssim_score > best_ssim:
        best_ssim = ssim_score
        best_psnr = psnr_score
        # remove the latest model
        if best_save_path is not None:
            os.remove(best_save_path)
        # Save the best model weights with SSIM and PSNR scores in the filename
        best_save_path = os.path.join(exp_dir, f"epoch{epoch}_ssim_{best_ssim:.4f}_psnr_{best_psnr:.2f}.pth")
        torch.save(model.state_dict(), best_save_path)
    
    pbar.set_description(f'SSIM: {ssim_score:.3f} / PSNR: {psnr_score:.3f} / Loss: {epoch_loss:.3f} / Best SSIM: {best_ssim:.3f}')