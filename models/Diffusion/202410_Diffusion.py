import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
import torchmetrics

import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from typing import List
from io import BytesIO
from PIL import Image

import wandb
from tqdm import tqdm
from datetime import datetime
import argparse

# Additional imports for enhancements
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# Updated import for torchmetrics to avoid deprecation warnings
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Argument parsing
parser = argparse.ArgumentParser(description="Enhanced EEG to fMRI 2D U-Net Training Script")
parser.add_argument('--dataset_name', type=str, default="01", help="Dataset identifier")
parser.add_argument('--data_root', type=str, default="/home/aca10131kr/gca50041/quan/Datasets/EEG2fMRI/h5_data/NODDI", help="Path to the dataset directory in h5 format")
parser.add_argument('--work_dir', type=str, default="/home/aca10131kr/scratch_eeg-to-fmri", help="Path to save experiments")

parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for optimizer")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer")
parser.add_argument('--alpha', type=float, default=0.7, help="Weight for SSIM loss")
parser.add_argument('--beta', type=float, default=0.3, help="Weight for MSE loss")
parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping")

args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2D U-Net Model Definition with Residual Connections and SE Blocks
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class UNet2D(nn.Module):
    def __init__(self, input_nc=10, output_nc=30, base_nf=64):
        super(UNet2D, self).__init__()
        self.enc1 = nn.Sequential(
            ResidualConvBlock(input_nc, base_nf),
            SEBlock(base_nf)
        )
        self.enc2 = nn.Sequential(
            ResidualConvBlock(base_nf, base_nf * 2),
            SEBlock(base_nf * 2)
        )
        self.enc3 = nn.Sequential(
            ResidualConvBlock(base_nf * 2, base_nf * 4),
            SEBlock(base_nf * 4)
        )
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            ResidualConvBlock(base_nf * 4 + base_nf * 2, base_nf * 2),
            SEBlock(base_nf * 2)
        )
        self.dec2 = nn.Sequential(
            ResidualConvBlock(base_nf * 2 + base_nf, base_nf),
            SEBlock(base_nf)
        )
        self.final_conv = nn.Conv2d(base_nf, output_nc, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d3 = self.up(e3)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.final_conv(d2)
        return self.sigmoid(out)

# Combined SSIM and MSE Loss Function using torchmetrics
class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(SSIM_MSE_Loss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, img1, img2):
        ssim_value = self.ssim(img1, img2)
        ssim_loss = 1 - ssim_value
        mse_loss = self.mse(img1, img2)
        return self.alpha * ssim_loss + self.beta * mse_loss

# Custom Dataset class for EEG-fMRI data with Augmentation
class EEGfMRIDataset2D(Dataset):
    def __init__(self, eeg_data, fmri_data, eeg_transform=None, fmri_transform=None):
        self.eeg_data = eeg_data
        self.fmri_data = fmri_data
        self.eeg_transform = eeg_transform
        self.fmri_transform = fmri_transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        fmri = self.fmri_data[idx]
        if self.eeg_transform:
            eeg = self.eeg_transform(eeg)
        if self.fmri_transform:
            fmri = self.fmri_transform(fmri)
        return eeg, fmri

# W&B Initialization
wandb.login()

# Function to pad data
def pad_data(data, target_size):
    # data: Tensor of shape [N, C, H, W]
    _, _, H, W = data.shape
    pad_h = target_size[0] - H
    pad_w = target_size[1] - W
    padding = [
        pad_w // 2, pad_w - pad_w // 2,  # Left, Right padding
        pad_h // 2, pad_h - pad_h // 2   # Top, Bottom padding
    ]
    data_padded = F.pad(data, padding, mode='constant', value=0)
    return data_padded

# Normalize data per channel
def normalize_per_channel(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    # data: Tensor of shape [N, C, H, W]
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    data_normalized = (data - mean) / (std + 1e-8)
    return data_normalized

# Print data statistics
def print_data_stats(name, data):
    print(f"{name} - Shape: {data.shape}, Min: {data.min().item():.4f}, Max: {data.max().item():.4f}, Mean: {data.mean().item():.4f}")

# Function to plot and compare labels and outputs
def plot_comparison_2d(labels, outputs, num_channels=3):
    num_plots = min(num_channels, labels.shape[1], outputs.shape[1])
    fig, axes = plt.subplots(num_plots, 3, figsize=(15, 5 * num_plots))

    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        label_img = labels[i].cpu().numpy()
        output_img = outputs[i].cpu().numpy()
        diff_img = np.abs(label_img - output_img)

        # Plot Ground Truth
        axes[i][0].imshow(label_img, cmap='gray')
        axes[i][0].set_title(f'Ground Truth - Channel {i+1}')
        axes[i][0].axis('off')

        # Plot Generated Output
        axes[i][1].imshow(output_img, cmap='gray')
        axes[i][1].set_title(f'Generated Output - Channel {i+1}')
        axes[i][1].axis('off')

        # Plot Difference
        axes[i][2].imshow(diff_img, cmap='hot')
        axes[i][2].set_title(f'Difference - Channel {i+1}')
        axes[i][2].axis('off')

    plt.tight_layout()
    buff = BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    image = Image.open(buff)

    plt.close()

    return image

# PSNR Calculation function
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    return psnr.item()

# Load data from a list of individuals
def load_h5_from_list(data_root: str, individual_list: List):
    eeg_data = []
    fmri_data = []

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Loading Individual {individual_name}')
        with h5py.File(Path(data_root) / f'{individual_name}.h5', 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            eeg_data.append(eeg_indv)
            fmri_data.append(fmri_indv)

    eeg_data = np.concatenate(eeg_data, axis=0)
    fmri_data = np.concatenate(fmri_data, axis=0)

    return eeg_data, fmri_data

# Main function
def main():
    # Define data augmentation transforms with limited rotation
    eeg_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),  # Reduced degrees to prevent distortion
    ])

    fmri_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
    ])

    # Use the same train and test individuals as originally specified
    # Train/test lists
    test_list = ['43', '44']
    train_list = [Path(indv).stem for indv in os.listdir(args.data_root) if Path(indv).stem not in test_list]

    print("Training Individuals:", sorted(train_list))
    print("Testing Individuals:", test_list)

    # Load training data
    print(f'Loading training data...')
    eeg_train, fmri_train = load_h5_from_list(args.data_root, individual_list=train_list)

    # Load testing data
    print(f'Loading testing data...')
    eeg_test, fmri_test = load_h5_from_list(args.data_root, individual_list=test_list)

    # Convert to tensors and permute dimensions
    eeg_train = torch.tensor(eeg_train, dtype=torch.float32).permute(0, 3, 1, 2)  # [N, C, H, W]
    eeg_test = torch.tensor(eeg_test, dtype=torch.float32).permute(0, 3, 1, 2)
    fmri_train = torch.tensor(fmri_train, dtype=torch.float32).permute(0, 3, 1, 2)
    fmri_test = torch.tensor(fmri_test, dtype=torch.float32).permute(0, 3, 1, 2)

    # Print data statistics before normalization
    print_data_stats("EEG Train Before Normalization", eeg_train)
    print_data_stats("fMRI Train Before Normalization", fmri_train)

    # Compute mean and std per channel for EEG and fMRI
    eeg_mean = eeg_train.mean(dim=[0, 2, 3])
    eeg_std = eeg_train.std(dim=[0, 2, 3])
    fmri_mean = fmri_train.mean(dim=[0, 2, 3])
    fmri_std = fmri_train.std(dim=[0, 2, 3])

    # Normalize per channel using training data statistics
    eeg_train = normalize_per_channel(eeg_train, eeg_mean, eeg_std)
    eeg_test = normalize_per_channel(eeg_test, eeg_mean, eeg_std)
    fmri_train = normalize_per_channel(fmri_train, fmri_mean, fmri_std)
    fmri_test = normalize_per_channel(fmri_test, fmri_mean, fmri_std)

    # Print data statistics after normalization
    print_data_stats("EEG Train After Normalization", eeg_train)
    print_data_stats("fMRI Train After Normalization", fmri_train)

    # Pad EEG data to match fMRI spatial dimensions
    target_size = fmri_train.shape[2:]  # Should be (64, 64)
    eeg_train = pad_data(eeg_train, target_size)
    eeg_test = pad_data(eeg_test, target_size)

    # Print data statistics after padding
    print_data_stats("EEG Train After Padding", eeg_train)

    # Ensure EEG and fMRI data have the same spatial dimensions
    assert eeg_train.shape[2:] == fmri_train.shape[2:], "EEG and fMRI data must have the same spatial dimensions."

    print("EEG Train Shape:", eeg_train.shape)
    print("fMRI Train Shape:", fmri_train.shape)
    print("EEG Test Shape:", eeg_test.shape)
    print("fMRI Test Shape:", fmri_test.shape)

    # Create PyTorch datasets and loaders with augmentation for training
    train_dataset = EEGfMRIDataset2D(eeg_data=eeg_train, fmri_data=fmri_train,
                                    eeg_transform=eeg_transform, fmri_transform=fmri_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Corrected dataset initialization: replaced 'transform=None' with 'eeg_transform=None'
    test_dataset = EEGfMRIDataset2D(eeg_data=eeg_test, fmri_data=fmri_test,
                                   eeg_transform=None, fmri_transform=None)  # No augmentation for test
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize a new W&B run with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f"dataset_{args.dataset_name}_unet2d_run_{timestamp}"

    exp_dir = os.path.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    run = wandb.init(project="eeg_fmri_project", name=exp_name, config=vars(args))

    # Initialize the 2D U-Net model
    model = UNet2D(input_nc=eeg_train.shape[1], output_nc=fmri_train.shape[1]).to(device)

    # Define the loss function
    criterion = SSIM_MSE_Loss(alpha=args.alpha, beta=args.beta).to(device)

    # Optimizer and scheduler (removed verbose=True to fix warning)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Early Stopping variables
    best_ssim = -1.0
    best_psnr = -1.0
    best_save_path = None
    epochs_no_improve = 0

    # Training loop with timing and SSIM tracking
    total_training_time = 0.0

    pbar = tqdm(range(args.num_epochs), leave=True)

    # Initialize SSIM metric
    calculate_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in pbar:
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        # Scheduler step based on validation SSIM
        # Will be updated after evaluation

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

                with autocast():
                    outputs = model(inputs)

                ssim_score += calculate_ssim(outputs, labels).item()
                psnr_score += calculate_psnr(outputs, labels)

        ssim_score /= len(test_loader)
        psnr_score /= len(test_loader)

        # Scheduler step
        scheduler.step(ssim_score)

        # Track the latest lr
        last_lr = optimizer.param_groups[0]["lr"]

        # Visualize the last batch
        labels_np = labels.cpu()
        outputs_np = outputs.cpu()
        image = plot_comparison_2d(labels_np, outputs_np, num_channels=3)  # Visualize first 3 channels

        run.log({
            "lr": last_lr,
            "loss": epoch_loss,
            "ssim": ssim_score,
            "psnr": psnr_score,
            "image": wandb.Image(image),
        })

        # Save the best model based on SSIM
        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_psnr = psnr_score
            epochs_no_improve = 0
            if best_save_path is not None and os.path.exists(best_save_path):
                os.remove(best_save_path)
            best_save_path = os.path.join(exp_dir, f"epoch{epoch+1}_ssim_{best_ssim:.4f}_psnr_{best_psnr:.2f}.pth")
            torch.save(model.state_dict(), best_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        pbar.set_description(f'Epoch {epoch+1}/{args.num_epochs} - SSIM: {ssim_score:.4f} / PSNR: {psnr_score:.2f} / Loss: {epoch_loss:.4f} / Best SSIM: {best_ssim:.4f}')

    # Save the final model
    final_model_path = os.path.join(exp_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)

    # Log total training time
    run.log({"total_training_time_sec": total_training_time})

    # Get W&B run URL
    run_url = run.get_url()
    print(f"W&B Run URL: {run_url}")

    wandb.finish()

if __name__ == '__main__':
    main()
