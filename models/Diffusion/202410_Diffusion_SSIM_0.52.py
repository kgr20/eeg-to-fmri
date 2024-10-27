import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Added for interpolate

from torch.utils.data import DataLoader, Dataset
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
parser = argparse.ArgumentParser(description="EEG to fMRI U-Net Training Script")
parser.add_argument('--dataset_name', type=str, default="01", help="Dataset identifier")
parser.add_argument('--data_root', type=str, default="/home/aca10131kr/gca50041/quan/Datasets/EEG2fMRI/h5_data/NODDI", help="Path to the dataset directory in h5 format")
parser.add_argument('--work_dir', type=str, default="/home/aca10131kr/scratch_eeg-to-fmri", help="Path to save experiments")

parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for optimizer")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer")

args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, input_nc=10, output_nc=30, num_filters=64):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(input_nc, num_filters)
        self.enc2 = self.conv_block(num_filters, num_filters * 2)
        self.enc3 = self.conv_block(num_filters * 2, num_filters * 4)
        self.enc4 = self.conv_block(num_filters * 4, num_filters * 8)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)  # Use ceil_mode to handle odd dimensions
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.conv_block(num_filters * 8 + num_filters * 4, num_filters * 4)
        self.dec3 = self.conv_block(num_filters * 4 + num_filters * 2, num_filters * 2)
        self.dec2 = self.conv_block(num_filters * 2 + num_filters, num_filters)
        self.dec1 = self.conv_block(num_filters, num_filters)
        self.final_conv = nn.Conv2d(num_filters, output_nc, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return block

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d4 = F.interpolate(e4, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(d4, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=x.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return self.sigmoid(out)

# Combined SSIM and MSE Loss Function
class SSIM_MSE_Loss(nn.Module):
    def __init__(self):
        super(SSIM_MSE_Loss, self).__init__()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        ssim_loss = 1 - self.ssim(img1, img2)
        mse_loss = self.mse(img1, img2)
        return 0.5 * ssim_loss + 0.5 * mse_loss

# Custom Dataset class for EEG-fMRI data
class EEGfMRIDataset(Dataset):
    def __init__(self, eeg_data, fmri_data):
        self.eeg_data = eeg_data
        self.fmri_data = fmri_data

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        fmri = self.fmri_data[idx]
        return eeg, fmri

# W&B Initialization (if you use W&B for logging)
wandb.login()

# Function to plot and compare labels and outputs
def plot_comparison(labels, outputs, slice_idx=15):
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
    image = Image.open(buff)

    plt.close()

    return image

# Normalize data to a specific range
def normalize_data(data: np.ndarray, scale_range: Tuple=None):
    """Normalize data to range [a, b]"""
    new_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    if scale_range is not None:
        a, b = scale_range
        assert a <= b, f'Invalid range: {scale_range}'
        new_data = (b - a) * new_data + a
    return new_data

# PSNR Calculation function
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
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
    # Use the same train and test individuals as originally specified
    # Train/test Quan-Kris
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

    # Transpose data to match PyTorch's [N, C, H, W] format
    eeg_train = eeg_train.transpose(0, 3, 1, 2)
    fmri_train = fmri_train.transpose(0, 3, 1, 2)
    eeg_test = eeg_test.transpose(0, 3, 1, 2)
    fmri_test = fmri_test.transpose(0, 3, 1, 2)

    # Normalize the data to range [0, 1]
    eeg_train = normalize_data(eeg_train, scale_range=(0, 1))
    fmri_train = normalize_data(fmri_train, scale_range=(0, 1))
    eeg_test = normalize_data(eeg_test, scale_range=(0, 1))
    fmri_test = normalize_data(fmri_test, scale_range=(0, 1))

    print("EEG Train Shape:", eeg_train.shape)
    print("fMRI Train Shape:", fmri_train.shape)
    print("EEG Test Shape:", eeg_test.shape)
    print("fMRI Test Shape:", fmri_test.shape)

    # Convert to tensors
    eeg_train = torch.tensor(eeg_train, dtype=torch.float32)
    fmri_train = torch.tensor(fmri_train, dtype=torch.float32)
    eeg_test = torch.tensor(eeg_test, dtype=torch.float32)
    fmri_test = torch.tensor(fmri_test, dtype=torch.float32)

    # Resize EEG data to match fMRI spatial dimensions
    target_size = (64, 64)  # Assuming fMRI spatial dimensions are 64x64
    eeg_train = F.interpolate(eeg_train, size=target_size, mode='bilinear', align_corners=True)
    eeg_test = F.interpolate(eeg_test, size=target_size, mode='bilinear', align_corners=True)

    # Create PyTorch datasets and loaders
    train_dataset = EEGfMRIDataset(eeg_data=eeg_train, fmri_data=fmri_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = EEGfMRIDataset(eeg_data=eeg_test, fmri_data=fmri_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize a new W&B run with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f"dataset_{args.dataset_name}_unet_run_{timestamp}"

    exp_dir = os.path.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    run = wandb.init(project="eeg_fmri_project", name=exp_name)

    # Initialize the U-Net model
    model = UNet(input_nc=10, output_nc=30).to(device)

    # Define the loss function
    criterion = SSIM_MSE_Loss().to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop with timing and SSIM tracking
    total_training_time = 0.0
    best_ssim = -1.0
    best_psnr = -1.0
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

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch + epoch_loss)

        # Track the latest lr
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

                outputs = model(inputs)
                ssim_score += calculate_ssim(outputs, labels).item()
                psnr_score += calculate_psnr(outputs, labels)

        ssim_score /= len(test_loader)
        psnr_score /= len(test_loader)

        # Visualize the last batch
        labels_np = labels.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        image = plot_comparison(labels_np, outputs_np, slice_idx=15)

        run.log({
            "lr": last_lr,
            "loss": epoch_loss,
            "ssim": ssim_score,
            "psnr": psnr_score,
            "image": wandb.Image(image),
        })

        # Save the best model
        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_psnr = psnr_score
            if best_save_path is not None:
                os.remove(best_save_path)
            best_save_path = os.path.join(exp_dir, f"epoch{epoch}_ssim_{best_ssim:.4f}_psnr_{best_psnr:.2f}.pth")
            torch.save(model.state_dict(), best_save_path)

        pbar.set_description(f'Epoch {epoch+1}/{args.num_epochs} - SSIM: {ssim_score:.3f} / PSNR: {psnr_score:.3f} / Loss: {epoch_loss:.3f} / Best SSIM: {best_ssim:.3f}')

    # Save the final model
    final_model_path = os.path.join(exp_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)

    wandb.finish()

if __name__ == '__main__':
    main()
