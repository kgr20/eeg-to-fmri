import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
parser = argparse.ArgumentParser(description="EEG to fMRI Autoencoder Training Script")
parser.add_argument('--dataset_name', type=str, default="01", help="Dataset identifier")
parser.add_argument('--data_root', type=str, default="/home/aca10131kr/gca50041/quan/Datasets/EEG2fMRI/h5_data/NODDI", help="Path to the dataset directory in h5 format")
parser.add_argument('--work_dir', type=str, default="/home/aca10131kr/scratch_eeg-to-fmri", help="Path to save experiments")

parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for optimizer")

args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Autoencoder model
class DeeperWiderConvAutoencoder2D(nn.Module):
    def __init__(self, input_nc=10, output_nc=30, output_size=64):
        super(DeeperWiderConvAutoencoder2D, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=True),
            nn.Conv2d(output_nc, output_nc, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# SSIM Loss Function
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)

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
    eeg_data = None
    fmri_data = None

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Loading Individual {individual_name}')
        with h5py.File(Path(data_root) / f'{individual_name}.h5', 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            eeg_data = eeg_indv if eeg_data is None else np.concatenate([eeg_data, eeg_indv], axis=0)
            fmri_data = fmri_indv if fmri_data is None else np.concatenate([fmri_data, fmri_indv], axis=0)

    return eeg_data, fmri_data

# Main function
def main():
    # Use the same train and test individuals as originally specified
    # Uncomment and adjust as per your original script
    # Train/test from David
    # train_list = ['32', '35', '36', '37', '38', '39', '40', '42']
    # test_list = ['43', '44']

    # Train/test Quan-Kris (as per your original script)
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

    # Create PyTorch datasets and loaders
    train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train, dtype=torch.float32),
                                   fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test, dtype=torch.float32),
                                  fmri_data=torch.tensor(fmri_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize a new W&B run with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f"dataset_{args.dataset_name}_full_data_run_{timestamp}"

    exp_dir = os.path.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    run = wandb.init(project="eeg_fmri_project", name=exp_name)

    # Initialize the model
    model = DeeperWiderConvAutoencoder2D(input_nc=10, output_nc=30, output_size=64).to(device)

    # Define the loss function
    criterion = SSIMLoss().to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
        scheduler.step(epoch_loss)

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
        image = plot_comparison(labels_np, outputs_np, slice_idx=16)

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

        pbar.set_description(f'SSIM: {ssim_score:.3f} / PSNR: {psnr_score:.3f} / Loss: {epoch_loss:.3f} / Best SSIM: {best_ssim:.3f}')

    # Save the final model
    final_model_path = os.path.join(exp_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)

    wandb.finish()

if __name__ == '__main__':
    main()
