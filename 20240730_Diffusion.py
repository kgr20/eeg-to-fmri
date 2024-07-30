import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from io import BytesIO
from PIL import Image
import wandb
from datetime import datetime
import torchmetrics
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F

# Argument parsing
parser = argparse.ArgumentParser(description="EEG to fMRI Diffusion Model Training Script")
parser.add_argument('--dataset_name', type=str, default="01", help="Dataset identifier")
parser.add_argument('--data_root', type=str, default="/home/aca10131kr/gca50041/quan/Datasets/EEG2fMRI/h5_data/NODDI", help="Path to the dataset directory in h5 format")
parser.add_argument('--work_dir', type=str, default="/home/aca10131kr/scratch_eeg-to-fmri", help="Path to save experiments")
parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for optimizer")
args = parser.parse_args()

filename = "20240730_Diffusion.py"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset class
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

# Load h5 data function
def load_h5_from_list(data_root: str, individual_list: list):
    eeg_data = None
    fmri_data = None

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Individual {individual_name}')
        with h5py.File(os.path.join(data_root, f'{individual_name}.h5'), 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            eeg_data = eeg_indv if eeg_data is None else np.concatenate([eeg_data, eeg_indv], axis=0)
            fmri_data = fmri_indv if fmri_data is None else np.concatenate([fmri_data, fmri_indv], axis=0)
    
    return eeg_data, fmri_data

# Normalize the data to range [0, 1]
def normalize_data(data: np.ndarray, scale_range: tuple = None):
    new_data = (data - data.min()) / (data.max() - data.min())
    if scale_range is not None:
        a, b = scale_range
        assert a <= b, f'Invalid range: {scale_range}'
        new_data = (b - a) * new_data + a
    return new_data

# Define Encoder, DiffusionProcess, and Decoder classes
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(256 * 68 * 3, 1024)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x

class DiffusionProcess(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionProcess, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, x, noise_level):
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        return x_noisy

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(1024, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(30, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.view(x.size(0), 256, 8, 8)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        x = x.permute(0, 2, 3, 1)
        return x

encoder = Encoder().to(device)
diffusion_process = DiffusionProcess(latent_dim=1024).to(device)
decoder = Decoder().to(device)

# Define loss function and optimizer
criterion = nn.SSIMLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(diffusion_process.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

# Functions to calculate SSIM and PSNR
ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

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

# Training function
# Training function
def train(model, train_loader, criterion, optimizer, device):
    encoder, diffusion_process, decoder = model
    encoder.train()
    diffusion_process.train()
    decoder.train()

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            latent = encoder(inputs)
            noisy_latent = diffusion_process(latent, noise_level=0.1)
            outputs = decoder(noisy_latent)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluation and logging to WandB
        encoder.eval()
        diffusion_process.eval()
        decoder.eval()
        ssim_score = 0.0
        psnr_score = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                latent = encoder(inputs)
                noisy_latent = diffusion_process(latent, noise_level=0.1)
                outputs = decoder(noisy_latent)
                ssim_score += criterion.ssim(outputs, labels).item()
                psnr_score += calculate_psnr(outputs, labels)

        ssim_score /= len(test_loader)
        psnr_score /= len(test_loader)

        # Visualize the last batch
        labels_np = labels.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        image = plot_comparison(labels_np, outputs_np, slice_idx=16)

        # Log metrics and image to WandB
        wandb.log({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "ssim": ssim_score,
            "psnr": psnr_score,
            "image": wandb.Image(image),
        })

        # Print SSIM and PSNR for the current epoch
        print(f'SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.4f}')

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    encoder, diffusion_process, decoder = model
    encoder.eval()
    diffusion_process.eval()
    decoder.eval()

    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            latent = encoder(inputs)
            noisy_latent = diffusion_process(latent, noise_level=0.1)
            outputs = decoder(noisy_latent)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f'Average Test Loss: {avg_loss:.4f}')

# Load the data
train_list = ['32']
test_list = ['32']
print(f'Loading train data ...')
eeg_train, fmri_train = load_h5_from_list(args.data_root, train_list)
print(f'Loading test data ...')
eeg_test, fmri_test = load_h5_from_list(args.data_root, test_list)

print("Shapes")
print(f"EEG Train Shape: {eeg_train.shape}")
print(f"fMRI Train Shape: {fmri_train.shape}")
print(f"EEG Test Shape: {eeg_test.shape}")
print(f"fMRI Test Shape: {fmri_test.shape}")

# Normalize the data to range [0, 1]
eeg_train = normalize_data(eeg_train, scale_range=(0, 1))
fmri_train = normalize_data(fmri_train, scale_range=(0, 1))
eeg_test = normalize_data(eeg_test, scale_range=(0, 1))
fmri_test = normalize_data(fmri_test, scale_range=(0, 1))

train_dataset = TensorDataset(torch.tensor(eeg_train, dtype=torch.float32), torch.tensor(fmri_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(eeg_test, dtype=torch.float32), torch.tensor(fmri_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize a new W&B run with the current timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
exp_name = f"{filename}_dataset_{args.dataset_name}_run_{timestamp}"

exp_dir = os.path.join(args.work_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

run = wandb.init(project="eeg_fmri_project", name=exp_name)

# Train the model
model = (encoder, diffusion_process, decoder)
train(model, train_loader, criterion, optimizer, device)

# Evaluate the model
evaluate(model, test_loader, criterion, device)

# End W&B run
wandb.finish()
