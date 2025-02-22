import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Tuple, List
from io import BytesIO
from PIL import Image
import wandb
from tqdm import tqdm
from datetime import datetime
from diffusers import DDPMScheduler
import torchmetrics
from torchvision import transforms
import argparse
from torchmetrics.image import StructuralSimilarityIndexMeasure

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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Define the UNet model for latent diffusion
class LatentUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LatentUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),  # [B, 64, D, H, W]
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),          # [B, 128, D, H, W]
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)                  # [B, 128, D//2, H//2, W//2]
        )
        self.middle = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),         # [B, 256, D//2, H//2, W//2]
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),         # [B, 512, D//2, H//2, W//2]
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2), # [B, 256, D, H, W]
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),         # [B, 128, D, H, W]
            nn.ReLU(inplace=True),
            nn.Conv3d(128, out_channels, kernel_size=3, padding=1) # [B, out_channels, D, H, W]
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x1 = self.encoder(x)
        print(f"Encoded shape: {x1.shape}")
        x2 = self.middle(x1)
        print(f"Middle shape: {x2.shape}")
        x3 = self.decoder(x2)
        print(f"Decoded shape: {x3.shape}")
        return x3

# Define the model
model = LatentUNet(in_channels=64, out_channels=30).to(device)  # Adjusted in_channels to 64 for EEG latent space

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Define the loss function
criterion = nn.MSELoss().to(device)

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

# Normalize data
def normalize_data(data: np.ndarray, scale_range: Tuple=None):
    """Normalize data to range [a, b]"""
    new_data = (data - data.min())/(data.max() - data.min())
    if scale_range is not None:
        a, b = scale_range
        assert a <= b, f'Invalid range: {scale_range}'
        new_data = (b-a)*new_data + a
    return new_data

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# Load h5 data function
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

# Load the data
train_list = ['32']
test_list = ['32']

print(f'Loading train data ...')
eeg_train, fmri_train = load_h5_from_list(args.data_root, train_list)
print(f'Loading test data ...')
eeg_test, fmri_test = load_h5_from_list(args.data_root, test_list)

print("Shapes")
print(f"EEG Train Shape: {eeg_train.shape}")   # (287, 64, 269, 10)
print(f"fMRI Train Shape: {fmri_train.shape}") # (287, 64, 64, 30)
print(f"EEG Test Shape: {eeg_test.shape}")     # (287, 64, 269, 10)
print(f"fMRI Test Shape: {fmri_test.shape}")   # (287, 64, 64, 30)

# Transpose to match PyTorch's [B, C, D, H, W] format
eeg_train = eeg_train.transpose(0, 1, 3, 2)  # (287, 64, 10, 269)
fmri_train = fmri_train.transpose(0, 3, 1, 2)  # (287, 30, 64, 64)
eeg_test = eeg_test.transpose(0, 1, 3, 2)  # (287, 64, 10, 269)
fmri_test = fmri_test.transpose(0, 3, 1, 2)  # (287, 30, 64, 64)

print("Shapes after transposition")
print(f"EEG Train Shape: {eeg_train.shape}")   # (287, 64, 10, 269)
print(f"fMRI Train Shape: {fmri_train.shape}") # (287, 30, 64, 64)
print(f"EEG Test Shape: {eeg_test.shape}")     # (287, 64, 10, 269)
print(f"fMRI Test Shape: {fmri_test.shape}")

# Normalize the data to range [0, 1]
eeg_train = normalize_data(eeg_train, scale_range=(0, 1))
fmri_train = normalize_data(fmri_train, scale_range=(0, 1))
eeg_test = normalize_data(eeg_test, scale_range=(0, 1))
fmri_test = normalize_data(fmri_test, scale_range=(0, 1))

# DataLoader
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

# Update the train and test datasets with the transposed data
train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train, dtype=torch.float32),
                               fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test, dtype=torch.float32),
                              fmri_data=torch.tensor(fmri_test, dtype=torch.float32))

# Re-create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Sample and check shapes again
sample_image = torch.tensor(eeg_train[0:1], dtype=torch.float32).to(device)
print("Input shape after transposing:", sample_image.shape)  # (1, 64, 10, 269)

# Initialize a new W&B run with the current timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
exp_name = f"dataset_{args.dataset_name}_run_{timestamp}"

exp_dir = os.path.join(args.work_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

run = wandb.init(project="eeg_fmri_project", name=exp_name)

# Initialize the scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

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
        inputs = inputs.to(device)  # (batch_size, 64, 10, 269)
        labels = labels.to(device)  # (batch_size, 30, 64, 64)

        optimizer.zero_grad()
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
        noise = torch.randn_like(inputs).to(device)  # (batch_size, 64, 10, 269)
        noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)  # (batch_size, 64, 10, 269)
        outputs = model(noisy_inputs)  # (batch_size, 30, 64, 64)

        # Ensure shapes match before SSIM calculation
        if outputs.shape != labels.shape:
            print(f"Shape mismatch - outputs: {outputs.shape}, labels: {labels.shape}")
            outputs = outputs[:, :, :labels.shape[2], :labels.shape[3]]  # Align shapes

        # Calculate SSIM loss
        ssim_loss = 1 - ssim_metric(outputs, labels)  # Use outputs directly
        loss = ssim_loss
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

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
            inputs = inputs.to(device)  # (batch_size, 64, 10, 269)
            labels = labels.to(device)  # (batch_size, 30, 64, 64)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
            noise = torch.randn_like(inputs).to(device)  # (batch_size, 64, 10, 269)
            noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)  # (batch_size, 64, 10, 269)
            outputs = model(noisy_inputs)  # (batch_size, 30, 64, 64)

            # Ensure shapes match before SSIM calculation
            if outputs.shape != labels.shape:
                print(f"Shape mismatch - outputs: {outputs.shape}, labels: {labels.shape}")
                outputs = outputs[:, :, :labels.shape[2], :labels.shape[3]]  # Align shapes

            ssim_score += calculate_ssim(outputs, labels).item()  # Use outputs directly
            psnr_score += calculate_psnr(outputs, labels)  # Use outputs directly

    ssim_score /= len(test_loader)
    psnr_score /= len(test_loader)

    # visualize the last batch
    labels_np = labels.cpu().numpy()
    outputs_np = outputs.cpu().numpy()  # Use outputs directly for visualization
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
        if best_save_path is not None:
            os.remove(best_save_path)
        best_save_path = os.path.join(exp_dir, f"epoch{epoch}_ssim_{best_ssim:.4f}_psnr_{best_psnr:.2f}.pth")
        torch.save(model.state_dict(), best_save_path)

    pbar.set_description(f'SSIM: {ssim_score:.3f} / PSNR: {psnr_score:.3f} / Loss: {epoch_loss:.3f} / Best SSIM: {best_ssim:.3f}')

# Save final model
final_model_path = os.path.join(exp_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)

# End W&B run
wandb.finish()
