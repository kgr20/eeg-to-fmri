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

# Define the model
class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 30, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        print(f"After conv1 shape: {x.shape}")
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        print(f"After conv2 shape: {x.shape}")
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        print(f"After conv3 shape: {x.shape}")
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        print(f"After conv4 shape: {x.shape}")
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        print(f"After conv5 shape: {x.shape}")
        return x

model = SimpleConvModel().to(device)

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Define the combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    def forward(self, outputs, labels):
        mse_loss = self.mse_loss(outputs, labels)
        ssim_loss = 1 - self.ssim_metric(outputs, labels)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

criterion = CombinedLoss().to(device)

def plot_comparison(labels, outputs, slice_idx=16):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    label_slice = labels[0, :, :, slice_idx]
    output_slice = outputs[0, :, :, slice_idx]
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
def load_h5_from_list(data_root: str, individual_list: List[str]):
    eeg_data = []
    fmri_data = []

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Individual {individual_name}')
        file_path = Path(data_root) / f'{individual_name}.h5'
        with h5py.File(file_path, 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            eeg_data.append(eeg_indv)
            fmri_data.append(fmri_indv)
    
    eeg_data = np.concatenate(eeg_data, axis=0)
    fmri_data = np.concatenate(fmri_data, axis=0)

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

# Create DataLoader objects
train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train, dtype=torch.float32),
                               fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test, dtype=torch.float32),
                              fmri_data=torch.tensor(fmri_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

calculate_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

for epoch in pbar:
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        inputs = inputs.to(device)  # (batch_size, 64, 269, 10)
        labels = labels.to(device)  # (batch_size, 64, 64, 30)

        optimizer.zero_grad()
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
        noise = torch.randn_like(inputs).to(device)  # (batch_size, 64, 269, 10)
        noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)  # (batch_size, 64, 269, 10)
        outputs = model(noisy_inputs)  # (batch_size, 30, 64, 64)

        print(f"Noisy inputs shape: {noisy_inputs.shape}")
        print(f"Model outputs shape: {outputs.shape}")

        # Ensure shapes match before SSIM calculation
        if outputs.shape != labels.shape:
            print(f"Shape mismatch - outputs: {outputs.shape}, labels: {labels.shape}")
            outputs = outputs[:, :, :labels.shape[2], :labels.shape[3]]  # Align shapes

        # Calculate combined loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_training_time += epoch_duration

    # Evaluation
    model.eval()
    ssim_score = 0.0
    psnr_score = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            print(f"Batch inputs shape (eval): {inputs.shape}")
            print(f"Batch labels shape (eval): {labels.shape}")
            inputs = inputs.to(device)  # (batch_size, 64, 269, 10)
            labels = labels.to(device)  # (batch_size, 64, 64, 30)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
            noise = torch.randn_like(inputs).to(device)  # (batch_size, 64, 269, 10)
            noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)  # (batch_size, 64, 269, 10)
            outputs = model(noisy_inputs)  # (batch_size, 30, 64, 64)

            print(f"Noisy inputs shape (eval): {noisy_inputs.shape}")
            print(f"Model outputs shape (eval): {outputs.shape}")

            ssim_score += calculate_ssim(outputs, labels).item()
            psnr_score += calculate_psnr(outputs, labels)

    ssim_score /= len(test_loader)
    psnr_score /= len(test_loader)

    # visualize the last batch
    labels_np = labels.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    image = plot_comparison(labels_np, outputs_np, slice_idx=16)

    run.log({
        "lr": optimizer.param_groups[0]["lr"],
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
