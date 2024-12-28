import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from io import BytesIO
from PIL import Image
import wandb
from tqdm import tqdm
from datetime import datetime
from diffusers import DDPMScheduler, UNet2DModel
import torchmetrics
import argparse
import os

# ---------------------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="EEG to fMRI Diffusion Model Training Script")
parser.add_argument('--dataset_name', type=str, default="01", help="Dataset identifier")
parser.add_argument('--data_root', type=str, default="/home/quan/Dataset/Aillis_data/EEG2fMRI/h5_data/NODDI",
                    help="Path to the dataset directory in h5 format")
parser.add_argument('--work_dir', type=str, default="/home/quan/WorkSpace/Kris/eeg-to-fmri/scratch",
                    help="Path to save experiments")
parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for optimizer")
parser.add_argument('--alpha', type=float, default=0.5, 
                    help="Weight for MSE vs. (1 - SSIM) in the combined loss; 1=all MSE, 0=all SSIM")

args = parser.parse_args()

# ---------------------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------
# Load .h5 data
# ---------------------------------------------------------------------------------------
def load_h5_from_list(data_root: str, individual_list: List):
    eeg_data = None
    fmri_data = None

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Individual {individual_name}')
        h5_path = Path(data_root) / f'{individual_name}.h5'
        if not h5_path.exists():
            print(f"[Warning] File {h5_path} does not exist. Skipping.")
            continue
        with h5py.File(h5_path, 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            if eeg_data is None:
                eeg_data = eeg_indv
            else:
                eeg_data = np.concatenate([eeg_data, eeg_indv], axis=0)

            if fmri_data is None:
                fmri_data = fmri_indv
            else:
                fmri_data = np.concatenate([fmri_data, fmri_indv], axis=0)
    
    if eeg_data is None or fmri_data is None:
        raise ValueError("No data loaded. Please check your individual_list and data_root.")
    
    return eeg_data, fmri_data

# ---------------------------------------------------------------------------------------
# Load the data
# ---------------------------------------------------------------------------------------

# Define the test_list with specific individuals for testing
test_list = ['43', '44']  # Individuals to be used for testing

# Construct train_list by excluding test_list individuals
train_list = [
    Path(indv).stem
    for indv in os.listdir(args.data_root)
    if Path(indv).stem not in test_list and (Path(args.data_root) / indv).is_file()
]

print("Training Individuals:", sorted(train_list))
print("Testing Individuals:", test_list)

# Load training data
print("Loading train data...")
eeg_train, fmri_train = load_h5_from_list(args.data_root, train_list)

# Load testing data
print("Loading test data...")
eeg_test, fmri_test = load_h5_from_list(args.data_root, test_list)

print("Shapes before transpose:")
print(f"EEG Train Shape: {eeg_train.shape}")
print(f"fMRI Train Shape: {fmri_train.shape}")
print(f"EEG Test Shape: {eeg_test.shape}")
print(f"fMRI Test Shape: {fmri_test.shape}")

# ---------------------------------------------------------------------------------------
# Transpose to match PyTorch [N, C, H, W]
# ---------------------------------------------------------------------------------------
eeg_train = eeg_train.transpose(0, 3, 1, 2)   # (B, 10, 64, 269)
fmri_train = fmri_train.transpose(0, 3, 1, 2) # (B, 30, 64, 64)
eeg_test = eeg_test.transpose(0, 3, 1, 2)
fmri_test = fmri_test.transpose(0, 3, 1, 2)

print("Shapes after transpose:")
print(f"EEG Train Shape: {eeg_train.shape}")
print(f"fMRI Train Shape: {fmri_train.shape}")
print(f"EEG Test Shape: {eeg_test.shape}")
print(f"fMRI Test Shape: {fmri_test.shape}")

# ---------------------------------------------------------------------------------------
# Normalize data
# ---------------------------------------------------------------------------------------
def normalize_data(data: np.ndarray, scale_range: Tuple=None):
    """
    Normalize data to range [a, b].
    """
    dmin, dmax = data.min(), data.max()
    new_data = (data - dmin) / (dmax - dmin + 1e-8)
    if scale_range is not None:
        a, b = scale_range
        new_data = (b - a) * new_data + a
    return new_data

eeg_train = normalize_data(eeg_train, (0, 1))
fmri_train = normalize_data(fmri_train, (0, 1))
eeg_test = normalize_data(eeg_test, (0, 1))
fmri_test = normalize_data(fmri_test, (0, 1))

# ---------------------------------------------------------------------------------------
# Build Dataset / DataLoader
# ---------------------------------------------------------------------------------------
train_dataset = EEGfMRIDataset(
    torch.tensor(eeg_train, dtype=torch.float32),
    torch.tensor(fmri_train, dtype=torch.float32)
)
test_dataset = EEGfMRIDataset(
    torch.tensor(eeg_test, dtype=torch.float32),
    torch.tensor(fmri_test, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ---------------------------------------------------------------------------------------
# Custom UNet2DModel
# ---------------------------------------------------------------------------------------
class CustomUNet2DModel(UNet2DModel):
    def forward(self, sample, timesteps=None):
        return super().forward(sample, timesteps)

# ---------------------------------------------------------------------------------------
# Encoder
#   We reflect-pad from w=269 -> w=288 so that after 4 stride=2 convs, we end up with w=18.
#   Reflection pad in PyTorch requires specifying (left, right, top, bottom) for 4D input.
# ---------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_channels=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x shape: (B,10,64,269)
        b, c, h, w = x.shape
        if w < 288:
            pad_amt = 288 - w  # e.g. 19
            # For 4D input [N,C,H,W], reflect/replicate only works on the last 2 dims.
            # pad=(left, right, top, bottom).
            # We want to pad the width dimension by pad_amt on the right only.
            # => (left=0, right=pad_amt, top=0, bottom=0)
            x = F.pad(x, (0, pad_amt, 0, 0), mode='reflect')
            print(f"[Encoder] Reflection pad from width={w} to 288")

        x = self.encoder(x)
        print(f"[Encoder] output after convs: {x.shape}")
        return x  # (B,512,4,18)

# ---------------------------------------------------------------------------------------
# Create the U-Net
# ---------------------------------------------------------------------------------------
def create_diffusion_unet():
    unet = CustomUNet2DModel(
        sample_size=(4,18),
        in_channels=512,
        out_channels=512,
        layers_per_block=1,
        block_out_channels=(256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D")
    )
    return unet

# ---------------------------------------------------------------------------------------
# Decoder
#   (512,4,18) -> up1->(256,8,36) -> up2->(128,16,72) -> up3->(64,32,72) -> up4->(64,64,72)
#   final convs => width:72->64, channels:64->30
# ---------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, output_channels=30):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(2,1), stride=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(2,1), stride=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Compress width from 72->64
        self.conv_width = nn.Conv2d(64, 64, kernel_size=(1,9), stride=1)
        # Channels from 64->30
        self.conv_channels = nn.Conv2d(64, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"[Decoder] input: {x.shape}")
        x = self.up1(x)   # -> (256,8,36)
        x = self.up2(x)   # -> (128,16,72)
        x = self.up3(x)   # -> (64,32,72)
        x = self.up4(x)   # -> (64,64,72)
        print(f"[Decoder] after up4: {x.shape}")

        x = self.conv_width(x)  # -> (64,64,64)
        print(f"[Decoder] after width-compress: {x.shape}")

        x = self.conv_channels(x)  # -> (30,64,64)
        x = self.sigmoid(x)
        print(f"[Decoder] output: {x.shape}")
        return x

# ---------------------------------------------------------------------------------------
# Complete model
# ---------------------------------------------------------------------------------------
class EEGtoFMRIDiffusionModel(nn.Module):
    def __init__(self, encoder, diffusion_model, decoder):
        super().__init__()
        self.encoder = encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder

    def forward(self, x, timesteps):
        print(f"[Model] Input shape: {x.shape}")
        x = self.encoder(x)  # => (B,512,4,18)
        out = self.diffusion_model(x, timesteps).sample  # => (B,512,4,18)
        print(f"[UNet] output: {out.shape}")
        out = self.decoder(out)  # => (B,30,64,64)
        return out

# ---------------------------------------------------------------------------------------
# Instantiate model
# ---------------------------------------------------------------------------------------
encoder = Encoder(input_channels=10).to(device)
unet_diffusion = create_diffusion_unet().to(device)
decoder = Decoder(output_channels=30).to(device)
model = EEGtoFMRIDiffusionModel(encoder, unet_diffusion, decoder).to(device)

# ---------------------------------------------------------------------------------------
# Optimizer & Combined Loss
# ---------------------------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# We'll combine MSE with (1 - SSIM)
def combined_mse_ssim_loss(pred, target, ssim_module, alpha=0.5):
    mse_val = F.mse_loss(pred, target)
    ssim_val = ssim_module(pred, target)  # in [0,1]
    return alpha * mse_val + (1 - alpha) * (1 - ssim_val)

# ---------------------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------
# PSNR Calculation
# ---------------------------------------------------------------------------------------
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# ---------------------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------------------
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
exp_name = f"{script_name}_dataset_{args.dataset_name}_run_{timestamp}"

exp_dir = os.path.join(args.work_dir, exp_name)
os.makedirs(exp_dir, exist_ok=True)

run = wandb.init(project="eeg_fmri_project", name=exp_name)

# ---------------------------------------------------------------------------------------
# Diffusion Scheduler (Minimal Noise)
# ---------------------------------------------------------------------------------------
scheduler = DDPMScheduler(num_train_timesteps=10)  # drastically reduced from 1000

# ---------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------
best_ssim = -1.0
best_psnr = -1.0
best_save_path = None

calculate_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

pbar = tqdm(range(args.num_epochs), leave=True)
for epoch in pbar:
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Minimal noise injection:
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
        noise = torch.randn_like(inputs).to(device)
        noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)

        # If you want truly exact recon, feed the original 'inputs' to model(...)  
        # but let's do minimal noise for demonstration:
        outputs = model(noisy_inputs, timesteps)

        loss = combined_mse_ssim_loss(outputs, labels, calculate_ssim, alpha=args.alpha)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # Evaluation
    model.eval()
    ssim_score = 0.0
    psnr_score = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
            noise = torch.randn_like(inputs).to(device)
            noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)

            outputs = model(noisy_inputs, timesteps)

            ssim_score += calculate_ssim(outputs, labels).item()
            psnr_score += calculate_psnr(outputs, labels)

    ssim_score /= len(test_loader)
    psnr_score /= len(test_loader)

    # Visualization
    labels_np = labels.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    image = plot_comparison(labels_np, outputs_np, slice_idx=16)

    # Log to W&B
    run.log({
        "epoch": epoch,
        "loss": epoch_loss,
        "ssim": ssim_score,
        "psnr": psnr_score,
        "image": wandb.Image(image)
    })

    # Save best
    if ssim_score > best_ssim:
        best_ssim = ssim_score
        best_psnr = psnr_score
        if best_save_path is not None and os.path.exists(best_save_path):
            os.remove(best_save_path)
        best_save_path = os.path.join(
            exp_dir, f"epoch{epoch}_ssim_{best_ssim:.4f}_psnr_{best_psnr:.2f}.pth"
        )
        torch.save(model.state_dict(), best_save_path)

    pbar.set_description(
        f"Epoch {epoch} - Loss: {epoch_loss:.4f} | SSIM: {ssim_score:.3f} | PSNR: {psnr_score:.3f}"
    )

# Save final model
final_model_path = os.path.join(exp_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)

wandb.finish()