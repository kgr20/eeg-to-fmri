# Works but low SSIM, although the detail is there and seems to be really good. 
# 20240729_Diffusion_2.py_dataset_01_run_20240730-084012 (SSIM 0.17 or so)
# there is the inverted color issue which i think causes the low SSIM, high loss, and nevative psnr


import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from io import BytesIO
from PIL import Image
import wandb
from datetime import datetime
from diffusers import DDPMScheduler, UNet2DModel
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

filename = "20240729_Diffusion_2.py"

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

# Normalize the data to range [-1, 1]
def normalize_data(data: np.ndarray, scale_range: tuple = None):
    new_data = 2 * (data - data.min()) / (data.max() - data.min()) - 1
    if scale_range is not None:
        a, b = scale_range
        assert a <= b, f'Invalid range: {scale_range}'
        new_data = (b - a) * new_data + a
    return new_data

# Custom UNet-like architecture with diffusion
class CustomUNetDiffusion(nn.Module):
    def __init__(self, output_size):
        super(CustomUNetDiffusion, self).__init__()
        self.output_size = output_size

        # Encoder
        self.encoder1 = self.conv_block(10, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)
        self.encoder6 = self.conv_block(1024, 2048)

        # Bottleneck (latent space for diffusion)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder6 = self.upconv_block(2048, 1024)
        self.decoder5 = self.upconv_block(1024, 512)
        self.decoder4 = self.upconv_block(512, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder2 = self.upconv_block(128, 64)
        self.decoder1 = self.upconv_block(64, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, 30, kernel_size=1)  # Change the output channels to 30
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)  # Adjust the upsample size to (64, 64)
        self.final_activation = nn.Tanh()

        # Diffusion model scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Initialize SSIM
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, tensor, target_size):
        _, _, h, w = tensor.size()
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        cropped_tensor = tensor[:, :, i:i+th, j:j+tw]
        return cropped_tensor

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        # print(f"After encoder1: {x1.shape}")
        x2 = self.encoder2(x1)
        # print(f"After encoder2: {x2.shape}")
        x3 = self.encoder3(x2)
        # print(f"After encoder3: {x3.shape}")
        x4 = self.encoder4(x3)
        # print(f"After encoder4: {x4.shape}")
        x5 = self.encoder5(x4)
        # print(f"After encoder5: {x5.shape}")
        x6 = self.encoder6(x5)
        # print(f"After encoder6: {x6.shape}")

        # Bottleneck (diffusion process)
        x_bottleneck = self.bottleneck(x6)
        # print(f"Bottleneck output shape: {x_bottleneck.shape}")

        timesteps = torch.randint(0, 1000, (x.size(0),), device=x.device).long()
        noise = torch.randn_like(x_bottleneck).to(x.device)
        x_noisy = self.scheduler.add_noise(x_bottleneck, noise, timesteps)

        # print(f"Noisy input shape: {x_noisy.shape}")

        # Decoder
        x = self.center_crop(x_noisy, x6.shape[2:])
        x = self.decoder6(x + x6)
        # print(f"After decoder6: {x.shape}")

        x = self.center_crop(x, x5.shape[2:])
        x = self.decoder5(x + x5)
        # print(f"After decoder5: {x.shape}")

        x = self.center_crop(x, x4.shape[2:])
        x = self.decoder4(x + x4)
        # print(f"After decoder4: {x.shape}")

        x = self.center_crop(x, x3.shape[2:])
        x = self.decoder3(x + x3)
        # print(f"After decoder3: {x.shape}")

        x = self.center_crop(x, x2.shape[2:])
        x = self.decoder2(x + x2)
        # print(f"After decoder2: {x.shape}")

        x = self.center_crop(x, x1.shape[2:])
        x = self.decoder1(x + x1)
        # print(f"After decoder1: {x.shape}")

        x = self.final_conv(x)
        # print(f"After final_conv: {x.shape}")
        x = self.upsample(x)
        x = self.final_activation(x)  # Apply Tanh activation
        # print(f"Final output shape: {x.shape}")

        return x

def plot_comparison(labels, outputs, slice_idx=16):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
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

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def main():
    # Load the data
    # train_list = ['32']
    # test_list = ['32']

    # Train/test from David
    # train_list = ['32', '35', '36', '37', '38', '39', '40', '42']
    # test_list = ['43', '44']

    # Train/test Quan-Kris
    test_list = ['43', '44']
    train_list = [Path(indv).stem for indv in os.listdir(args.data_root) if Path(indv).stem not in test_list]
    
    print(f'Loading train data ...')
    eeg_train, fmri_train = load_h5_from_list(args.data_root, train_list)
    print(f'Loading test data ...')
    eeg_test, fmri_test = load_h5_from_list(args.data_root, test_list)
    
    print("Shapes")
    print(f"EEG Train Shape: {eeg_train.shape}")
    print(f"fMRI Train Shape: {fmri_train.shape}")
    print(f"EEG Test Shape: {eeg_test.shape}")
    print(f"fMRI Test Shape: {fmri_test.shape}")

    # Transpose to match PyTorch's [C, H, W] format
    eeg_train = eeg_train.transpose(0, 3, 1, 2)
    fmri_train = fmri_train.transpose(0, 3, 1, 2)
    eeg_test = eeg_test.transpose(0, 3, 1, 2)
    fmri_test = fmri_test.transpose(0, 3, 1, 2)

    print("Shapes after transposition")
    print(f"EEG Train Shape: {eeg_train.shape}")
    print(f"fMRI Train Shape: {fmri_train.shape}")
    print(f"EEG Test Shape: {eeg_test.shape}")
    print(f"fMRI Test Shape: {fmri_test.shape}")

    # Normalize the data to range [-1, 1]
    eeg_train = normalize_data(eeg_train, scale_range=(-1, 1))
    fmri_train = normalize_data(fmri_train, scale_range=(-1, 1))
    eeg_test = normalize_data(eeg_test, scale_range=(-1, 1))
    fmri_test = normalize_data(fmri_test, scale_range=(-1, 1))

    train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train, dtype=torch.float32),
                                   fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
    test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test, dtype=torch.float32),
                                  fmri_data=torch.tensor(fmri_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model
    model = CustomUNetDiffusion(output_size=fmri_test.shape[-1]).to(device)
    
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define the SSIM loss function
    criterion = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Initialize a new W&B run with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f"{filename}_dataset_{args.dataset_name}_run_{timestamp}"

    exp_dir = os.path.join(args.work_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    run = wandb.init(project="eeg_fmri_project", name=exp_name)
    
    # Training loop
    num_epochs = args.num_epochs
    total_training_time = 0.0
    best_ssim = -1.0
    best_psnr = -1.0
    best_model_weights = None
    best_save_path = None

    pbar = tqdm(range(num_epochs), leave=True)

    for epoch in pbar:
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
            noise = torch.randn_like(inputs).to(device)
            noisy_inputs = model.scheduler.add_noise(inputs, noise, timesteps)
            outputs = model(noisy_inputs)

            loss = 1 - criterion(outputs, labels)
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
                inputs = inputs.to(device)
                labels = labels.to(device)
                timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
                noise = torch.randn_like(inputs).to(device)
                noisy_inputs = model.scheduler.add_noise(inputs, noise, timesteps)
                outputs = model(noisy_inputs)
                ssim_score += criterion(outputs, labels).item()
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

if __name__ == "__main__":
    main()