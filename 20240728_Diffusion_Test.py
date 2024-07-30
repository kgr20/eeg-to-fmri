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
from tqdm import tqdm
from datetime import datetime
from diffusers import DDPMScheduler
import torchmetrics
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description="EEG to fMRI Diffusion Model Training Script")
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
def load_h5_from_list(data_root: str, individual_list: List[str]):
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

# Define the SimpleDiffusionModel
class SimpleDiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleDiffusionModel, self).__init__()
        
        # Encoder (Downsampling path)
        self.encoder1 = self.conv_block(in_channels, 64)  # First convolution block
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # First pooling layer
        
        self.encoder2 = self.conv_block(64, 128)  # Second convolution block
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Second pooling layer
        
        self.encoder3 = self.conv_block(128, 256)  # Third convolution block
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Third pooling layer
        
        # Bottleneck (Deepest layer in the network)
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (Upsampling path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # First upsampling layer
        self.decoder3 = self.conv_block(512, 256)  # First decoding block
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Second upsampling layer
        self.decoder2 = self.conv_block(256, 128)  # Second decoding block
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Third upsampling layer
        self.decoder1 = self.conv_block(128, 64)  # Third decoding block
        
        # Additional layers to adjust the final dimensions
        self.conv_final1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=(1, 1))  # Adjust dimensions
        self.conv_final2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=(1, 2))  # Adjust dimensions
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # Output convolution layer

    def conv_block(self, in_channels, out_channels):
        # Define a block of two convolutional layers with ReLU activation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # First conv layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second conv layer
            nn.ReLU(inplace=True)  # ReLU activation
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)  # First convolution block
        print(f"After conv1: {enc1.shape} (expected: (N, 64, 64, 269))")
        
        enc2 = self.encoder2(self.pool1(enc1))  # First pooling + second convolution block
        print(f"After pool1 + conv2: {enc2.shape} (expected: (N, 128, 32, 134))")
        
        enc3 = self.encoder3(self.pool2(enc2))  # Second pooling + third convolution block
        print(f"After pool2 + conv3: {enc3.shape} (expected: (N, 256, 16, 67))")
        
        bottleneck = self.bottleneck(self.pool3(enc3))  # Third pooling + bottleneck convolution block
        print(f"After pool3 + bottleneck: {bottleneck.shape} (expected: (N, 512, 8, 33))")
        
        # Decoder path
        dec3 = self.upconv3(bottleneck)  # First upsampling
        print(f"After upconv3: {dec3.shape} (expected: (N, 256, 16, 66))")
        dec3 = self.center_crop(enc3, dec3)  # Center crop to match encoder output size
        dec3 = torch.cat((enc3, dec3), dim=1)  # Concatenate with encoder output
        print(f"After concat with enc3: {dec3.shape} (expected: (N, 512, 16, 67))")
        dec3 = self.decoder3(dec3)  # First decoding block
        print(f"After decoder3: {dec3.shape} (expected: (N, 256, 16, 67))")
        
        dec2 = self.upconv2(dec3)  # Second upsampling
        print(f"After upconv2: {dec2.shape} (expected: (N, 128, 32, 134))")
        dec2 = self.center_crop(enc2, dec2)  # Center crop to match encoder output size
        dec2 = torch.cat((enc2, dec2), dim=1)  # Concatenate with encoder output
        print(f"After concat with enc2: {dec2.shape} (expected: (N, 256, 32, 134))")
        dec2 = self.decoder2(dec2)  # Second decoding block
        print(f"After decoder2: {dec2.shape} (expected: (N, 128, 32, 134))")
        
        dec1 = self.upconv1(dec2)  # Third upsampling
        print(f"After upconv1: {dec1.shape} (expected: (N, 64, 64, 268))")
        dec1 = self.center_crop(enc1, dec1)  # Center crop to match encoder output size
        dec1 = torch.cat((enc1, dec1), dim=1)  # Concatenate with encoder output
        print(f"After concat with enc1: {dec1.shape} (expected: (N, 128, 64, 269))")
        dec1 = self.decoder1(dec1)  # Third decoding block
        print(f"After decoder1: {dec1.shape} (expected: (N, 64, 64, 269))")
        
        # Adjusting dimensions to match the final output shape
        dec1 = self.conv_final1(dec1)
        print(f"After conv_final1: {dec1.shape} (expected: (N, 128, 64, 269))")
        
        dec1 = self.conv_final2(dec1)
        print(f"After conv_final2: {dec1.shape} (expected: (N, 64, 64, 135))")
        
        dec1 = nn.functional.interpolate(dec1, size=(64, 64, 30), mode='bilinear', align_corners=False)
        print(f"After interpolation: {dec1.shape} (expected: (N, 64, 64, 30))")
        
        final_output = self.final_conv(dec1)  # Final convolution to match output dimensions
        print(f"After final_conv: {final_output.shape} (expected: (N, 64, 64, 30))")
        
        return final_output
    
    def center_crop(self, enc_feature, dec_feature):
        """Center crop dec_feature to the size of enc_feature."""
        _, _, h, w = enc_feature.shape
        dec_feature = nn.functional.interpolate(dec_feature, size=(h, w), mode='bilinear', align_corners=False)
        return dec_feature

def normalize_data(data: np.ndarray, scale_range: Tuple=None):
    """Normalize data to range [a, b]"""
    new_data = (data - data.min())/(data.max() - data.min())
    if scale_range is not None:
        a, b = scale_range
        assert a <= b, f'Invalid range: {scale_range}'
        new_data = (b-a)*new_data + a
    return new_data

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

def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def main():
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

    # Normalize the data to range [0, 1]
    eeg_train = normalize_data(eeg_train, scale_range=(0, 1))
    fmri_train = normalize_data(fmri_train, scale_range=(0, 1))
    eeg_test = normalize_data(eeg_test, scale_range=(0, 1))
    fmri_test = normalize_data(fmri_test, scale_range=(0, 1))

    train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train, dtype=torch.float32),
                                   fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
    test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test, dtype=torch.float32),
                                  fmri_data=torch.tensor(fmri_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model
    model = SimpleDiffusionModel(in_channels=10, out_channels=30).to(device)
    
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define the loss function
    criterion = nn.MSELoss().to(device)
    
    # Initialize the scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize a new W&B run with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f"dataset_{args.dataset_name}_run_{timestamp}"

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

    calculate_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in pbar:
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
            noise = torch.randn_like(inputs).to(device)
            noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)
            outputs = model(noisy_inputs)

            loss = criterion(outputs, labels)
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
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
                noise = torch.randn_like(inputs).to(device)
                noisy_inputs = scheduler.add_noise(inputs, noise, timesteps)
                outputs = model(noisy_inputs)
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
