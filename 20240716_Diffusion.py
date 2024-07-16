import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time
import torchmetrics
import os
from typing import Tuple
from io import BytesIO
from PIL import Image
import wandb
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms

# Hyperparameters
hyperparameters = {
    'dataset_name': '01',
    'data_path': '/home/aca10131kr/datasets',  # Set your dataset path here
    'work_dir': '/home/aca10131kr/experiments',  # Set your working directory here
    'num_epochs': 5,
    'batch_size': 64,
    'lr': 0.0001,
    'weight_decay': 1e-5,
    'train_subset': 10,
    'test_subset': 2
}

# Load the data
data_path = os.path.join(hyperparameters['data_path'], f"{hyperparameters['dataset_name']}_eeg_fmri_data.h5")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# W&B login
wandb.login(key="ac810e6bfc6fc90ede8806af8be88689bd635524")  # Add your W&B key here

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution (adjust as per your fMRI data)
    train_batch_size = hyperparameters['batch_size']
    eval_batch_size = hyperparameters['batch_size']  # how many images to sample during evaluation
    num_epochs = hyperparameters['num_epochs']
    gradient_accumulation_steps = 1
    learning_rate = hyperparameters['lr']
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = hyperparameters['work_dir']  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

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
        assert a <= b, f'Invalid range: {scale_range}'
        new_data = (b-a)*new_data + a

    return new_data

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

print(f'Loading data ...')
# each data has: [N_sample, H, W, C]
with h5py.File(data_path, 'r') as f:
    eeg_train = np.array(f['eeg_train'][:hyperparameters['train_subset']])
    fmri_train = np.array(f['fmri_train'][:hyperparameters['train_subset']])
    eeg_test = np.array(f['eeg_test'][:hyperparameters['test_subset']])
    fmri_test = np.array(f['fmri_test'][:hyperparameters['test_subset']])

# Print shapes before transposing
print(f"EEG Train Shape before transpose: {eeg_train.shape}")
print(f"fMRI Train Shape before transpose: {fmri_train.shape}")
print(f"EEG Test Shape before transpose: {eeg_test.shape}")
print(f"fMRI Test Shape before transpose: {fmri_test.shape}")

# In PyTorch, 1 data sample is represented as [C, H, W]
eeg_train = eeg_train.transpose(0, 3, 1, 2)
fmri_train = fmri_train.transpose(0, 3, 1, 2)

eeg_test = eeg_test.transpose(0, 3, 1, 2)
fmri_test = fmri_test.transpose(0, 3, 1, 2)

# Print shapes after transposing
print(f"EEG Train Shape after transpose: {eeg_train.shape}")
print(f"fMRI Train Shape after transpose: {fmri_train.shape}")
print(f"EEG Test Shape after transpose: {eeg_test.shape}")
print(f"fMRI Test Shape after transpose: {fmri_test.shape}")

# Normalize the data to range [a, b]
eeg_train = normalize_data(eeg_train, scale_range=(0, 1))
fmri_train = normalize_data(fmri_train, scale_range=(0, 1))

eeg_test = normalize_data(eeg_test, scale_range=(0, 1))
fmri_test = normalize_data(fmri_test, scale_range=(0, 1))

print(f"Normalized EEG Train Shape: {eeg_train.shape}")
print(f"Normalized fMRI Train Shape: {fmri_train.shape}")
print(f"Normalized EEG Test Shape: {eeg_test.shape}")
print(f"Normalized fMRI Test Shape: {fmri_test.shape}")

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

# Resizing to ensure compatibility
resize_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),  # Resize to 64x64
])

# Apply the transformation to the EEG data
eeg_train_resized = np.array([resize_transform(torch.tensor(image, dtype=torch.float32).permute(1, 2, 0)).permute(2, 0, 1) for image in eeg_train])
eeg_test_resized = np.array([resize_transform(torch.tensor(image, dtype=torch.float32).permute(1, 2, 0)).permute(2, 0, 1) for image in eeg_test])

# Ensure the number of channels remains the same
eeg_train_resized = eeg_train_resized[:, :10, :, :]
eeg_test_resized = eeg_test_resized[:, :10, :, :]

# Update the train and test datasets with the resized images
train_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_train_resized, dtype=torch.float32),
                               fmri_data=torch.tensor(fmri_train, dtype=torch.float32))
test_dataset = EEGfMRIDataset(eeg_data=torch.tensor(eeg_test_resized, dtype=torch.float32),
                              fmri_data=torch.tensor(fmri_test, dtype=torch.float32))

# Re-create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

# Sample and check shapes again
sample_image = torch.tensor(eeg_train_resized[0:1], dtype=torch.float32).to(device)
print("Input shape after resizing:", sample_image.shape)

# Expected shape: [1, 10, 64, 64], Actual: ...
print("Expected output shape: [1, 30, 64, 64]")

# Initialize a new W&B run with the current timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
exp_name = f"dataset_{hyperparameters['dataset_name']}_run_{timestamp}"

exp_dir = os.path.join(hyperparameters['work_dir'], exp_name)
os.makedirs(exp_dir, exist_ok=True)

run = wandb.init(project="eeg_fmri_project", name=exp_name)

# Initialize the model, eeg: [H, W, 10], fmri: [H, W, 30]
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=10,  # the number of input channels, 10 for EEG data
    out_channels=30,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

# Initialize the scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Add noise to the image at a given timestep
timesteps = torch.tensor([50], dtype=torch.long).to(device)  # Example timestep value
noise = torch.randn(sample_image.shape).to(device)  # Generate random noise
noisy_image = scheduler.add_noise(sample_image, noise, timesteps)

# Pass the noisy image and the timestep to the model
output = model(noisy_image, timesteps)
print("Output shape:", output.sample.shape)

# Ensure that res_hidden_states has the same dimensions as hidden_states
def fix_dimensions(hidden_states, res_hidden_states):
    fixed_res_hidden_states = []
    for res_hs in res_hidden_states:
        print(f"Original res_hidden_states shape: {res_hs.shape}")
        while hidden_states.dim() > res_hs.dim():
            res_hs = res_hs.unsqueeze(-1)
            print(f"Unsqueezing res_hidden_states to shape: {res_hs.shape}")
        while hidden_states.dim() < res_hs.dim():
            hidden_states = hidden_states.unsqueeze(-1)
            print(f"Unsqueezing hidden_states to shape: {hidden_states.shape}")
        fixed_res_hidden_states.append(res_hs)
    return hidden_states, tuple(fixed_res_hidden_states)

# Updated forward function in your model
class MyUNetModel(UNet2DModel):
    def forward(self, hidden_states, timesteps):
        temb = None
        for name, layer in self.named_children():
            if "time_proj" in name:
                temb = layer(timesteps)
            elif "time_embedding" in name:
                temb = layer(temb)
            elif "resnet" in name or "down_blocks" in name or "up_blocks" in name:
                if isinstance(layer, nn.ModuleList):
                    for sublayer in layer:
                        if isinstance(sublayer, tuple):
                            sublayer, res_hidden_states = sublayer
                            hidden_states, res_hidden_states = fix_dimensions(hidden_states, res_hidden_states)
                        hidden_states = sublayer(hidden_states, temb)
                else:
                    hidden_states = layer(hidden_states, temb)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states

# Example usage
model = MyUNetModel(
    sample_size=config.image_size,
    in_channels=10,
    out_channels=30,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

# Function to check intermediate shapes
def check_shapes(input_tensor, model, timesteps):
    shapes = []
    x = input_tensor
    temb = None
    for name, layer in model.named_children():
        if "time_proj" in name:
            print(f"Applying time_proj: {name}")
            print(f"Timesteps: {timesteps.shape}")
            temb = layer(timesteps)  # Ensure timesteps is 1D
        elif "time_embedding" in name:
            print(f"Applying time_embedding: {name}")
            print(f"temb shape: {temb.shape}")
            temb = layer(temb)
        elif "resnet" in name or "down_blocks" in name or "up_blocks" in name:
            print(f"Applying ResNet or block: {name}")
            if isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    x = sublayer(x, temb)
                    if isinstance(x, tuple):
                        print(f"Sublayer output before unpacking: shapes")
                        x, res_hidden_states = x
                        print(f"hidden_states shape: {x.shape}")
                        for idx, res_hidden_state in enumerate(res_hidden_states):
                            print(f"res_hidden_states[{idx}] shape: {res_hidden_state.shape}")
                            shapes.append((f"{name}.{sublayer}.res_hidden_states[{idx}]", res_hidden_state.shape))
                        shapes.append((f"{name}.{sublayer}.hidden_states", x.shape))
                    else:
                        shapes.append((f"{name}.{sublayer}", x.shape))
            else:
                x = layer(x, temb)
                if isinstance(x, tuple):
                    print(f"Layer output before unpacking: shapes")
                    x, res_hidden_states = x
                    print(f"hidden_states shape: {x.shape}")
                    for idx, res_hidden_state in enumerate(res_hidden_states):
                        print(f"res_hidden_states[{idx}] shape: {res_hidden_state.shape}")
                        shapes.append((f"{name}.res_hidden_states[{idx}]", res_hidden_state.shape))
                    shapes.append((f"{name}.hidden_states", x.shape))
                else:
                    shapes.append((name, x.shape))
        else:
            print(f"Applying other layer: {name}")
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]  # Unpack if the layer returns a tuple
            shapes.append((name, x.shape))
    return shapes

# Run this after loading the model
timesteps = torch.tensor([50], dtype=torch.long).to(device)
noisy_image = scheduler.add_noise(sample_image, noise, timesteps)
print("Timesteps shape before check_shapes:", timesteps.shape)
intermediate_shapes = check_shapes(noisy_image, model, timesteps)
print("Intermediate shapes:", intermediate_shapes)
