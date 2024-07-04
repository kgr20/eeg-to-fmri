import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import torchmetrics
import os
import datetime
import wandb
from datetime import datetime

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset
dataset_name = "01"

# Load the data
data_path = f'/content/drive/MyDrive/{dataset_name}_eeg_fmri_data.h5'

with h5py.File(data_path, 'r') as f:
    eeg_train = np.array(f['eeg_train'][:])
    fmri_train = np.array(f['fmri_train'][:])
    eeg_test = np.array(f['eeg_test'][:])
    fmri_test = np.array(f['fmri_test'][:])

# Normalize the data
eeg_train = eeg_train / np.max(eeg_train)
fmri_train = fmri_train / np.max(fmri_train)
eeg_test = eeg_test / np.max(eeg_test)
fmri_test = fmri_test / np.max(fmri_test)

# Further normalize training and testing data
eeg_train = (eeg_train - np.min(eeg_train)) / (np.max(eeg_train) - np.min(eeg_train))
fmri_train = (fmri_train - np.min(fmri_train)) / (np.max(fmri_train) - np.min(fmri_train))
eeg_test = (eeg_test - np.min(eeg_test)) / (np.max(eeg_test) - np.min(eeg_test))
fmri_test = (fmri_test - np.min(fmri_test)) / (np.max(fmri_test) - np.min(fmri_test))

# Create a dataset class
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

print("EEG Train Shape:", eeg_train.shape)
print("fMRI Train Shape:", fmri_train.shape)
print("EEG Test Shape:", eeg_test.shape)
print("fMRI Test Shape:", fmri_test.shape)

# For easy access
num_epochs = 200

# Number of observations to include from the training and test sets (including all for NODDI here)
train_subset_size = 3731  
test_subset_size = 861

# Create new dataset and dataloader for the subset and test set
eeg_train_subset = eeg_train[:train_subset_size]
fmri_train_subset = fmri_train[:train_subset_size]
eeg_test_subset = eeg_test[:test_subset_size]
fmri_test_subset = fmri_test[:test_subset_size]

train_dataset = EEGfMRIDataset(torch.tensor(eeg_train_subset, dtype=torch.float32), torch.tensor(fmri_train_subset, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = EEGfMRIDataset(torch.tensor(eeg_test_subset, dtype=torch.float32), torch.tensor(fmri_test_subset, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the deeper and wider model
class DeeperWiderConvAutoencoder(nn.Module):
    def __init__(self):
        super(DeeperWiderConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(True),
            nn.Conv3d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(2048),
            nn.ReLU(True),
            nn.Dropout3d(0.5)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(True),
            nn.ConvTranspose3d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(size=(64, 64, 28), mode='trilinear', align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
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

# Initialize a new W&B run with the current timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
run = wandb.init(project="eeg_fmri_project", name=f"dataset_{dataset_name}_run_{timestamp}")

# Initialize the model
model = DeeperWiderConvAutoencoder().to(device)

# Define the loss function
criterion = SSIMLoss().to(device)

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Gradient clipping
clip_value = 1.0

# Training loop with timing and SSIM tracking
num_epochs = num_epochs
total_training_time = 0.0
best_ssim = -1.0
best_psnr = -1.0
best_model_weights = None

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.permute(0, 4, 1, 2, 3)[:, :, :, :, :28]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        running_loss += loss.item()

    scheduler.step(running_loss)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_training_time += epoch_duration

    model.eval()
    ssim_score = 0.0
    psnr_score = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.permute(0, 4, 1, 2, 3)[:, :, :, :, :28]
            outputs = model(inputs)
            ssim_score += 1 - criterion(outputs, labels).item()
            psnr_score += calculate_psnr(outputs, labels)

    ssim_score /= len(test_loader)
    psnr_score /= len(test_loader)

    run.log({
        "epoch": epoch + 1,
        "loss": running_loss / len(train_loader),
        "ssim": ssim_score,
        "psnr": psnr_score,
        "epoch_duration": epoch_duration,
        "total_training_time": total_training_time / 60,
    })

    if ssim_score > best_ssim:
        best_ssim = ssim_score
        best_psnr = psnr_score
        best_model_weights = model.state_dict()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f} dB, Time: {epoch_duration:.2f} sec, Total Time: {total_training_time/60:.2f} min')

# Save the best model weights with SSIM and PSNR scores in the filename
torch.save(best_model_weights, f'/content/drive/MyDrive/Models/{dataset_name}_model_best_ssim_{best_ssim:.4f}_psnr_{best_psnr:.2f}.pth')

def plot_comparison(labels, outputs, slice_idx, depth_idx, output_dir='/content/drive/MyDrive/output_images'):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    depth_idx = min(depth_idx, labels.shape[-1] - 1)

    label_slice = labels[slice_idx, 0, :, :, depth_idx]
    output_slice = outputs[slice_idx, 0, :, :, depth_idx]
    diff_slice = np.abs(label_slice - output_slice)

    axes[0].imshow(label_slice, cmap='gray')
    axes[0].set_title('Ground Truth')

    axes[1].imshow(output_slice, cmap='gray')
    axes[1].set_title('Generated Output')

    axes[2].imshow(diff_slice, cmap='gray')
    axes[2].set_title('Difference')

    plt.suptitle(f'Depth Index: {depth_idx}')
    current_time = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    filename = os.path.join(output_dir, f'{current_time}_comparison_slice{slice_idx}_depth{depth_idx}.png')
    plt.savefig(filename)
    plt.close()

    run.log({"comparison_image": wandb.Image(filename)})

model.eval()
test_batch = next(iter(test_loader))
inputs, labels = test_batch
inputs, labels = inputs.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(inputs)

inputs_np = inputs.cpu().numpy()
labels_np = labels.cpu().numpy()
outputs_np = outputs.cpu().numpy()

labels_np = np.transpose(labels_np, (0, 4, 1, 2, 3))

plot_comparison(labels_np, outputs_np, slice_idx=0, depth_idx=16)

print('Finished')

run.finish()