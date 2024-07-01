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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
data_path = '/home/aca10131kr/datasets/01_eeg_fmri_data.h5'
with h5py.File(data_path, 'r') as f:
    eeg_data = np.array(f['eeg_train'][:])
    fmri_data = np.array(f['fmri_train'][:])

# Normalize the data
eeg_data = eeg_data / np.max(eeg_data)
fmri_data = fmri_data / np.max(fmri_data)

# Split the data into training and testing sets
eeg_train, eeg_test, fmri_train, fmri_test = train_test_split(eeg_data, fmri_data, test_size=0.2, random_state=42)

# Further normalize training and testing data
eeg_train = (eeg_train - np.min(eeg_train)) / (np.max(eeg_train) - np.min(eeg_train))
fmri_train = (fmri_train - np.min(fmri_train)) / (np.max(fmri_train) - np.min(fmri_train))
eeg_test = (eeg_test - np.min(eeg_test)) / (np.max(eeg_test) - np.min(eeg_test))
fmri_test = (fmri_test - np.min(fmri_test)) / (np.max(fmri_test) - np.min(fmri_test))

# Select a small subset of the data (e.g., 200 samples)
subset_size = 200
eeg_subset = eeg_train[:subset_size]
fmri_subset = fmri_train[:subset_size]

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

# Create new dataset and dataloader for the subset
subset_dataset = EEGfMRIDataset(torch.tensor(eeg_subset, dtype=torch.float32), torch.tensor(fmri_subset, dtype=torch.float32))
subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

# Define the deeper and wider model
class DeeperWiderConvAutoencoder(nn.Module):
    def __init__(self):
        super(DeeperWiderConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 32, 135, 5)
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, 16, 68, 3)
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, 8, 34, 2)
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),  # (B, 512, 4, 17, 1)
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1), # (B, 1024, 2, 9, 1)
            nn.BatchNorm3d(1024),
            nn.ReLU(True),
            nn.Conv3d(1024, 2048, kernel_size=3, stride=2, padding=1), # (B, 2048, 1, 5, 1)
            nn.BatchNorm3d(2048),
            nn.ReLU(True),
            nn.Dropout3d(0.5)  # Dropout layer
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1024, 2, 9, 1)
            nn.BatchNorm3d(1024),
            nn.ReLU(True),
            nn.ConvTranspose3d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 512, 4, 17, 1)
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 256, 8, 34, 2)
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, 16, 68, 3)
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, 32, 135, 5)
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, 64, 270, 10)
            nn.Upsample(size=(64, 64, 30), mode='trilinear', align_corners=True),  # Ensure final output size matches
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # Change shape from (B, 64, 269, 10, 1) to (B, 1, 64, 269, 10)
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
num_epochs = 20  # Reduce the number of epochs
total_training_time = 0.0
best_ssim = -1.0
best_model_weights = None

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    for inputs, labels in subset_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Permute labels to match the shape of outputs
        labels = labels.permute(0, 4, 1, 2, 3)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)  # Use SSIM loss directly
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        running_loss += loss.item()

    scheduler.step(running_loss)  # Adjust learning rate based on the running loss

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_training_time += epoch_duration

    # Calculate SSIM and PSNR score for the epoch
    model.eval()
    ssim_score = 0.0
    psnr_score = 0.0
    with torch.no_grad():
        for inputs, labels in subset_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.permute(0, 4, 1, 2, 3)
            outputs = model(inputs)
            ssim_score += 1 - criterion(outputs, labels).item()
            psnr_score += calculate_psnr(outputs, labels)

    ssim_score /= len(subset_loader)
    psnr_score /= len(subset_loader)

    # Save the model weights if the SSIM score improves
    if ssim_score > best_ssim:
        best_ssim = ssim_score
        best_model_weights = model.state_dict()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(subset_loader):.4f}, SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f} dB, Time: {epoch_duration:.2f} sec, Total Time: {total_training_time/60:.2f} min')

# Save the best model weights
torch.save(best_model_weights, '/content/drive/MyDrive/model_best_ssim.pth')

# Function to plot comparison between ground truth and generated output
def plot_comparison(labels, outputs, slice_idx, depth_idx, output_dir='/content/drive/MyDrive/output_images'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get specific slice and depth
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
    plt.savefig(os.path.join(output_dir, f'comparison_slice{slice_idx}_depth{depth_idx}.png'))
    plt.close()

# Put the model in evaluation mode
model.eval()

# Get a batch of test inputs
test_batch = next(iter(subset_loader))
inputs, labels = test_batch
inputs, labels = inputs.to(device), labels.to(device)

# Generate outputs
with torch.no_grad():
    outputs = model(inputs)

# Convert to NumPy arrays
inputs_np = inputs.cpu().numpy()
labels_np = labels.cpu().numpy()
outputs_np = outputs.cpu().numpy()

# Permute labels to match the shape of outputs
labels_np = np.transpose(labels_np, (0, 4, 1, 2, 3))

# Plot a comparison for a specific slice and depth, and save the images
plot_comparison(labels_np, outputs_np, slice_idx=0, depth_idx=16)

print('Finished Training')
