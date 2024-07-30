import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data (example shapes based on your description)
# Assuming data is loaded as numpy arrays
import numpy as np
eeg_train = np.random.rand(3731, 64, 269, 10)
fmri_train = np.random.rand(3731, 64, 64, 30)
eeg_test = np.random.rand(861, 64, 269, 10)
fmri_test = np.random.rand(861, 64, 64, 30)

# Convert to PyTorch tensors
eeg_train_tensor = torch.tensor(eeg_train, dtype=torch.float32)
fmri_train_tensor = torch.tensor(fmri_train, dtype=torch.float32)
eeg_test_tensor = torch.tensor(eeg_test, dtype=torch.float32)
fmri_test_tensor = torch.tensor(fmri_test, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(eeg_train_tensor, fmri_train_tensor)
test_dataset = TensorDataset(eeg_test_tensor, fmri_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Conv2d layer: input channels = 64, output channels = 128, kernel size = 3, stride = 2, padding = 1
        # This layer reduces the height and width from 269x10 to 135x5
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Conv2d layer: input channels = 128, output channels = 256, kernel size = 3, stride = 2, padding = 1
        # This layer reduces the height and width from 135x5 to 68x3
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # Fully connected layer to map from the flattened convolutional output to the latent space
        # The input dimension 256 * 68 * 3 is the flattened size of the previous layer's output
        self.fc = nn.Linear(256 * 68 * 3, 1024)  # Adjust the dimensions as per your data shape

    def forward(self, x):
        print("Input to encoder:", x.shape)
        x = torch.relu(self.conv1(x))
        print("After conv1:", x.shape)
        x = torch.relu(self.conv2(x))
        print("After conv2:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        print("After flattening:", x.shape)
        x = torch.relu(self.fc(x))
        print("After fc:", x.shape)
        return x

encoder = Encoder().to(device)

class DiffusionProcess(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionProcess, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, x, noise_level):
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        print("After diffusion process:", x_noisy.shape)
        return x_noisy

diffusion_process = DiffusionProcess(latent_dim=1024).to(device)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Fully connected layer to map from the latent space to the dimensions suitable for deconvolution
        self.fc = nn.Linear(1024, 256 * 8 * 8)
        
        # ConvTranspose2d layer: input channels = 256, output channels = 128, kernel size = 3, stride = 2, padding = 1, output_padding = 1
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # ConvTranspose2d layer: input channels = 128, output channels = 64, kernel size = 3, stride = 2, padding = 1, output_padding=1
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # ConvTranspose2d layer: input channels = 64, output channels = 32, kernel size = 3, stride = 2, padding = 1, output_padding=1
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # ConvTranspose2d layer: input channels = 32, output channels = 30, kernel size = 3, stride = 1, padding = 1
        self.deconv4 = nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1)
        
        # Conv2d layer: input channels = 30, output channels = 64, kernel size = 1, stride = 1, padding = 0
        self.final_conv = nn.Conv2d(30, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        print("After fc:", x.shape)
        x = x.view(x.size(0), 256, 8, 8)  # Reshape to the expected dimensions for the deconvolution
        print("After reshaping:", x.shape)
        x = torch.relu(self.deconv1(x))
        print("After deconv1:", x.shape)
        x = torch.relu(self.deconv2(x))
        print("After deconv2:", x.shape)
        x = torch.relu(self.deconv3(x))
        print("After deconv3:", x.shape)
        x = torch.sigmoid(self.deconv4(x))
        print("After deconv4:", x.shape)
        x = x.permute(0, 2, 3, 1)  # Permute dimensions to [batch_size, 64, 64, 30]
        print("After permutation:", x.shape)
        return x

decoder = Decoder().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(diffusion_process.parameters()) + list(decoder.parameters()), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    encoder, diffusion_process, decoder = model
    encoder.train()
    diffusion_process.train()
    decoder.train()
    
    for epoch in range(10):  # Number of epochs
        for eeg, fmri in train_loader:
            eeg, fmri = eeg.to(device), fmri.to(device)
            
            # Forward pass
            latent = encoder(eeg)
            noisy_latent = diffusion_process(latent, noise_level=0.1)
            output = decoder(noisy_latent)
            
            # Compute loss
            loss = criterion(output, fmri)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Train the model
model = (encoder, diffusion_process, decoder)
train(model, train_loader, criterion, optimizer, device)

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    encoder, diffusion_process, decoder = model
    encoder.eval()
    diffusion_process.eval()
    decoder.eval()
    
    with torch.no_grad():
        total_loss = 0
        for eeg, fmri in test_loader:
            eeg, fmri = eeg.to(device), fmri.to(device)
            
            # Forward pass
            latent = encoder(eeg)
            noisy_latent = diffusion_process(latent, noise_level=0.1)
            output = decoder(noisy_latent)
            
            # Compute loss
            loss = criterion(output, fmri)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}')

# Evaluate the model
evaluate(model, test_loader, criterion, device)
