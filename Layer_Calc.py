import torch
import torch.nn as nn
import numpy as np

class SimpleDiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleDiffusionModel, self).__init__()
        
        # Encoder (Downsampling path)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (Upsampling path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        print(f"After encoder1: {enc1.shape}")
        enc2 = self.encoder2(self.pool1(enc1))
        print(f"After encoder2: {enc2.shape}")
        enc3 = self.encoder3(self.pool2(enc2))
        print(f"After encoder3: {enc3.shape}")
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        print(f"After bottleneck: {bottleneck.shape}")
        
        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        print(f"After decoder3: {dec3.shape}")
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        print(f"After decoder2: {dec2.shape}")
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        print(f"After decoder1: {dec1.shape}")
        
        final_output = self.final_conv(dec1)
        print(f"After final_conv: {final_output.shape}")
        
        return final_output

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDiffusionModel(in_channels=10, out_channels=30).to(device)

# Test the model with a sample input
input_shape = (1, 10, 64, 269)  # (batch_size, channels, height, width)
sample_input = torch.randn(input_shape).to(device)

# Forward pass through the model to check dimensions
output = model(sample_input)
print(f"Output shape: {output.shape}")
