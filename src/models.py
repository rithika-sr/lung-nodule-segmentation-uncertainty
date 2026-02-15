"""
U-Net Model Architecture for 3D Medical Image Segmentation
Includes dropout layers for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv3D -> BatchNorm -> ReLU -> Conv3D -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Concatenate with skip connection -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x1, x2):
        # x1: upsampled features, x2: skip connection from encoder
        x1 = self.up(x1)
        
        # Handle size mismatch (if any)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for medical image segmentation with uncertainty quantification
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32, dropout_rate=0.2):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale CT)
            out_channels: Number of output channels (1 for binary segmentation)
            init_features: Number of features in first layer
            dropout_rate: Dropout probability for uncertainty estimation
        """
        super(UNet3D, self).__init__()
        
        features = init_features
        
        # Encoder (downsampling path)
        self.encoder1 = DoubleConv(in_channels, features, dropout_rate)
        self.pool1 = Down(features, features * 2, dropout_rate)
        self.pool2 = Down(features * 2, features * 4, dropout_rate)
        self.pool3 = Down(features * 4, features * 8, dropout_rate)
        
        # Bottleneck
        self.bottleneck = Down(features * 8, features * 16, dropout_rate)
        
        # Decoder (upsampling path)
        self.upconv4 = Up(features * 16, features * 8, dropout_rate)
        self.upconv3 = Up(features * 8, features * 4, dropout_rate)
        self.upconv2 = Up(features * 4, features * 2, dropout_rate)
        self.upconv1 = Up(features * 2, features, dropout_rate)
        
        # Output layer
        self.out = nn.Conv3d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.pool1(enc1)
        enc3 = self.pool2(enc2)
        enc4 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)
        
        # Output
        out = self.out(dec1)
        return torch.sigmoid(out)  # Binary segmentation
    
    def enable_dropout(self):
        """
        Enable dropout during inference for Monte Carlo uncertainty estimation
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout3d):
                module.train()


def get_model(in_channels=1, out_channels=1, init_features=32, dropout_rate=0.2):
    """
    Factory function to create U-Net model
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        init_features: Initial feature maps
        dropout_rate: Dropout probability
        
    Returns:
        model: UNet3D instance
    """
    model = UNet3D(in_channels, out_channels, init_features, dropout_rate)
    return model


def count_parameters(model):
    """
    Count trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing 3D U-Net Model...")
    
    model = get_model(in_channels=1, out_channels=1, init_features=16, dropout_rate=0.2)
    
    # Create dummy input (batch_size=2, channels=1, depth=64, height=64, width=64)
    x = torch.randn(2, 1, 64, 64, 64)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    print("\n Model architecture created successfully!")