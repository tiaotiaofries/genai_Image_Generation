"""
GAN Model Implementation for MNIST Digit Generation
Assignment 3 - Implements the specified Generator and Discriminator architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    """
    Generator network for GAN
    
    Architecture as specified in assignment:
    - Input: Noise vector of shape (BATCH_SIZE, 100)
    - Fully connected layer to 7 × 7 × 128, then reshape
    - ConvTranspose2D: 128 → 64, kernel size 4, stride 2, padding 1 → output size 14 × 14
    - Followed by BatchNorm2D and ReLU
    - ConvTranspose2D: 64 → 1, kernel size 4, stride 2, padding 1 → output size 28 × 28
    - Followed by Tanh activation
    """
    
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        # Fully connected layer to reshape to 7x7x128
        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)
        
        # ConvTranspose layers as specified
        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )  # Output: 14x14
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        
        self.conv_transpose2 = nn.ConvTranspose2d(
            in_channels=64, 
            out_channels=1, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )  # Output: 28x28
        
    def forward(self, noise):
        """
        Forward pass of generator
        
        Args:
            noise: Input noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated images of shape (batch_size, 1, 28, 28)
        """
        # Pass through fully connected layer and reshape
        x = self.fc(noise)  # (batch_size, 7*7*128)
        x = x.view(-1, 128, 7, 7)  # Reshape to (batch_size, 128, 7, 7)
        
        # First ConvTranspose + BatchNorm + ReLU
        x = self.conv_transpose1(x)  # (batch_size, 64, 14, 14)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        # Second ConvTranspose + Tanh
        x = self.conv_transpose2(x)  # (batch_size, 1, 28, 28)
        x = torch.tanh(x)
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for GAN
    
    Architecture as specified in assignment:
    - Input: Image of shape (1, 28, 28)
    - Conv2D: 1 → 64, kernel size 4, stride 2, padding 1 → output size 14 × 14
    - Followed by LeakyReLU(0.2)
    - Conv2D: 64 → 128, kernel size 4, stride 2, padding 1 → output size 7 × 7
    - Followed by BatchNorm2D and LeakyReLU(0.2)
    - Flatten and apply Linear layer to get a single output (real/fake probability)
    """
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Conv layers as specified
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )  # Output: 14x14
        
        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )  # Output: 7x7
        
        self.batch_norm2 = nn.BatchNorm2d(128)
        
        # Linear layer for final classification
        self.fc = nn.Linear(128 * 7 * 7, 1)
        
    def forward(self, img):
        """
        Forward pass of discriminator
        
        Args:
            img: Input image tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Probability tensor of shape (batch_size, 1) indicating real/fake
        """
        # First Conv + LeakyReLU
        x = self.conv1(img)  # (batch_size, 64, 14, 14)
        x = F.leaky_relu(x, 0.2)
        
        # Second Conv + BatchNorm + LeakyReLU
        x = self.conv2(x)  # (batch_size, 128, 7, 7)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x, 0.2)
        
        # Flatten and apply linear layer
        x = x.view(-1, 128 * 7 * 7)  # Flatten
        x = self.fc(x)  # (batch_size, 1)
        
        return x


class GAN:
    """
    Complete GAN model combining Generator and Discriminator
    """
    
    def __init__(self, noise_dim=100, device='cpu'):
        self.device = device
        self.noise_dim = noise_dim
        
        # Initialize networks
        self.generator = Generator(noise_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        # Initialize optimizers (will be set during training)
        self.g_optimizer = None
        self.d_optimizer = None
        
    def generate_noise(self, batch_size):
        """Generate random noise for the generator"""
        return torch.randn(batch_size, self.noise_dim, device=self.device)
    
    def generate_images(self, num_images=1):
        """
        Generate images using the trained generator
        
        Args:
            num_images: Number of images to generate
            
        Returns:
            Generated images tensor of shape (num_images, 1, 28, 28)
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self.generate_noise(num_images)
            generated_images = self.generator(noise)
        return generated_images
    
    def get_model_info(self):
        """Get information about the GAN model"""
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        
        return {
            "generator_parameters": g_params,
            "discriminator_parameters": d_params,
            "total_parameters": g_params + d_params,
            "noise_dimension": self.noise_dim,
            "output_size": "28x28",
            "device": str(self.device)
        }


def test_gan_architecture():
    """Test the GAN architecture with dummy data"""
    print("🧪 Testing GAN Architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test Generator
    print("\n🎨 Testing Generator...")
    generator = Generator(noise_dim=100).to(device)
    noise = torch.randn(4, 100).to(device)  # Batch of 4
    generated_images = generator(noise)
    print(f"   Input noise shape: {noise.shape}")
    print(f"   Generated images shape: {generated_images.shape}")
    print(f"   Generated images range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
    assert generated_images.shape == (4, 1, 28, 28), f"Expected (4, 1, 28, 28), got {generated_images.shape}"
    print("   ✅ Generator architecture correct!")
    
    # Test Discriminator
    print("\n🔍 Testing Discriminator...")
    discriminator = Discriminator().to(device)
    real_images = torch.randn(4, 1, 28, 28).to(device)  # Batch of 4
    predictions = discriminator(real_images)
    print(f"   Input images shape: {real_images.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    assert predictions.shape == (4, 1), f"Expected (4, 1), got {predictions.shape}"
    print("   ✅ Discriminator architecture correct!")
    
    # Test complete GAN
    print("\n🤖 Testing Complete GAN...")
    gan = GAN(noise_dim=100, device=device)
    generated = gan.generate_images(num_images=3)
    print(f"   Generated images shape: {generated.shape}")
    model_info = gan.get_model_info()
    print(f"   Generator parameters: {model_info['generator_parameters']:,}")
    print(f"   Discriminator parameters: {model_info['discriminator_parameters']:,}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print("   ✅ Complete GAN working!")
    
    print(f"\n🎉 All architecture tests passed!")
    return True


if __name__ == "__main__":
    test_gan_architecture()