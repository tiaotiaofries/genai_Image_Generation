"""
Training script for GAN on MNIST dataset
Assignment 3 - Train GAN to generate handwritten digits
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from gan_model import GAN, Generator, Discriminator


def setup_mnist_dataset(batch_size=64, data_dir='./data'):
    """
    Setup MNIST dataset for training
    
    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load MNIST data
        
    Returns:
        DataLoader for MNIST training data
    """
    print(f"📦 Setting up MNIST dataset...")
    
    # MNIST preprocessing - normalize to [-1, 1] to match Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Download and load MNIST dataset
    mnist_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    print(f"   ✅ MNIST dataset loaded: {len(mnist_dataset)} samples")
    print(f"   📊 Batch size: {batch_size}")
    print(f"   🔢 Number of batches: {len(dataloader)}")
    
    return dataloader


def train_gan(epochs=50, batch_size=64, learning_rate=0.0002, save_interval=10):
    """
    Train the GAN on MNIST dataset
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizers
        save_interval: Save model every N epochs
    """
    print(f"🚀 Starting GAN Training on MNIST")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Setup dataset
    dataloader = setup_mnist_dataset(batch_size)
    
    # Initialize GAN
    gan = GAN(noise_dim=100, device=device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizers (using Adam as commonly used for GANs)
    g_optimizer = optim.Adam(gan.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(gan.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Training labels
    real_label = 1.0
    fake_label = 0.0
    
    # Training history
    g_losses = []
    d_losses = []
    
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    
    print(f"\n🎯 Starting Training Loop...")
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for i, (real_images, _) in enumerate(dataloader):
            batch_size_current = real_images.size(0)
            real_images = real_images.to(device)
            
            # Create labels
            real_labels = torch.full((batch_size_current, 1), real_label, device=device)
            fake_labels = torch.full((batch_size_current, 1), fake_label, device=device)
            
            # =======================
            # Train Discriminator
            # =======================
            d_optimizer.zero_grad()
            
            # Real images
            real_output = gan.discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = gan.generate_noise(batch_size_current)
            fake_images = gan.generator(noise)
            fake_output = gan.discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # =======================
            # Train Generator
            # =======================
            g_optimizer.zero_grad()
            
            # Generate fake images and get discriminator output
            fake_output = gan.discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)  # Want discriminator to think fake is real
            
            g_loss.backward()
            g_optimizer.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if i % 100 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
        
        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"✅ Epoch [{epoch+1}/{epochs}] completed - "
              f"Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
        
        # Save model and generate sample images every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_model_and_samples(gan, epoch + 1, avg_g_loss, avg_d_loss)
    
    # Save final model
    save_model_and_samples(gan, epochs, g_losses[-1], d_losses[-1], final=True)
    
    # Plot training losses
    plot_training_losses(g_losses, d_losses, epochs)
    
    print(f"\n🎉 Training completed!")
    return gan, g_losses, d_losses


def save_model_and_samples(gan, epoch, g_loss, d_loss, final=False):
    """Save model and generate sample images"""
    suffix = "_final" if final else f"_epoch_{epoch}"
    
    # Save model state
    model_path = f"./models/gan_mnist{suffix}.pth"
    torch.save({
        'generator_state_dict': gan.generator.state_dict(),
        'discriminator_state_dict': gan.discriminator.state_dict(),
        'epoch': epoch,
        'g_loss': g_loss,
        'd_loss': d_loss,
        'model_info': gan.get_model_info()
    }, model_path)
    
    # Generate and save sample images
    gan.generator.eval()
    with torch.no_grad():
        # Generate 16 sample images
        sample_noise = gan.generate_noise(16)
        sample_images = gan.generator(sample_noise)
        
        # Convert to numpy and denormalize
        sample_images = sample_images.cpu().numpy()
        sample_images = (sample_images + 1) / 2  # Denormalize from [-1,1] to [0,1]
        
        # Create grid plot
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            row, col = i // 4, i % 4
            axes[row, col].imshow(sample_images[i, 0], cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle(f'Generated MNIST Digits - Epoch {epoch}')
        plt.tight_layout()
        
        # Save plot
        sample_path = f"./models/samples{suffix}.png"
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   💾 Saved model and samples: {model_path}, {sample_path}")


def plot_training_losses(g_losses, d_losses, epochs):
    """Plot training losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), g_losses, label='Generator Loss', color='blue')
    plt.plot(range(1, epochs + 1), d_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('./models/training_losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   📊 Training loss plot saved: ./models/training_losses.png")


def load_trained_model(model_path='./models/gan_mnist_final.pth', device='cpu'):
    """
    Load a trained GAN model
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded GAN model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create GAN and load state
    gan = GAN(noise_dim=100, device=device)
    gan.generator.load_state_dict(checkpoint['generator_state_dict'])
    gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    print(f"✅ Loaded trained model from epoch {checkpoint['epoch']}")
    print(f"   Final losses - G: {checkpoint['g_loss']:.4f}, D: {checkpoint['d_loss']:.4f}")
    
    return gan


def test_quick_training():
    """Quick training test with few epochs"""
    print("🧪 Testing Quick Training (5 epochs)...")
    gan, g_losses, d_losses = train_gan(epochs=5, batch_size=32, save_interval=5)
    print("✅ Quick training test completed!")
    return gan


if __name__ == "__main__":
    print("🎮 GAN Training Script for MNIST")
    print("Choose an option:")
    print("1. Quick test training (5 epochs)")
    print("2. Full training (50 epochs)")
    print("3. Custom training")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_quick_training()
    elif choice == "2":
        train_gan(epochs=50, batch_size=64, save_interval=10)
    elif choice == "3":
        epochs = int(input("Enter number of epochs: "))
        batch_size = int(input("Enter batch size (default 64): ") or 64)
        train_gan(epochs=epochs, batch_size=batch_size)
    else:
        print("Running quick test by default...")
        test_quick_training()