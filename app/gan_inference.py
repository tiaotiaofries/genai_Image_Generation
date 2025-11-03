"""
GAN Inference Module for FastAPI Integration
Assignment 3 - Provides GAN digit generation for API endpoints
"""
import torch
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from gan_model import GAN, load_trained_model
import os


class GANInference:
    """
    GAN inference class for generating MNIST digits via API
    """
    
    def __init__(self, model_path='./models/gan_mnist_final.pth'):
        """
        Initialize GAN inference
        
        Args:
            model_path: Path to trained GAN model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.gan = None
        self.is_loaded = False
        
        # Try to load model if it exists
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"⚠️ Model not found at {model_path}. Train model first!")
    
    def load_model(self):
        """Load the trained GAN model"""
        try:
            self.gan = load_trained_model(self.model_path, self.device)
            self.is_loaded = True
            print(f"✅ GAN model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load GAN model: {e}")
            self.is_loaded = False
    
    def generate_digits(self, num_digits=10, return_format='base64'):
        """
        Generate handwritten digits using the trained GAN
        
        Args:
            num_digits: Number of digits to generate (1-64)
            return_format: 'base64', 'pil', 'numpy', or 'grid'
            
        Returns:
            Generated digits in requested format
        """
        if not self.is_loaded:
            raise RuntimeError("GAN model not loaded. Train model first!")
        
        if not (1 <= num_digits <= 64):
            raise ValueError("num_digits must be between 1 and 64")
        
        # Generate digits
        self.gan.generator.eval()
        with torch.no_grad():
            noise = self.gan.generate_noise(num_digits)
            generated_images = self.gan.generator(noise)
            
            # Convert to numpy and denormalize
            images = generated_images.cpu().numpy()
            images = (images + 1) / 2  # Denormalize from [-1,1] to [0,1]
            images = np.clip(images, 0, 1)  # Ensure valid range
        
        if return_format == 'numpy':
            return images
        elif return_format == 'pil':
            return [self._numpy_to_pil(img[0]) for img in images]
        elif return_format == 'base64':
            return [self._numpy_to_base64(img[0]) for img in images]
        elif return_format == 'grid':
            return self._create_image_grid(images, num_digits)
        else:
            raise ValueError("return_format must be 'base64', 'pil', 'numpy', or 'grid'")
    
    def generate_single_digit(self, return_base64=True):
        """
        Generate a single digit
        
        Args:
            return_base64: If True, return base64 string, else PIL Image
            
        Returns:
            Single generated digit
        """
        digits = self.generate_digits(num_digits=1, 
                                     return_format='base64' if return_base64 else 'pil')
        return digits[0]
    
    def generate_digit_grid(self, grid_size=4):
        """
        Generate a grid of digits for display
        
        Args:
            grid_size: Size of grid (grid_size x grid_size digits)
            
        Returns:
            PIL Image of digit grid
        """
        num_digits = grid_size * grid_size
        return self.generate_digits(num_digits=num_digits, return_format='grid')
    
    def _numpy_to_pil(self, img_array):
        """Convert numpy array to PIL Image"""
        # Convert to uint8
        img_uint8 = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_uint8, mode='L')
    
    def _numpy_to_base64(self, img_array):
        """Convert numpy array to base64 string"""
        pil_img = self._numpy_to_pil(img_array)
        
        # Convert PIL to base64
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_base64
    
    def _create_image_grid(self, images, num_digits):
        """Create a grid of images"""
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_digits)))
        
        # Create matplotlib figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(grid_size * grid_size):
            if i < num_digits:
                axes[i].imshow(images[i, 0], cmap='gray')
                axes[i].set_title(f'Digit {i+1}')
            else:
                axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        return Image.open(img_buffer)
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        info = self.gan.get_model_info()
        info.update({
            "status": "Model loaded",
            "model_path": self.model_path,
            "can_generate": True
        })
        return info


# Global inference instance
gan_inference = GANInference()


def test_gan_inference():
    """Test GAN inference functionality"""
    print("🧪 Testing GAN Inference...")
    
    if not gan_inference.is_loaded:
        print("❌ No trained model available. Run training first!")
        return False
    
    # Test single digit generation
    print("🎨 Generating single digit...")
    try:
        digit_base64 = gan_inference.generate_single_digit(return_base64=True)
        print(f"   ✅ Generated single digit (base64 length: {len(digit_base64)})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test multiple digits
    print("🎨 Generating multiple digits...")
    try:
        digits = gan_inference.generate_digits(num_digits=5, return_format='base64')
        print(f"   ✅ Generated {len(digits)} digits")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test grid generation
    print("🎨 Generating digit grid...")
    try:
        grid_img = gan_inference.generate_digit_grid(grid_size=3)
        print(f"   ✅ Generated grid image: {grid_img.size}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test model info
    print("ℹ️ Model info...")
    info = gan_inference.get_model_info()
    print(f"   Status: {info['status']}")
    if 'total_parameters' in info:
        print(f"   Parameters: {info['total_parameters']:,}")
    
    print("🎉 All GAN inference tests passed!")
    return True


if __name__ == "__main__":
    test_gan_inference()