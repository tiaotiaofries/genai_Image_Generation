"""
Integration script to add GAN endpoints to existing Module 6 FastAPI
Assignment 3 - Extends the RNN API with GAN digit generation
"""
import sys
import os
import shutil

def integrate_gan_with_module6_api():
    """
    Integrate GAN endpoints with the existing Module 6 FastAPI application
    """
    print("🔧 Integrating GAN with Module 6 FastAPI...")
    
    # Paths
    current_dir = "/Users/szening/assignment3_gan_mnist"
    module6_dir = "/Users/szening/sps_genai"
    module6_app_dir = f"{module6_dir}/app"
    
    # Copy GAN model files to Module 6 project
    print("📁 Copying GAN files to Module 6 project...")
    
    # Copy GAN model implementation
    shutil.copy(f"{current_dir}/gan_model.py", f"{module6_dir}/gan_model.py")
    print("   ✅ Copied gan_model.py")
    
    # Copy trained model if it exists
    if os.path.exists(f"{current_dir}/models/gan_mnist_final.pth"):
        os.makedirs(f"{module6_dir}/models", exist_ok=True)
        shutil.copy(f"{current_dir}/models/gan_mnist_final.pth", f"{module6_dir}/models/gan_mnist_final.pth")
        print("   ✅ Copied trained GAN model")
    else:
        print("   ⚠️ No trained model found - you'll need to train first")
    
    # Create GAN inference module in Module 6 app
    gan_inference_content = '''"""
GAN Inference for Module 6 API Integration
"""
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gan_model import GAN
except ImportError:
    print("Warning: GAN model not found. Train GAN model first.")
    GAN = None


class GANInference:
    def __init__(self, model_path='./models/gan_mnist_final.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.gan = None
        self.is_loaded = False
        
        if os.path.exists(model_path) and GAN is not None:
            self.load_model()
    
    def load_model(self):
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create GAN and load state
            self.gan = GAN(noise_dim=100, device=self.device)
            self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            self.is_loaded = True
            print(f"✅ GAN model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load GAN model: {e}")
            self.is_loaded = False
    
    def generate_digits(self, num_digits=1):
        if not self.is_loaded:
            return None
        
        self.gan.generator.eval()
        with torch.no_grad():
            noise = self.gan.generate_noise(num_digits)
            generated_images = self.gan.generator(noise)
            
            # Convert to base64
            images = generated_images.cpu().numpy()
            images = (images + 1) / 2  # Denormalize
            images = np.clip(images, 0, 1)
            
            base64_images = []
            for img in images:
                img_uint8 = (img[0] * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8, mode='L')
                
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                base64_images.append(img_base64)
            
            return base64_images
    
    def get_model_info(self):
        if not self.is_loaded:
            return {"status": "GAN model not loaded"}
        
        return {
            "status": "GAN model loaded",
            "can_generate": True,
            "device": str(self.device)
        }

# Global instance
gan_inference = GANInference()
'''
    
    with open(f"{module6_app_dir}/gan_inference.py", "w") as f:
        f.write(gan_inference_content)
    print("   ✅ Created gan_inference.py in Module 6 app")
    
    # Read existing main.py
    main_py_path = f"{module6_app_dir}/main.py"
    with open(main_py_path, "r") as f:
        main_content = f.read()
    
    # Check if GAN endpoints already added
    if "generate_digit" in main_content:
        print("   ⚠️ GAN endpoints already exist in main.py")
        return
    
    # Add GAN import and endpoints
    gan_additions = '''

# GAN Integration for Assignment 3
try:
    from app.gan_inference import gan_inference
    GAN_AVAILABLE = True
except ImportError:
    print("Warning: GAN inference not available")
    GAN_AVAILABLE = False

class DigitGenerationRequest(BaseModel):
    num_digits: int = Field(default=1, ge=1, le=16, description="Number of digits to generate")

@app.post("/generate_digit")
def generate_single_digit():
    """Generate a single handwritten digit using GAN (Assignment 3)"""
    if not GAN_AVAILABLE or not gan_inference.is_loaded:
        raise HTTPException(status_code=503, detail="GAN model not available. Train model first.")
    
    try:
        digits = gan_inference.generate_digits(num_digits=1)
        if digits:
            return {"digit_image": digits[0], "message": "Generated handwritten digit"}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate digit")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/generate_digits")
def generate_multiple_digits(request: DigitGenerationRequest):
    """Generate multiple handwritten digits using GAN (Assignment 3)"""
    if not GAN_AVAILABLE or not gan_inference.is_loaded:
        raise HTTPException(status_code=503, detail="GAN model not available. Train model first.")
    
    try:
        digits = gan_inference.generate_digits(num_digits=request.num_digits)
        if digits:
            return {
                "digits": digits,
                "count": len(digits),
                "message": f"Generated {len(digits)} handwritten digits"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate digits")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/gan_model_info")
def get_gan_model_info():
    """Get GAN model information (Assignment 3)"""
    if not GAN_AVAILABLE:
        return {"status": "GAN not available", "can_generate": False}
    
    return gan_inference.get_model_info()
'''
    
    # Update root endpoint to include GAN endpoints
    updated_main = main_content.replace(
        'return {"status": "ok", "endpoints": ["/generate", "/generate_with_rnn", "/embed"]}',
        'return {"status": "ok", "endpoints": ["/generate", "/generate_with_rnn", "/embed", "/generate_digit", "/generate_digits", "/gan_model_info"]}'
    )
    
    # Add GAN code before the last line
    lines = updated_main.split('\n')
    insert_index = len(lines) - 1  # Before the last line
    gan_lines = gan_additions.split('\n')
    
    # Insert GAN code
    for i, line in enumerate(gan_lines):
        lines.insert(insert_index + i, line)
    
    updated_main = '\n'.join(lines)
    
    # Write updated main.py
    with open(main_py_path, "w") as f:
        f.write(updated_main)
    print("   ✅ Updated main.py with GAN endpoints")
    
    print("🎉 Integration completed!")
    print("\n📋 New endpoints added to Module 6 API:")
    print("   POST /generate_digit - Generate single handwritten digit")
    print("   POST /generate_digits - Generate multiple handwritten digits")
    print("   GET /gan_model_info - Get GAN model information")
    
    print("\n🐳 Restart Docker container to apply changes:")
    print("   docker stop sps-genai")
    print("   docker run -d --rm --name sps-genai -p 8000:80 sps-genai-rnn")


def create_test_script():
    """Create a test script for the integrated API"""
    test_content = '''#!/usr/bin/env python3
"""
Test script for integrated Module 6 + GAN API
Assignment 3 - Test all endpoints including new GAN functionality
"""
import requests
import json
import time

def test_integrated_api(base_url="http://localhost:8000"):
    """Test the integrated API with GAN endpoints"""
    print("🧪 Testing Module 6 + GAN Integrated API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\\n1️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Endpoints: {data['endpoints']}")
            
            if "/generate_digit" in data['endpoints']:
                print("   ✅ GAN endpoints found!")
            else:
                print("   ⚠️ GAN endpoints not found")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Original RNN endpoint
    print("\\n2️⃣ Testing RNN endpoint...")
    try:
        response = requests.post(f"{base_url}/generate_with_rnn", json={
            "start_word": "the count",
            "length": 10
        })
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ RNN generation: '{data['generated_text'][:50]}...'")
        else:
            print(f"   ❌ RNN Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ RNN Error: {e}")
    
    # Test 3: GAN model info
    print("\\n3️⃣ Testing GAN model info...")
    try:
        response = requests.get(f"{base_url}/gan_model_info")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   GAN Status: {data.get('status', 'Unknown')}")
            print(f"   Can Generate: {data.get('can_generate', False)}")
            
            if data.get('can_generate', False):
                print("   ✅ GAN model ready!")
            else:
                print("   ⚠️ GAN model not ready - train first")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Single digit generation
    print("\\n4️⃣ Testing single digit generation...")
    try:
        response = requests.post(f"{base_url}/generate_digit")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Generated digit (base64 length: {len(data['digit_image'])})")
            print(f"   Message: {data['message']}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Multiple digits generation
    print("\\n5️⃣ Testing multiple digits generation...")
    try:
        response = requests.post(f"{base_url}/generate_digits", json={
            "num_digits": 3
        })
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Generated {data['count']} digits")
            print(f"   Message: {data['message']}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\\n🎉 API testing completed!")

if __name__ == "__main__":
    test_integrated_api()
'''
    
    with open("/Users/szening/assignment3_gan_mnist/test_integrated_api.py", "w") as f:
        f.write(test_content)
    print("✅ Created test script: test_integrated_api.py")


if __name__ == "__main__":
    print("🚀 Assignment 3 - GAN Integration with Module 6 API")
    print("This script will integrate GAN endpoints with the existing RNN API")
    
    choice = input("\\nProceed with integration? (y/n): ").strip().lower()
    if choice == 'y':
        integrate_gan_with_module6_api()
        create_test_script()
        
        print("\\n📋 Next steps:")
        print("1. Train the GAN model: cd assignment3_gan_mnist && python train_gan.py")
        print("2. Rebuild Docker image: cd /Users/szening/sps_genai && docker build -t sps-genai-rnn .")
        print("3. Restart container: docker stop sps-genai && docker run -d --rm --name sps-genai -p 8000:80 sps-genai-rnn")
        print("4. Test API: python test_integrated_api.py")
    else:
        print("Integration cancelled.")