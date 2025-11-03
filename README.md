# Assignment 3: Image Generation

## Overview
This repository contains the implementation of a Generative Adversarial Network (GAN) using the MNIST dataset.

## Project Structure
```
genai_Image_Generation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── gan_model.py                 # GAN architecture implementation
├── train_gan.py                 # Training script
├── app/                         # FastAPI integration
│   ├── main.py                  # API endpoints
│   └── gan_inference.py         # GAN model inference
├── models/                      # Trained model files
└── tests/                       # Test scripts
```

## Assignment Outline

### GAN Architecture Implementation
Implementation of GAN with specified architecture:

**Generator:**
- Input: Noise vector (BATCH_SIZE, 100)
- Fully connected layer to 7×7×128, then reshape
- ConvTranspose2D: 128→64, kernel=4, stride=2, padding=1 → output 14×14
- ConvTranspose2D: 64→1, kernel=4, stride=2, padding=1 → output 28×28
- BatchNorm2D and ReLU activations, final Tanh activation

**Discriminator:**
- Input: Image (1, 28, 28)
- Conv2D: 1→64, kernel=4, stride=2, padding=1 → output 14×14
- Conv2D: 64→128, kernel=4, stride=2, padding=1 → output 7×7
- LeakyReLU(0.2) activations, BatchNorm2D
- Flatten and Linear layer for single output (real/fake probability)

### Model Deployment
- Train GAN on MNIST dataset
- Integrate with FastAPI from Module 6 
- Add endpoint for digit generation

## Installation & Usage

### Setup Environment
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train_gan.py
```

### Run API Server
```bash
cd app
uvicorn main:app --reload --port 8001
```

### Test Digit Generation
```bash
curl -X POST http://localhost:8001/generate_digits \
  -H 'Content-Type: application/json' \
  -d '{"num_digits": 10}'
```
