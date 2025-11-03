"""
FastAPI Integration for GAN MNIST Digit Generation
Assignment 3 - Add GAN endpoints to existing Module 6 API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gan_inference import GANInference

app = FastAPI(
    title="Assignment 3 - GAN MNIST API",
    description="API for generating handwritten digits using GAN",
    version="1.0.0"
)

# Initialize GAN inference
gan_inference = GANInference()


# Pydantic models for API requests/responses
class DigitGenerationRequest(BaseModel):
    num_digits: int = Field(default=1, ge=1, le=64, description="Number of digits to generate (1-64)")
    format: str = Field(default="base64", description="Return format: 'base64' or 'grid'")


class SingleDigitResponse(BaseModel):
    digit_image: str = Field(description="Base64 encoded PNG image of generated digit")
    message: str = Field(description="Status message")


class MultipleDigitsResponse(BaseModel):
    digits: List[str] = Field(description="List of base64 encoded PNG images")
    count: int = Field(description="Number of generated digits")
    message: str = Field(description="Status message")


class GridDigitResponse(BaseModel):
    grid_image: str = Field(description="Base64 encoded PNG image of digit grid")
    grid_size: int = Field(description="Grid dimensions (grid_size x grid_size)")
    total_digits: int = Field(description="Total number of digits in grid")
    message: str = Field(description="Status message")


class ModelInfoResponse(BaseModel):
    status: str = Field(description="Model loading status")
    can_generate: bool = Field(description="Whether model can generate digits")
    total_parameters: Optional[int] = Field(description="Total number of model parameters")
    device: Optional[str] = Field(description="Device model is running on")


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Assignment 3 - GAN MNIST Digit Generation API",
        "version": "1.0.0",
        "endpoints": {
            "/generate_digit": "Generate a single handwritten digit",
            "/generate_digits": "Generate multiple handwritten digits",
            "/generate_digit_grid": "Generate a grid of handwritten digits",
            "/gan_model_info": "Get GAN model information"
        },
        "model_loaded": gan_inference.is_loaded
    }


@app.post("/generate_digit", response_model=SingleDigitResponse)
def generate_single_digit():
    """
    Generate a single handwritten digit using GAN
    
    Returns:
        JSON response with base64 encoded digit image
    """
    if not gan_inference.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="GAN model not loaded. Please train the model first."
        )
    
    try:
        digit_base64 = gan_inference.generate_single_digit(return_base64=True)
        
        return SingleDigitResponse(
            digit_image=digit_base64,
            message="Successfully generated handwritten digit"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating digit: {str(e)}")


@app.post("/generate_digits", response_model=MultipleDigitsResponse)
def generate_multiple_digits(request: DigitGenerationRequest):
    """
    Generate multiple handwritten digits using GAN
    
    Args:
        request: DigitGenerationRequest with num_digits parameter
        
    Returns:
        JSON response with list of base64 encoded digit images
    """
    if not gan_inference.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="GAN model not loaded. Please train the model first."
        )
    
    try:
        if request.format == "grid" and request.num_digits > 1:
            # Generate grid format
            grid_img = gan_inference.generate_digits(
                num_digits=request.num_digits, 
                return_format='grid'
            )
            
            # Convert PIL to base64
            import io
            import base64
            img_buffer = io.BytesIO()
            grid_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            grid_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return {
                "digits": [grid_base64],
                "count": 1,
                "message": f"Successfully generated {request.num_digits} digits in grid format"
            }
        else:
            # Generate individual digits
            digits = gan_inference.generate_digits(
                num_digits=request.num_digits,
                return_format='base64'
            )
            
            return MultipleDigitsResponse(
                digits=digits,
                count=len(digits),
                message=f"Successfully generated {len(digits)} handwritten digits"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating digits: {str(e)}")


@app.post("/generate_digit_grid", response_model=GridDigitResponse)
def generate_digit_grid(grid_size: int = Field(default=4, ge=2, le=8)):
    """
    Generate a grid of handwritten digits using GAN
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size digits)
        
    Returns:
        JSON response with base64 encoded grid image
    """
    if not gan_inference.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="GAN model not loaded. Please train the model first."
        )
    
    try:
        grid_img = gan_inference.generate_digit_grid(grid_size=grid_size)
        
        # Convert PIL to base64
        import io
        import base64
        img_buffer = io.BytesIO()
        grid_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        grid_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        total_digits = grid_size * grid_size
        
        return GridDigitResponse(
            grid_image=grid_base64,
            grid_size=grid_size,
            total_digits=total_digits,
            message=f"Successfully generated {total_digits} digits in {grid_size}x{grid_size} grid"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating digit grid: {str(e)}")


@app.get("/gan_model_info", response_model=ModelInfoResponse)
def get_gan_model_info():
    """
    Get information about the loaded GAN model
    
    Returns:
        JSON response with model information
    """
    info = gan_inference.get_model_info()
    
    return ModelInfoResponse(
        status=info.get("status", "Unknown"),
        can_generate=info.get("can_generate", False),
        total_parameters=info.get("total_parameters"),
        device=info.get("device")
    )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": gan_inference.is_loaded,
        "api_version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GAN MNIST API...")
    print("📝 Available endpoints:")
    print("   POST /generate_digit - Generate single digit")
    print("   POST /generate_digits - Generate multiple digits")
    print("   POST /generate_digit_grid - Generate digit grid")
    print("   GET /gan_model_info - Model information")
    print("   GET /health - Health check")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)