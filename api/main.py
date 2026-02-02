# -*- coding: utf-8 -*-
"""
AnimeGAN API - Transform photos into anime-style images

A FastAPI-based REST API for converting images to anime style.
Uses image processing techniques for anime-style transformation.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from enum import Enum
import io
import base64

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Create FastAPI app
app = FastAPI(
    title="AnimeGAN API",
    description="Transform your photos into stunning anime-style images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Available styles
class AnimeStyle(str, Enum):
    SHINKAI = "Shinkai"
    HAYAO = "Hayao"
    PAPRIKA = "Paprika"

# Enhancement types
class EnhancementType(str, Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    WITH_LINES = "with_lines"
    MAXIMUM = "maximum"


# =====================
# Image Processing Functions
# =====================

def apply_anime_face_transform(image: np.ndarray) -> np.ndarray:
    """Apply transformations to make faces more anime-like"""
    img_pil = Image.fromarray(image)

    # Detect face (if possible) and apply specific transformations
    try:
        img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
    except Exception:
        pass

    # Smooth skin areas (anime has smooth skin)
    img_array = np.array(img_pil)
    smoothed = cv2.bilateralFilter(img_array, 15, 80, 80)

    # Sharpen edges (anime has sharp lines)
    sharpened = Image.fromarray(smoothed)
    sharpened = sharpened.filter(ImageFilter.SHARPEN)
    sharpened = sharpened.filter(ImageFilter.SHARPEN)

    return np.array(sharpened)


def enhance_anime_style(image: np.ndarray) -> np.ndarray:
    """Post-process to enhance anime character look"""
    img_pil = Image.fromarray(image)

    enhancer = ImageEnhance.Color(img_pil)
    img_pil = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.3)

    enhancer = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer.enhance(2.0)

    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(1.1)

    return np.array(img_pil)


def create_anime_lines(image: np.ndarray) -> np.ndarray:
    """Create clean anime-style line art overlay"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)

    edges = cv2.bitwise_not(edges)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result = cv2.multiply(image.astype(float), edges_rgb.astype(float) / 255.0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_cartoon_effect(image: np.ndarray) -> np.ndarray:
    """Apply cartoon/anime effect using edge detection and color quantization"""
    # Convert to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    for _ in range(3):
        img_bgr = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    
    # Convert to grayscale and apply median blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 2)
    
    # Reduce colors using k-means clustering
    data = np.float32(img_bgr).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img_bgr.shape)
    
    # Combine edges with quantized colors
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized, edges_colored)
    
    # Convert back to RGB
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    
    return cartoon_rgb


def apply_style_filter(image: np.ndarray, style: str) -> np.ndarray:
    """Apply different anime style filters based on the selected style"""
    
    if style == "Shinkai":
        # Shinkai style: more realistic with beautiful colors
        # Enhance blue/cyan tones typical of Shinkai films
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.2, 0, 255).astype(np.uint8)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Add slight blue tint
        blue_overlay = np.zeros_like(result)
        blue_overlay[:, :] = [30, 60, 90]  # Light blue
        result = cv2.addWeighted(result, 0.92, blue_overlay, 0.08, 0)
        
    elif style == "Hayao":
        # Hayao/Ghibli style: soft, warm, painterly
        # Apply warm color tones
        img_pil = Image.fromarray(image)
        
        # Add warmth
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(1.3)
        
        # Soften
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        result = np.array(img_pil)
        
        # Add warm overlay
        warm_overlay = np.zeros_like(result)
        warm_overlay[:, :] = [40, 30, 20]  # Warm sepia-like
        result = cv2.addWeighted(result, 0.93, warm_overlay, 0.07, 0)
        
    elif style == "Paprika":
        # Paprika style: vibrant, high contrast, saturated colors
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)  # Boost saturation
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.15, 0, 255).astype(np.uint8)  # Boost brightness
        result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Increase contrast
        img_pil = Image.fromarray(result)
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(1.3)
        result = np.array(img_pil)
        
    else:
        result = image
        
    return np.clip(result, 0, 255).astype(np.uint8)


def transform_image(image: np.ndarray, style: str) -> dict:
    """Transform image using anime-style processing"""
    
    # Pre-process for anime style
    img_preprocessed = apply_anime_face_transform(image)
    
    # Apply cartoon effect
    cartoon = apply_cartoon_effect(img_preprocessed)
    
    # Apply style-specific filter
    styled = apply_style_filter(cartoon, style)
    
    # Generate all enhancement versions
    result_basic = styled
    result_enhanced = enhance_anime_style(styled)
    result_with_lines = create_anime_lines(result_enhanced)
    result_max = enhance_anime_style(result_with_lines)
    
    return {
        "basic": result_basic,
        "enhanced": result_enhanced,
        "with_lines": result_with_lines,
        "maximum": result_max
    }


# =====================
# API Endpoints
# =====================

@app.get("/")
async def root():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to AnimeGAN API! 🎨",
        "description": "Transform your photos into stunning anime-style images",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "transform": "POST /transform",
            "models": "GET /models",
            "health": "GET /health"
        },
        "available_styles": ["Shinkai", "Hayao", "Paprika"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AnimeGAN API"}


@app.get("/models")
async def list_models():
    """List available anime style models"""
    return {
        "styles": [
            {
                "name": "Shinkai",
                "description": "Makoto Shinkai style - realistic anime with beautiful backgrounds",
                "best_for": "Landscapes and realistic anime characters"
            },
            {
                "name": "Hayao",
                "description": "Studio Ghibli style - soft, painterly anime aesthetic",
                "best_for": "Creating Ghibli-style artworks"
            },
            {
                "name": "Paprika",
                "description": "Vibrant and colorful anime style",
                "best_for": "Creating vivid, high-contrast anime images"
            }
        ],
        "enhancement_types": [
            {
                "name": "basic",
                "description": "Direct output from the style filter"
            },
            {
                "name": "enhanced",
                "description": "Enhanced colors, contrast, and sharpness"
            },
            {
                "name": "with_lines",
                "description": "Enhanced with anime-style line art"
            },
            {
                "name": "maximum",
                "description": "Maximum anime effect with all enhancements"
            }
        ]
    }


@app.post("/transform")
async def transform_to_anime(
    file: UploadFile = File(..., description="Image file to transform (JPEG, PNG, WebP)"),
    style: AnimeStyle = Query(
        default=AnimeStyle.SHINKAI,
        description="Anime style to use"
    ),
    enhancement: EnhancementType = Query(
        default=EnhancementType.ENHANCED,
        description="Enhancement type for the output"
    )
):
    """
    Transform an image into anime style.
    
    Upload an image and receive an anime-styled version back.
    
    - **file**: The image file to transform (JPEG, PNG, or WebP)
    - **style**: The anime style (Shinkai, Hayao, or Paprika)
    - **enhancement**: The enhancement level (basic, enhanced, with_lines, maximum)
    """
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{request_id}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read and process image
        img_original = cv2.imread(str(upload_path))
        if img_original is None:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image")
        
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        # Transform image
        results = transform_image(img_original, style.value)
        
        # Get the requested enhancement
        result_image = results[enhancement.value]
        
        # Save output
        output_filename = f"{request_id}_{enhancement.value}.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        Image.fromarray(result_image).save(str(output_path), quality=95)
        
        # Return the transformed image
        return FileResponse(
            path=str(output_path),
            media_type="image/jpeg",
            filename=f"anime_{file.filename}",
            headers={
                "X-Request-ID": request_id,
                "X-Style-Used": style.value,
                "X-Enhancement": enhancement.value
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up upload file
        if upload_path.exists():
            upload_path.unlink()


@app.post("/transform/all")
async def transform_to_anime_all_versions(
    file: UploadFile = File(..., description="Image file to transform (JPEG, PNG, WebP)"),
    style: AnimeStyle = Query(
        default=AnimeStyle.SHINKAI,
        description="Anime style to use"
    )
):
    """
    Transform an image into anime style and get all enhancement versions.
    
    Returns URLs to all 4 versions: basic, enhanced, with_lines, and maximum.
    
    - **file**: The image file to transform (JPEG or PNG)
    - **style**: The anime style (Shinkai, Hayao, or Paprika)
    """
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{request_id}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read and process image
        img_original = cv2.imread(str(upload_path))
        if img_original is None:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image")
        
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        # Transform image
        results = transform_image(img_original, style.value)
        
        # Save all versions
        output_files = {}
        for enhancement_type, result_image in results.items():
            output_filename = f"{request_id}_{enhancement_type}.jpg"
            output_path = OUTPUT_DIR / output_filename
            Image.fromarray(result_image).save(str(output_path), quality=95)
            output_files[enhancement_type] = f"/outputs/{output_filename}"
        
        return JSONResponse(content={
            "request_id": request_id,
            "style": style.value,
            "original_filename": file.filename,
            "outputs": output_files,
            "message": "Successfully transformed image into all anime styles!"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up upload file
        if upload_path.exists():
            upload_path.unlink()


@app.get("/outputs/{filename}")
async def get_output_image(filename: str):
    """
    Retrieve a generated output image by filename.
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=filename
    )


# Serve static files for outputs
@app.on_event("startup")
async def startup_event():
    """Ensure output directory exists on startup"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
