"""
Preprocessing utilities for sketch images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def resize_image(image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Resize image while maintaining aspect ratio and padding if necessary.
    
    Args:
        image: Input PIL Image
        target_size: Target (width, height)
    
    Returns: 
        Resized PIL Image
    """
    # Calculate aspect ratio
    width, height = image. size
    target_width, target_height = target_size
    
    ratio = min(target_width / width, target_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    # Resize with high quality
    resized = image.resize(new_size, Image. Resampling. LANCZOS)
    
    # Create new image with padding
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_width - new_size[0]) // 2
    paste_y = (target_height - new_size[1]) // 2
    new_image.paste(resized, (paste_x, paste_y))
    
    return new_image


def apply_canny_edge(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """
    Apply Canny edge detection to an image.
    
    Args:
        image: Input PIL Image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge-detected PIL Image
    """
    # Convert to numpy array
    img_array = np. array(image)
    
    # Convert to grayscale if needed
    if len(img_array. shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)


def clean_sketch(image: Image.Image, denoise_strength: int = 10) -> Image.Image:
    """
    Clean and enhance a sketch image. 
    
    Args: 
        image: Input PIL Image
        denoise_strength:  Strength of denoising
    
    Returns:
        Cleaned PIL Image
    """
    img_array = np. array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Threshold to clean up
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to RGB
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return Image. fromarray(result)


def invert_sketch(image: Image. Image) -> Image.Image:
    """
    Invert sketch colors (black lines on white background to white lines on black).
    
    Args:
        image:  Input PIL Image
    
    Returns: 
        Inverted PIL Image
    """
    img_array = np.array(image)
    inverted = 255 - img_array
    return Image.fromarray(inverted)


def normalize_sketch(image: Image. Image) -> Image.Image:
    """
    Normalize sketch for ControlNet processing.
    
    Args:
        image: Input PIL Image
    
    Returns: 
        Normalized PIL Image
    """
    img_array = np.array(image)
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Normalize to 0-255 range
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def preprocess_for_controlnet(
    image: Image. Image,
    mode: str = "scribble",
    target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """
    Complete preprocessing pipeline for ControlNet.
    
    Args:
        image: Input PIL Image
        mode: ControlNet mode ('scribble', 'canny', 'lineart', 'hed')
        target_size:  Target image size
    
    Returns:
        Preprocessed PIL Image
    """
    # Resize first
    image = resize_image(image, target_size)
    
    # Apply mode-specific preprocessing
    if mode == "canny":
        image = apply_canny_edge(image)
    elif mode == "scribble": 
        image = clean_sketch(image)
    elif mode in ["lineart", "hed"]: 
        image = normalize_sketch(image)
    
    return image
