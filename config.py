"""
Configuration settings for the Sketch-to-Image generation system.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    
    # Base Stable Diffusion model
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    
    # ControlNet models for different modes
    controlnet_models: dict = None
    
    # Device configuration
    device:  str = "cuda" if torch.cuda. is_available() else "cpu"
    dtype: torch.dtype = torch. float16 if torch.cuda.is_available() else torch.float32
    
    # Generation defaults
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    default_controlnet_scale: float = 1.0
    
    # Image settings
    default_width: int = 512
    default_height: int = 512
    
    def __post_init__(self):
        if self.controlnet_models is None: 
            self.controlnet_models = {
                "scribble": "lllyasviel/sd-controlnet-scribble",
                "canny": "lllyasviel/sd-controlnet-canny",
                "lineart": "lllyasviel/control_v11p_sd15_lineart",
                "hed": "lllyasviel/sd-controlnet-hed",
            }


@dataclass
class AppConfig:
    """Configuration for the Gradio application."""
    
    # Server settings
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share:  bool = False
    
    # UI settings
    title: str = "🎨 Sketch to Image Generator"
    description:  str = """
    Transform your hand-drawn sketches into realistic images using AI! 
    
    **How to use:**
    1. Draw a sketch on the canvas or upload an image
    2. Enter a text prompt describing the desired output
    3. Select a style preset and ControlNet mode
    4. Click Generate! 
    """
    
    # Theme
    theme: str = "soft"


# Global configuration instances
model_config = ModelConfig()
app_config = AppConfig()
