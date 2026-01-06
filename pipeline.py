"""
ControlNet + Stable Diffusion pipeline for sketch-to-image generation.
"""

import torch
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from controlnet_aux import (
    CannyDetector,
    HEDdetector,
    LineartDetector
)

from config import model_config
from preprocessing import preprocess_for_controlnet
from styles.presets import apply_style_to_prompt


class SketchToImagePipeline:
    """
    Pipeline for converting sketches to images using ControlNet and Stable Diffusion. 
    """
    
    def __init__(self, mode: str = "scribble"):
        """
        Initialize the pipeline.
        
        Args:
            mode: ControlNet mode ('scribble', 'canny', 'lineart', 'hed')
        """
        self. mode = mode
        self.device = model_config.device
        self.dtype = model_config. dtype
        self.pipe = None
        self.controlnet = None
        self.processors = {}
        
        print(f"Initializing pipeline on {self.device}...")
        self._load_pipeline(mode)
        self._load_processors()
    
    def _load_pipeline(self, mode:  str):
        """Load the ControlNet and Stable Diffusion pipeline."""
        controlnet_model_id = model_config.controlnet_models. get(mode)
        
        if controlnet_model_id is None: 
            raise ValueError(f"Unknown mode: {mode}")
        
        print(f"Loading ControlNet model: {controlnet_model_id}")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=self.dtype
        )
        
        print(f"Loading Stable Diffusion model:  {model_config. sd_model_id}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_config. sd_model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        
        # Use efficient scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self. pipe.scheduler.config
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            try:
                self. pipe.enable_xformers_memory_efficient_attention()
            except Exception: 
                pass  # xformers not available
        
        print("Pipeline loaded successfully!")
    
    def _load_processors(self):
        """Load image processors for different modes."""
        self.processors = {
            "canny": CannyDetector(),
            "hed": HEDdetector. from_pretrained("lllyasviel/Annotators"),
            "lineart": LineartDetector.from_pretrained("lllyasviel/Annotators"),
        }
    
    def switch_mode(self, mode: str):
        """Switch to a different ControlNet mode."""
        if mode != self.mode:
            self.mode = mode
            self._load_pipeline(mode)
    
    def process_sketch(self, sketch: Image.Image) -> Image.Image:
        """
        Process sketch image based on current mode.
        
        Args:
            sketch: Input sketch image
        
        Returns: 
            Processed image ready for ControlNet
        """
        # Resize first
        sketch = preprocess_for_controlnet(
            sketch, 
            mode=self.mode,
            target_size=(model_config.default_width, model_config. default_height)
        )
        
        # Apply mode-specific processor
        if self. mode == "canny" and "canny" in self.processors:
            sketch = self.processors["canny"](sketch)
        elif self.mode == "hed" and "hed" in self. processors:
            sketch = self.processors["hed"](sketch)
        elif self.mode == "lineart" and "lineart" in self. processors:
            sketch = self.processors["lineart"](sketch)
        
        return sketch
    
    def generate(
        self,
        sketch: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        style:  str = "none",
        num_inference_steps:  Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = -1
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate an image from a sketch. 
        
        Args:
            sketch:  Input sketch image
            prompt: Text prompt describing desired output
            negative_prompt: Things to avoid in the output
            style: Style preset name
            num_inference_steps: Number of denoising steps
            guidance_scale:  Guidance scale for generation
            controlnet_conditioning_scale: ControlNet influence strength
            seed: Random seed (-1 for random)
        
        Returns:
            Tuple of (generated_image, processed_sketch)
        """
        # Apply style preset
        styled_prompt, style_negative, style_guidance, style_steps = apply_style_to_prompt(
            prompt, style
        )
        
        # Use style defaults if not specified
        if num_inference_steps is None:
            num_inference_steps = style_steps
        if guidance_scale is None:
            guidance_scale = style_guidance
        
        # Combine negative prompts
        full_negative = f"{negative_prompt}, {style_negative}".strip(", ")
        
        # Process sketch
        processed_sketch = self. process_sketch(sketch)
        
        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self. device).manual_seed(seed)
        
        # Generate image
        result = self.pipe(
            prompt=styled_prompt,
            negative_prompt=full_negative,
            image=processed_sketch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        )
        
        return result. images[0], processed_sketch


# Global pipeline instance (lazy loaded)
_pipeline_instance: Optional[SketchToImagePipeline] = None


def get_pipeline(mode: str = "scribble") -> SketchToImagePipeline: 
    """Get or create the global pipeline instance."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = SketchToImagePipeline(mode)
    elif _pipeline_instance.mode != mode:
        _pipeline_instance. switch_mode(mode)
    
    return _pipeline_instance


def generate_image(
    sketch:  Image.Image,
    prompt: str,
    negative_prompt: str = "",
    style: str = "none",
    mode: str = "scribble",
    num_inference_steps: int = 30,
    guidance_scale:  float = 7.5,
    controlnet_scale: float = 1.0,
    seed: int = -1
) -> Tuple[Image.Image, Image.Image]: 
    """
    Convenience function to generate an image from a sketch. 
    
    Returns:
        Tuple of (generated_image, processed_sketch)
    """
    pipeline = get_pipeline(mode)
    return pipeline.generate(
        sketch=sketch,
        prompt=prompt,
        negative_prompt=negative_prompt,
        style=style,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        seed=seed
    )
