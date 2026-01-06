"""
Style presets for image generation.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class StylePreset:
    """A style preset with prompt modifiers."""
    name: str
    prompt_prefix: str
    prompt_suffix: str
    negative_prompt:  str
    guidance_scale: float = 7.5
    num_inference_steps:  int = 30


# Define all style presets
STYLE_PRESETS:  Dict[str, StylePreset] = {
    "realistic":  StylePreset(
        name="Realistic / Photorealistic",
        prompt_prefix="photorealistic, highly detailed, professional photography,",
        prompt_suffix=", 8k uhd, sharp focus, realistic lighting, natural colors",
        negative_prompt="cartoon, anime, drawing, painting, sketch, low quality, blurry, distorted, deformed",
        guidance_scale=7.5,
        num_inference_steps=35
    ),
    
    "anime": StylePreset(
        name="Anime / Manga",
        prompt_prefix="anime style, manga art, studio ghibli inspired,",
        prompt_suffix=", vibrant colors, clean lines, detailed anime illustration",
        negative_prompt="photorealistic, photo, 3d render, ugly, deformed, noisy, blurry",
        guidance_scale=8.0,
        num_inference_steps=30
    ),
    
    "oil_painting": StylePreset(
        name="Oil Painting",
        prompt_prefix="oil painting, classical art style, masterpiece,",
        prompt_suffix=", rich colors, visible brushstrokes, museum quality, by renowned artist",
        negative_prompt="photo, digital art, modern, cartoon, low quality",
        guidance_scale=7.5,
        num_inference_steps=35
    ),
    
    "watercolor": StylePreset(
        name="Watercolor",
        prompt_prefix="watercolor painting, soft edges, delicate,",
        prompt_suffix=", translucent colors, artistic, flowing pigments, paper texture",
        negative_prompt="photo, digital, harsh edges, dark colors, cartoon",
        guidance_scale=7.0,
        num_inference_steps=30
    ),
    
    "digital_art": StylePreset(
        name="Digital Art",
        prompt_prefix="digital art, concept art, trending on artstation,",
        prompt_suffix=", highly detailed, vibrant, professional digital painting",
        negative_prompt="photo, blurry, low resolution, amateur, sketch",
        guidance_scale=8.0,
        num_inference_steps=30
    ),
    
    "futuristic": StylePreset(
        name="Futuristic / Sci-Fi",
        prompt_prefix="futuristic, sci-fi, cyberpunk, high tech,",
        prompt_suffix=", neon lights, advanced technology, sleek design, dystopian",
        negative_prompt="medieval, old, rustic, vintage, low tech, blurry",
        guidance_scale=8.0,
        num_inference_steps=35
    ),
    
    "fantasy":  StylePreset(
        name="Fantasy",
        prompt_prefix="fantasy art, magical, epic, mythical,",
        prompt_suffix=", enchanted, mystical atmosphere, dramatic lighting, ethereal",
        negative_prompt="modern, realistic, photo, mundane, boring",
        guidance_scale=8.0,
        num_inference_steps=35
    ),
    
    "architectural": StylePreset(
        name="Architectural",
        prompt_prefix="architectural visualization, professional render,",
        prompt_suffix=", detailed architecture, realistic materials, proper perspective, 3d render quality",
        negative_prompt="cartoon, sketch, low quality, distorted perspective",
        guidance_scale=7.5,
        num_inference_steps=35
    ),
    
    "sketch_enhanced": StylePreset(
        name="Enhanced Sketch",
        prompt_prefix="detailed sketch, professional illustration,",
        prompt_suffix=", clean lines, shading, artistic drawing",
        negative_prompt="photo, color, painting, blurry",
        guidance_scale=6.5,
        num_inference_steps=25
    ),
    
    "none": StylePreset(
        name="No Style (Custom)",
        prompt_prefix="",
        prompt_suffix="",
        negative_prompt="low quality, blurry, distorted, deformed, ugly",
        guidance_scale=7.5,
        num_inference_steps=30
    ),
}


def get_style_preset(style_name: str) -> StylePreset: 
    """Get a style preset by name."""
    return STYLE_PRESETS.get(style_name, STYLE_PRESETS["none"])


def get_style_names() -> list:
    """Get list of all style preset names."""
    return list(STYLE_PRESETS.keys())


def apply_style_to_prompt(prompt: str, style_name: str) -> tuple: 
    """
    Apply a style preset to a prompt.
    
    Returns:
        Tuple of (modified_prompt, negative_prompt, guidance_scale, steps)
    """
    style = get_style_preset(style_name)
    
    modified_prompt = f"{style.prompt_prefix} {prompt} {style.prompt_suffix}".strip()
    
    return (
        modified_prompt,
        style. negative_prompt,
        style.guidance_scale,
        style.num_inference_steps
    )
