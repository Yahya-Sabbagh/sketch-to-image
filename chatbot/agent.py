"""
AI Agent for automated sketch-to-image pipeline.
"""

from PIL import Image
from typing import Tuple, Optional, Dict, Any
import random


class SketchAnalyzer:
    """Analyzes sketches to determine content and suggest prompts."""
    
    def __init__(self):
        self.content_keywords = {
            "building":  ["building", "architecture", "house", "tower", "skyscraper", "structure"],
            "character": ["person", "character", "human", "figure", "portrait", "face"],
            "landscape": ["landscape", "nature", "mountain", "tree", "scenery", "outdoor"],
            "vehicle": ["car", "vehicle", "airplane", "ship", "boat", "motorcycle"],
            "animal": ["animal", "dog", "cat", "bird", "creature", "pet"],
            "object": ["object", "item", "product", "furniture", "device", "tool"],
        }
        
        self.style_suggestions = {
            "building": ["architectural", "futuristic", "realistic"],
            "character":  ["anime", "realistic", "digital_art", "fantasy"],
            "landscape": ["realistic", "oil_painting", "watercolor", "fantasy"],
            "vehicle": ["futuristic", "realistic", "digital_art"],
            "animal":  ["realistic", "anime", "watercolor"],
            "object":  ["realistic", "digital_art", "futuristic"],
        }
        
        self.prompt_templates = {
            "building": [
                "detailed architectural design, modern building, professional visualization",
                "futuristic structure with glass and steel, cityscape",
                "elegant building with intricate details, sunset lighting"
            ],
            "character": [
                "detailed character portrait, expressive features, dynamic pose",
                "fantasy character with detailed costume, dramatic lighting",
                "anime character with vibrant colors, detailed design"
            ],
            "landscape": [
                "breathtaking landscape, natural beauty, golden hour lighting",
                "serene nature scene with mountains and water, peaceful atmosphere",
                "dramatic outdoor scenery, volumetric lighting, epic scale"
            ],
            "vehicle": [
                "sleek vehicle design, futuristic technology, detailed render",
                "high-performance machine, dynamic angle, studio lighting",
                "concept vehicle with innovative features, professional design"
            ],
            "animal": [
                "majestic animal portrait, natural habitat, detailed fur/feathers",
                "cute creature with expressive eyes, soft lighting",
                "wild animal in action, dynamic pose, nature background"
            ],
            "object": [
                "product design visualization, clean background, studio lighting",
                "detailed object render, realistic materials, professional quality",
                "innovative design concept, futuristic aesthetics, high detail"
            ],
        }
    
    def analyze_sketch(self, sketch: Image.Image) -> Dict[str, Any]:
        """
        Analyze a sketch and return content information.
        
        Note: This is a simplified analyzer. In production, you would use
        a vision model (like CLIP or BLIP) for actual content detection.
        """
        # Get image properties
        width, height = sketch.size
        aspect_ratio = width / height
        
        # Determine likely content based on aspect ratio (simplified heuristic)
        if aspect_ratio > 1.5:
            likely_content = "landscape"
        elif aspect_ratio < 0.7:
            likely_content = "character"
        else: 
            likely_content = random.choice(["building", "character", "object"])
        
        return {
            "likely_content": likely_content,
            "aspect_ratio": aspect_ratio,
            "size": (width, height),
            "suggested_styles": self.style_suggestions.get(likely_content, ["realistic"]),
            "suggested_prompts": self.prompt_templates. get(likely_content, [])
        }
    
    def generate_prompt(self, sketch: Image.Image, user_hint: str = "") -> str:
        """
        Generate an appropriate prompt for a sketch.
        
        Args:
            sketch: Input sketch image
            user_hint: Optional user-provided hint about content
        
        Returns:
            Generated prompt string
        """
        analysis = self.analyze_sketch(sketch)
        
        if user_hint:
            # Use user hint to guide prompt generation
            base_prompt = user_hint
        else:
            # Use template based on analysis
            templates = analysis["suggested_prompts"]
            base_prompt = random. choice(templates) if templates else "detailed artwork"
        
        # Add quality modifiers
        quality_modifiers = "high quality, detailed, professional, masterpiece"
        
        return f"{base_prompt}, {quality_modifiers}"


class SketchToImageAgent:
    """
    AI Agent that automates the entire sketch-to-image pipeline.
    """
    
    def __init__(self):
        self.analyzer = SketchAnalyzer()
        self.history = []
    
    def process(
        self,
        sketch: Image.Image,
        user_prompt: Optional[str] = None,
        preferred_style: Optional[str] = None
    ) -> Dict[str, Any]: 
        """
        Process a sketch through the automated pipeline.
        
        Args:
            sketch: Input sketch image
            user_prompt: Optional user-provided prompt
            preferred_style: Optional preferred style
        
        Returns: 
            Dictionary with generation parameters and suggestions
        """
        # Analyze the sketch
        analysis = self.analyzer.analyze_sketch(sketch)
        
        # Generate or use provided prompt
        if user_prompt:
            prompt = user_prompt
        else: 
            prompt = self. analyzer.generate_prompt(sketch)
        
        # Select style
        if preferred_style:
            style = preferred_style
        else:
            suggested_styles = analysis["suggested_styles"]
            style = suggested_styles[0] if suggested_styles else "realistic"
        
        # Determine optimal ControlNet mode
        content = analysis["likely_content"]
        if content in ["building", "object"]:
            mode = "lineart"
        elif content == "character":
            mode = "scribble"
        else:
            mode = "scribble"
        
        # Build result
        result = {
            "prompt":  prompt,
            "style": style,
            "mode": mode,
            "analysis": analysis,
            "parameters": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.0
            }
        }
        
        # Add to history
        self. history.append(result)
        
        return result
    
    def get_suggestions(self, sketch:  Image.Image) -> Dict[str, Any]:
        """Get suggestions without processing."""
        analysis = self.analyzer. analyze_sketch(sketch)
        
        return {
            "suggested_prompts": analysis["suggested_prompts"],
            "suggested_styles": analysis["suggested_styles"],
            "likely_content": analysis["likely_content"]
        }
    
    def clear_history(self):
        """Clear agent history."""
        self.history = []


# Global agent instance
_agent_instance:  Optional[SketchToImageAgent] = None


def get_agent() -> SketchToImageAgent: 
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SketchToImageAgent()
    return _agent_instance
