"""
Main Gradio interface for Sketch-to-Image generation.
"""

import gradio as gr
from PIL import Image
from typing import Optional, Tuple
import os

from pipeline import generate_image, get_pipeline
from styles.presets import get_style_names, STYLE_PRESETS
from preprocessing import preprocess_for_controlnet
from config import app_config
from chatbot. bot import create_chatbot_interface


def create_main_interface() -> gr.Blocks:
    """Create the main Gradio interface."""
    
    style_choices = [(STYLE_PRESETS[s]. name, s) for s in get_style_names()]
    mode_choices = [
        ("Scribble (rough sketches)", "scribble"),
        ("Canny (edge detection)", "canny"),
        ("Lineart (clean lines)", "lineart"),
        ("HED (soft edges)", "hed"),
    ]
    
    with gr.Blocks(
        title=app_config.title,
        theme=gr.themes. Soft()
    ) as demo:
        
        gr.Markdown(f"""
        # {app_config.title}
        {app_config.description}
        """)
        
        with gr. Tabs():
            # Main Generation Tab
            with gr.TabItem("🎨 Generate"):
                with gr. Row():
                    # Left column - Inputs
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Input")
                        
                        sketch_input = gr. Image(
                            label="Draw or Upload Sketch",
                            type="pil",
                            tool="sketch",
                            height=400,
                            brush_radius=3
                        )
                        
                        prompt_input = gr. Textbox(
                            label="Prompt",
                            placeholder="Describe what you want to generate (e.g., 'futuristic city with neon lights')",
                            lines=2
                        )
                        
                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt (optional)",
                            placeholder="What to avoid (e.g., 'blurry, low quality')",
                            lines=1
                        )
                        
                        with gr.Row():
                            style_dropdown = gr.Dropdown(
                                choices=style_choices,
                                value="realistic",
                                label="Style Preset"
                            )
                            
                            mode_dropdown = gr.Dropdown(
                                choices=mode_choices,
                                value="scribble",
                                label="ControlNet Mode"
                            )
                        
                        with gr. Accordion("⚙️ Advanced Settings", open=False):
                            steps_slider = gr. Slider(
                                minimum=10,
                                maximum=50,
                                value=30,
                                step=1,
                                label="Inference Steps"
                            )
                            
                            guidance_slider = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale"
                            )
                            
                            controlnet_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="ControlNet Strength"
                            )
                            
                            seed_input = gr.Number(
                                value=-1,
                                label="Seed (-1 for random)",
                                precision=0
                            )
                        
                        generate_btn = gr. Button("🚀 Generate Image", variant="primary", size="lg")
                    
                    # Right column - Outputs
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Output")
                        
                        with gr.Row():
                            processed_output = gr.Image(
                                label="Processed Sketch",
                                type="pil",
                                height=200
                            )
                            
                            generated_output = gr.Image(
                                label="Generated Image",
                                type="pil",
                                height=400
                            )
                        
                        # Gallery for history
                        gallery = gr.Gallery(
                            label="Generation History",
                            columns=4,
                            height=150
                        )
                
                # Generation history state
                history_state = gr.State([])
                
                def on_generate(
                    sketch, prompt, negative_prompt, style, mode,
                    steps, guidance, controlnet_scale, seed, history
                ):
                    if sketch is None:
                        gr.Warning("Please draw or upload a sketch first!")
                        return None, None, history
                    
                    if not prompt.strip():
                        prompt = "detailed artwork, high quality"
                    
                    try:
                        generated, processed = generate_image(
                            sketch=sketch,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            style=style,
                            mode=mode,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            controlnet_scale=float(controlnet_scale),
                            seed=int(seed)
                        )
                        
                        # Add to history
                        history. append(generated)
                        
                        return processed, generated, history
                    
                    except Exception as e:
                        gr.Error(f"Generation failed: {str(e)}")
                        return None, None, history
                
                generate_btn. click(
                    on_generate,
                    inputs=[
                        sketch_input, prompt_input, negative_prompt_input,
                        style_dropdown, mode_dropdown, steps_slider,
                        guidance_slider, controlnet_slider, seed_input, history_state
                    ],
                    outputs=[processed_output, generated_output, history_state]
                )
                
                # Update gallery when history changes
                history_state.change(
                    lambda h: h[-8:] if h else [],  # Show last 8 images
                    inputs=[history_state],
                    outputs=[gallery]
                )
            
            # Chatbot Tab
            with gr.TabItem("💬 Chatbot"):
                gr.Markdown("""
                ### 🤖 AI Chatbot Assistant
                
                Chat with our AI to generate images!  Upload a sketch and describe what you want. 
                """)
                
                # Embed chatbot interface
                chatbot_demo = create_chatbot_interface()
            
            # Examples Tab
            with gr. TabItem("📚 Examples"):
                gr.Markdown("""
                ### Example Generations
                
                Here are some examples of what you can create:
                
                | Sketch Type | Recommended Style | Recommended Mode |
                |-------------|-------------------|------------------|
                | Buildings | Architectural, Futuristic | Lineart |
                | Characters | Anime, Digital Art | Scribble |
                | Landscapes | Oil Painting, Realistic | Canny |
                | Products | Realistic, Digital Art | Lineart |
                
                **Tips:**
                - Use simple, clear lines for best results
                - Add text prompts to guide the style and details
                - Experiment with different ControlNet modes
                - Try various style presets to find your favorite
                """)
            
            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ### About This Project
                
                This sketch-to-image generation system uses **ControlNet** with **Stable Diffusion** 
                to transform hand-drawn sketches into realistic images. 
                
                #### Technologies Used:
                - **Stable Diffusion**:  Base image generation model
                - **ControlNet**:  Sketch-guided image generation
                - **Gradio**: Web interface
                - **Hugging Face Diffusers**: Model pipeline
                
                #### References:
                - [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
                - [ControlNet](https://github.com/lllyasviel/ControlNet)
                - [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
                - [Gradio](https://github.com/gradio-app/gradio)
                - [ControlNet Auxiliary Models](https://github.com/huggingface/controlnet_aux)
                
                #### License: 
                This project is for educational purposes. 
                """)
    
    return demo


def main():
    """Main entry point."""
    demo = create_main_interface()
    demo.launch(
        server_name=app_config.server_name,
        server_port=app_config.server_port,
        share=app_config.share
    )


if __name__ == "__main__":
    main()
