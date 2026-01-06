"""
Chatbot interface for sketch-to-image generation.
"""

import gradio as gr
from PIL import Image
from typing import Tuple, List, Optional
import os
import sys

# Add parent directory to path for imports
sys.path. insert(0, os. path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import generate_image
from chatbot.agent import get_agent
from styles.presets import get_style_names


class SketchChatbot:
    """
    Chatbot interface for sketch-to-image generation.
    """
    
    def __init__(self):
        self.agent = get_agent()
        self.current_sketch = None
        self.conversation_history = []
    
    def process_message(
        self,
        message: str,
        sketch: Optional[Image.Image],
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], Optional[Image.Image]]:
        """
        Process a chat message and optionally generate an image.
        
        Args:
            message: User message
            sketch: Optional sketch image
            history:  Conversation history
        
        Returns:
            Updated history and generated image (if any)
        """
        generated_image = None
        
        # Update current sketch if provided
        if sketch is not None:
            self.current_sketch = sketch
            response = "✅ Sketch received! I can see your drawing.  Tell me what style you'd like, or just say 'generate' to create an image with auto-detected settings."
        elif self.current_sketch is None:
            response = "👋 Hello! Please upload a sketch first, then tell me how you'd like it transformed.  You can specify a style (realistic, anime, oil painting, etc.) or let me auto-detect the best settings."
        else:
            # Parse user intent
            message_lower = message. lower()
            
            if any(word in message_lower for word in ["generate", "create", "make", "transform", "convert"]):
                # Generate image
                try:
                    # Get agent suggestions
                    agent_result = self.agent. process(
                        self.current_sketch,
                        user_prompt=message if len(message) > 20 else None
                    )
                    
                    # Check for style keywords in message
                    style = "none"
                    for style_name in get_style_names():
                        if style_name.replace("_", " ") in message_lower:
                            style = style_name
                            break
                    
                    if style == "none": 
                        style = agent_result["style"]
                    
                    # Generate
                    generated_image, _ = generate_image(
                        sketch=self.current_sketch,
                        prompt=agent_result["prompt"],
                        style=style,
                        mode=agent_result["mode"]
                    )
                    
                    response = f"🎨 Image generated!\n\n**Settings used:**\n- Style: {style}\n- Mode: {agent_result['mode']}\n- Prompt: {agent_result['prompt'][: 100]}.. .\n\nWant to try a different style?  Just tell me!"
                    
                except Exception as e: 
                    response = f"❌ Error generating image: {str(e)}\n\nPlease try again or adjust your request."
            
            elif any(word in message_lower for word in ["suggest", "help", "what", "options"]):
                # Provide suggestions
                suggestions = self.agent.get_suggestions(self.current_sketch)
                
                response = f"""💡 **Suggestions for your sketch:**

**Detected content:** {suggestions['likely_content']}

**Recommended styles:**
{chr(10).join(f'- {s}' for s in suggestions['suggested_styles'])}

**Sample prompts:**
{chr(10).join(f'- "{p[: 60]}..."' for p in suggestions['suggested_prompts'][:3])}

Just say "generate" or specify a style like "generate in anime style"! """
            
            elif any(word in message_lower for word in ["style", "list"]):
                # List available styles
                styles = get_style_names()
                response = f"""🎭 **Available styles:**

{chr(10).join(f'- {s. replace("_", " ").title()}' for s in styles)}

Say "generate in [style] style" to use one! """
            
            else:
                # Use message as prompt
                try:
                    agent_result = self. agent.process(
                        self.current_sketch,
                        user_prompt=message
                    )
                    
                    generated_image, _ = generate_image(
                        sketch=self.current_sketch,
                        prompt=message,
                        style=agent_result["style"],
                        mode=agent_result["mode"]
                    )
                    
                    response = f"🎨 Generated with your description:  \"{message[: 50]}...\"\n\nWant to try something different?"
                    
                except Exception as e: 
                    response = f"I'll use \"{message}\" as the generation prompt. Say 'generate' when ready, or upload a new sketch."
        
        # Update history
        history. append((message, response))
        self.conversation_history = history
        
        return history, generated_image
    
    def reset(self):
        """Reset the chatbot state."""
        self. current_sketch = None
        self.conversation_history = []
        self.agent.clear_history()
        return [], None, None


def create_chatbot_interface() -> gr.Blocks:
    """Create the Gradio chatbot interface."""
    
    chatbot_instance = SketchChatbot()
    
    with gr. Blocks(title="🎨 Sketch-to-Image Chatbot", theme=gr.themes. Soft()) as demo:
        gr.Markdown("""
        # 🎨 Sketch-to-Image Chatbot
        
        Upload a sketch and chat with me to transform it into a realistic image! 
        
        **How to use:**
        1. Upload your sketch below
        2. Tell me what style you want (or ask for suggestions)
        3. Say "generate" to create your image! 
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                sketch_input = gr.Image(
                    label="📝 Upload Sketch",
                    type="pil",
                    height=300
                )
                
                generated_output = gr.Image(
                    label="🖼️ Generated Image",
                    type="pil",
                    height=300
                )
            
            with gr. Column(scale=1):
                chatbot = gr. Chatbot(
                    label="💬 Chat",
                    height=400,
                    bubble_full_width=False
                )
                
                msg_input = gr. Textbox(
                    label="Your message",
                    placeholder="Type 'generate', ask for suggestions, or describe what you want...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr. Button("Send", variant="primary")
                    clear_btn = gr. Button("Reset")
        
        # Event handlers
        def on_submit(message, sketch, history):
            if not message.strip():
                return history, None, ""
            return *chatbot_instance. process_message(message, sketch, history), ""
        
        def on_sketch_upload(sketch, history):
            if sketch is not None: 
                return chatbot_instance.process_message("", sketch, history)
            return history, None
        
        def on_clear():
            return chatbot_instance. reset()
        
        submit_btn. click(
            on_submit,
            inputs=[msg_input, sketch_input, chatbot],
            outputs=[chatbot, generated_output, msg_input]
        )
        
        msg_input.submit(
            on_submit,
            inputs=[msg_input, sketch_input, chatbot],
            outputs=[chatbot, generated_output, msg_input]
        )
        
        sketch_input.change(
            on_sketch_upload,
            inputs=[sketch_input, chatbot],
            outputs=[chatbot, generated_output]
        )
        
        clear_btn. click(
            on_clear,
            outputs=[chatbot, generated_output, sketch_input]
        )
    
    return demo


def launch_chatbot(share:  bool = False):
    """Launch the chatbot interface."""
    demo = create_chatbot_interface()
    demo.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
    launch_chatbot()
