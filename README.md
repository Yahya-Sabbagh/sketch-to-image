# 🎨 Sketch-to-Image Generator

Transform your hand-drawn sketches into realistic images using AI!  This project uses **ControlNet** with **Stable Diffusion** to convert sketches into high-quality images while preserving the original structure. 

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🖌️ **Interactive Sketch Canvas**: Draw directly in the browser
- 📤 **Image Upload**: Use existing sketches
- 🎭 **Multiple Styles**: Realistic, Anime, Oil Painting, Watercolor, and more
- 🔧 **ControlNet Modes**: Scribble, Canny, Lineart, HED
- 💬 **AI Chatbot**:  Conversational interface for image generation
- 🤖 **AI Agent**: Auto-detects sketch content and suggests prompts
- 📊 **Side-by-side Comparison**: View sketch vs. generated image
- 🐳 **Docker Support**: Easy deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Yahya-Sabbagh/sketch-to-image. git
cd sketch-to-image
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python app. py
```

5. **Open in browser:**
```
http://localhost:7860
```

### Docker Deployment

```bash
docker build -t sketch-to-image .
docker run -p 7860:7860 --gpus all sketch-to-image
```

## 📖 Usage Guide

### Basic Generation

1. Draw a sketch on the canvas or upload an image
2. Enter a text prompt describing the desired output
3. Select a style preset (e.g., "Realistic", "Anime")
4. Choose a ControlNet mode: 
   - **Scribble**:  For rough, hand-drawn sketches
   - **Canny**: For edge-detected images
   - **Lineart**: For clean line drawings
   - **HED**: For soft edge detection
5. Click "Generate Image"

### Using the Chatbot

1. Go to the "Chatbot" tab
2. Upload your sketch
3. Chat naturally:  "Generate this as a futuristic building"
4. Ask for suggestions: "What styles would work best?"

### Style Presets

| Style | Best For |
|-------|----------|
| Realistic | Architectural renders, products |
| Anime | Characters, illustrations |
| Oil Painting | Artistic, classical look |
| Watercolor | Soft, artistic images |
| Digital Art | Concept art, illustrations |
| Futuristic | Sci-fi, tech designs |
| Fantasy | Magical, mythical scenes |
| Architectural | Building visualizations |

## 🗂️ Project Structure

```
sketch-to-image/
├── app.py                 # Main Gradio interface
├── pipeline.py            # ControlNet + SD pipeline
├── preprocessing.py       # Image preprocessing
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker deployment
├── chatbot/
│   ├── bot. py            # Chatbot interface
│   └── agent. py          # AI agent
├── styles/
│   └── presets.py        # Style presets
└── examples/
    ├── sketches/         # Sample sketches
    └── outputs/          # Generated outputs
```

## 🔧 Configuration

Edit `config.py` to customize: 

- Model paths
- Default generation parameters
- Server settings
- UI configuration

## 📚 References & Credits

### Models
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) by RunwayML
- [ControlNet](https://github.com/lllyasviel/ControlNet) by lllyasviel
- [ControlNet Models](https://huggingface.co/lllyasviel) on Hugging Face

### Libraries
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Gradio](https://github.com/gradio-app/gradio)
- [ControlNet Aux](https://github.com/huggingface/controlnet_aux)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

### Papers
- [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (ControlNet)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion)

## 🎥 Demo Video

To record a demo video: 

1. Run the application locally
2. Use screen recording software
3. Demonstrate: 
   - Drawing a sketch
   - Entering a prompt
   - Generating an image
   - Using different styles
   - Using the chatbot

## 🚀 Deployment Options

### Local Deployment
```bash
python app.py
```

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload all project files
3. Set `app_file:  app.py` in README metadata

### Docker
```bash
docker build -t sketch-to-image .
docker run -p 7860:7860 --gpus all sketch-to-image
```

## 📝 License

This project is for educational purposes as part of the "Bringing Sketches to Life with Generative AI" course project.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## 📧 Contact

- GitHub: [@Yahya-Sabbagh](https://github.com/Yahya-Sabbagh)
