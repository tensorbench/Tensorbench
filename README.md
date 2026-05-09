# 🧠 TensorBench

> **Benchmark your hardware for local AI workloads. Find the perfect configuration for image generation and LLM inference.**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt-6.4+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Beta-orange.svg)

TensorBench is a desktop application for evaluating your PC's performance in AI tasks. It analyzes your hardware, compares it against a comprehensive database, and provides personalized upgrade recommendations.

---

## ✨ Features (Beta v0.1.0)

### 🔍 Instant Hardware Detection
- Automatic detection of NVIDIA GPU, CPU, and RAM
- Real-time specs display: VRAM capacity, memory type, clock speeds

### 🎨 Professional Interface
- **Bento-grid layout**: Model cards with large numbers and progress bars
- **Animated profile switcher**: Smooth transitions between modes
- **Dark theme**: Comfortable for extended use

### 🔄 Dual Mode Operation
| Mode | Metric | Models |
|------|--------|--------|
| 🎨 **Image Synthesis** | it/s (iterations per second) | SD v1.5, SDXL, Flux.1-Dev, PixArt-alpha, HunyuanDiT |
| 📝 **LLM Inference** | tok/s (tokens per second) | Qwen2.5, Mistral-7B, Llama-3-8B/70B, Command-R |

### 📊 Visual Comparison
- **Baseline vs Upgraded**: Clear performance delta visualization
- **Color coding**: 🟢 optimal / 🟡 marginal / 🔴 upgrade needed
- **VRAM indicators**: Instant feedback on model memory requirements

### 💡 Smart Recommendations
- Personalized advice based on your configuration
- VRAM shortage warnings
- System balance optimization tips

---

## 🚀 Quick Start

### ▶️ For Users (Ready-to-run .exe)
1. Go to [Releases](https://github.com/tensorbench/Tensorbench/releases)
2. Download the latest `TensorBench.exe`
3. Run the file (first launch may take 5–10 seconds for extraction)
4. Allow access if antivirus prompts (this is a false positive from the packager)

> ⚠️ **Important**: NVIDIA driver must be installed for proper GPU detection.

### 🛠 For Developers (From source)
```bash
# 1. Clone the repository
git clone https://github.com/tensorbench/Tensorbench.git
cd Tensorbench

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python -m tensorbench.gui.main_window