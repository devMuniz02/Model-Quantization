# Model Quantization

> Effortlessly quantize and deploy Hugging Face models. This repo provides a streamlined pipeline to convert any HF model into a compressed format and upload it directly to your HF profile. Ideal for optimizing Large Language Models (LLMs) for resource-constrained environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/devMuniz02/Model-Quantization)](https://github.com/devMuniz02/Model-Quantization/issues)
[![GitHub stars](https://img.shields.io/github/stars/devMuniz02/Model-Quantization)](https://github.com/devMuniz02/Model-Quantization/stargazers)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Repository Setup](#repository-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## ✨ Features

- **Terminal-based Quantization**: Interactive command-line interface for easy model quantization
- **4-bit Quantization Support**: NF4 and FP4 quantization types for optimal compression
- **Double Quantization**: Enhanced compression with double quantization option
- **Memory Footprint Analysis**: Automatic calculation of original vs quantized model sizes
- **Hugging Face Integration**: Direct upload to Hugging Face Hub with generated model cards
- **Flexible Configuration**: Customizable compute and storage data types
- **Local Saving**: Option to save quantized models locally
- **Model Card Generation**: Automatic creation of detailed model cards for quantized models
- **Jupyter Notebook Support**: Interactive notebook for testing and experimentation

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Git
- Hugging Face account (for uploading models)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/devMuniz02/Model-Quantization.git

# Navigate to the project directory
cd Model-Quantization

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended for faster quantization)
- **RAM**: At least 8GB (16GB+ recommended for large models)
- **Storage**: Sufficient space for original and quantized models

## 📁 Project Structure

```
Model-Quantization/
├── assets/                 # Static assets (images, icons, etc.)
├── data/                   # Data files and datasets
├── docs/                   # Documentation files
├── notebooks/              # Jupyter notebooks for analysis and prototyping
│   └── initial_test.ipynb  # Interactive quantization testing notebook
├── scripts/                # Utility scripts and automation tools
├── src/                    # Source code
│   └── quantize_terminal.py # Main terminal-based quantization script
├── tests/                  # Unit tests and test files
├── LICENSE                 # MIT License file
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### Directory Descriptions

- **`assets/`**: Store static files like images, icons, fonts, and other media assets.
- **`data/`**: Place datasets, input files, and any data-related resources here.
- **`docs/`**: Additional documentation, guides, and project-related files.
- **`notebooks/`**: Jupyter notebooks for data exploration, prototyping, and demonstrations.
- **`scripts/`**: Utility scripts for automation, setup, deployment, or maintenance tasks.
- **`src/`**: Main source code for the project, including the quantization script.
- **`tests/`**: Unit tests, integration tests, and test-related files.

## 📖 Usage

### Basic Usage

Run the quantization script interactively:

```bash
python src/quantize_terminal.py
```

The script will guide you through:
1. Hugging Face authentication (if not already logged in)
2. Model selection
3. Quantization configuration
4. Output options (local save and/or Hub upload)

### Command Line Options

```bash
# Set Hugging Face token
python src/quantize_terminal.py --set-token YOUR_TOKEN

# Reset Hugging Face token
python src/quantize_terminal.py --reset-token

# Use default settings for quick quantization
python src/quantize_terminal.py --default
```

### Jupyter Notebook

For interactive testing and experimentation:

```bash
jupyter notebook notebooks/initial_test.ipynb
```

### Advanced Usage

The script supports various quantization configurations:

- **Quantization Types**: NF4 (recommended) or FP4
- **Double Quantization**: Reduces memory footprint further
- **Compute Dtypes**: float16, bfloat16, float32
- **Storage Dtypes**: float16, float32, int8, uint8, bfloat16

### Example Output

```
🤗 BitsAndBytes Model Quantizer
========================================
Logged in as: your-username

Configuration:
  Model: microsoft/DialoGPT-medium
  Quantization type: nf4
  Double quantization: True
  Compute dtype: bfloat16
  Storage dtype: uint8
  Output directory: quantized_model

📦 Original Footprint (FP32): 345.67 MB
✅ Quantized: 123.45 MB
📉 Actual Reduction: 64.3%

Model saved to: C:\path\to\Model-Quantization\quantized_model
Successfully pushed to https://huggingface.co/your-username/DialoGPT-medium-bnb-4bit-nf4-dq
```

## ⚙️ Configuration

### Quantization Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| Quantization Type | `nf4`, `fp4` | `nf4` | 4-bit quantization algorithm |
| Double Quantization | `True`, `False` | `True` | Enables double quantization for better compression |
| Compute Dtype | `float16`, `bfloat16`, `float32` | `bfloat16` | Dtype for computations during inference |
| Storage Dtype | `float16`, `float32`, `int8`, `uint8`, `bfloat16` | `uint8` | Dtype for storing quantized weights |

### Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

### Model Compatibility

The script automatically detects model types:
- **Causal LM**: GPT-style models (AutoModelForCausalLM)
- **Masked LM**: BERT-style models (AutoModelForMaskedLM)

### Memory Considerations

- **GPU Memory**: Ensure sufficient VRAM for model loading
- **System RAM**: Large models may require significant RAM during quantization
- **Storage**: Quantized models are typically 60-75% smaller than originals

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install additional dev tools
pip install black flake8 pytest

# Run tests
pytest

# Run linting
black src/
flake8 src/
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and modular

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Links:**
- **GitHub:** [https://github.com/devMuniz02/](https://github.com/devMuniz02/)
- **LinkedIn:** [https://www.linkedin.com/in/devmuniz](https://www.linkedin.com/in/devmuniz)
- **Hugging Face:** [https://huggingface.co/manu02](https://huggingface.co/manu02)
- **Portfolio:** [https://devmuniz02.github.io/](https://devmuniz02.github.io/)

Project Link: [https://github.com/devMuniz02/Model-Quantization](https://github.com/devMuniz02/Model-Quantization)

---

⭐ If you find this project helpful, please give it a star!
