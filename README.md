[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# Model Quantization

> Effortlessly quantize and deploy Hugging Face models. This repo provides a streamlined pipeline to convert any HF model into a compressed format and upload it directly to your HF profile. Ideal for optimizing Large Language Models (LLMs) for resource-constrained environments.

- [Features](#features) - [Installation](#installation) - [Repository Setup](#repository-setup) - [Usage](#usage) - [Configuration](#configuration) - [Contributing](#contributing) - [License](#license) - [Contact](#contact)

## Overview

Effortlessly quantize and deploy Hugging Face models. This repo provides a streamlined pipeline to convert any HF model into a compressed format and upload it directly to your HF profile. Ideal for optimizing Large Language Models (LLMs) for resource-constrained environments.

## Repository Structure

| Path | Description |
| --- | --- |
| `notebooks/` | Exploratory notebooks and experiment walkthroughs. |
| `scripts/` | Top-level project directory containing repository-specific resources. |
| `src/` | Primary source code for the application or library. |
| `.gitignore` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |
| `README.md` | Primary project documentation. |
| `requirements.txt` | Python dependency specification for local setup. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/Model-Quantization.git
   cd Model-Quantization
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Use the project-specific scripts or notebooks in the repository root to run the workflow.

## Features

- **Terminal-based Quantization**: Interactive command-line interface for easy model quantization
- **4-bit Quantization Support**: NF4 and FP4 quantization types for optimal compression
- **Double Quantization**: Enhanced compression with double quantization option
- **Memory Footprint Analysis**: Automatic calculation of original vs quantized model sizes
- **Hugging Face Integration**: Direct upload to Hugging Face Hub with generated model cards
- **Flexible Configuration**: Customizable compute and storage data types
- **Local Saving**: Option to save quantized models locally
- **Model Card Generation**: Automatic creation of detailed model cards for quantized models
- **Jupyter Notebook Support**: Interactive notebooks for testing and experimentation
- **Automated Trending Model Quantization**: Batch quantization of top trending models from Hugging Face

## Installation

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

## Usage

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

### Automated Quantization of Trending Models

Run the automated script to quantize trending models:

```bash
python scripts/auto_quantize_trending.py
```

This script will:
1. Fetch the top 50 trending models from Hugging Face
2. Filter models with < 10B parameters (excluding vision, multimodal, etc.)
3. Check against already quantized models in `data/quantized_models.json`
4. Quantize new models using default 4-bit NF4 quantization
5. Upload quantized models to your Hugging Face Hub
6. Update the JSON list with newly quantized models

### Command Line Options for Automated Script

```bash
# Run with Hugging Face token
python scripts/auto_quantize_trending.py --hf-token YOUR_TOKEN

# Limit the number of models to quantize
python scripts/auto_quantize_trending.py --max-models 10
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Links:**
- **GitHub:** [https://github.com/devMuniz02/](https://github.com/devMuniz02/)
- **LinkedIn:** [https://www.linkedin.com/in/devmuniz](https://www.linkedin.com/in/devmuniz)
- **Hugging Face:** [https://huggingface.co/manu02](https://huggingface.co/manu02)
- **Portfolio:** [https://devmuniz02.github.io/](https://devmuniz02.github.io/)

Project Link: [https://github.com/devMuniz02/Model-Quantization](https://github.com/devMuniz02/Model-Quantization)

---

⭐ If you find this project helpful, please give it a star!

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Repository Setup](#repository-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure

```
Model-Quantization/
├── assets/                 # Static assets (images, icons, etc.)
├── data/                   # Data files and datasets
│   └── quantized_models.json # JSON list of already quantized models
├── docs/                   # Documentation files
├── notebooks/              # Jupyter notebooks for analysis and prototyping
│   ├── initial_test.ipynb  # Interactive quantization testing notebook
│   ├── quant_scratch.ipynb # Scratch notebook for quantization experiments
│   └── trendinghfmodels.ipynb # Analysis of trending Hugging Face models
├── scripts/                # Utility scripts and automation tools
│   └── auto_quantize_trending.py # Automated quantization of trending models
├── src/                    # Source code
│   └── quantize_terminal.py # Main terminal-based quantization script
├── tests/                  # Unit tests and test files
├── LICENSE                 # MIT License file
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### Directory Descriptions

- **`assets/`**: Store static files like images, icons, fonts, and other media assets.
- **`data/`**: Place datasets, input files, and any data-related resources here. Contains `quantized_models.json` for tracking quantized models.
- **`docs/`**: Additional documentation, guides, and project-related files.
- **`notebooks/`**: Jupyter notebooks for data exploration, prototyping, and demonstrations.
  - `initial_test.ipynb`: Interactive quantization testing notebook
  - `trendinghfmodels.ipynb`: Analysis and exploration of trending Hugging Face models
- **`scripts/`**: Utility scripts for automation, setup, deployment, or maintenance tasks. Includes `auto_quantize_trending.py` for automated quantization.
- **`src/`**: Main source code for the project, including the quantization script.
- **`tests/`**: Unit tests, integration tests, and test-related files.

## ️ Configuration

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

## Contributing

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
