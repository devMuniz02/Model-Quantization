#!/usr/bin/env python3
"""
Terminal-based BitsAndBytes Quantizer

This script quantizes a HuggingFace model using BitsAndBytes 4-bit quantization
and saves the quantized model locally.

Run the script and follow the interactive prompts to configure quantization settings.
"""

import os
import sys
from typing import Optional
import argparse
import tempfile

import torch
from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM
from huggingface_hub import login, logout, HfApi, ModelCard, ModelCardData
import bitsandbytes.nn
from bitsandbytes.nn import Linear4bit


# DTYPE_MAPPING remains the same
DTYPE_MAPPING = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def get_model_size(model) -> float:
    """
    Calculate the size of a PyTorch model in gigabytes.

    Args:
        model: PyTorch model

    Returns:
        float: Size of the model in GB
    """
    # Get model state dict
    state_dict = model.state_dict()

    # Calculate total size in bytes
    total_size = 0
    for param in state_dict.values():
        # Calculate bytes for each parameter
        total_size += param.nelement() * param.element_size()

    # Convert bytes to gigabytes (1 GB = 1,073,741,824 bytes)
    size_gb = total_size / (1024 ** 3)
    size_gb = round(size_gb, 2)

    return size_gb


def create_model_card(
    model_name: str,
    quant_type_4: str,
    double_quant_4: str,
    compute_type_4: str,
    quant_storage_4: str,
    orig_mem: float,
    quant_mem: float,
    dtype_name: str,
    reduction: float,
) -> str:
    """
    Create a model card for the quantized model.

    Args:
        model_name: Original model name
        quant_type_4: Quantization type
        double_quant_4: Double quantization setting
        compute_type_4: Compute dtype
        quant_storage_4: Storage dtype
        orig_mem: Original memory in MB
        quant_mem: Quantized memory in MB
        dtype_name: Original dtype name
        reduction: Memory reduction percentage

    Returns:
        str: Model card content
    """
    # Create YAML header
    yaml_header = f"""---
base_model: {model_name}
language: en
license: apache-2.0
tags:
- quantized
- 4bit
- bnb
- transformers
model_name: {model_name.split('/')[-1]}-bnb-4bit-{quant_type_4}
---
"""
    
    # Create content
    content = f"""
# {model_name.split('/')[-1]} (Quantized)

## Description
This model is a 4-bit quantized version of the original [`{model_name}`](https://huggingface.co/{model_name}) model, optimized for reduced memory usage while maintaining performance.

## Quantization Details
- **Quantization Type**: 4-bit
- **bnb_4bit_quant_type**: {quant_type_4}
- **bnb_4bit_use_double_quant**: {double_quant_4}
- **bnb_4bit_compute_dtype**: {compute_type_4}
- **bnb_4bit_quant_storage**: {quant_storage_4}
- **Original Footprint**: {orig_mem:.2f} MB ({dtype_name})
- **Quantized Footprint**: {quant_mem:.2f} MB
- **Memory Reduction**: {reduction:.1f}%

## Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

model_name = "{model_name.split('/')[-1]}-bnb-4bit-{quant_type_4}"
model = AutoModelForSequenceClassification.from_pretrained(
    "your-username/{model_name}",  # Replace with your HF username
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
tokenizer = AutoTokenizer.from_pretrained("your-username/{model_name}")
```
"""
    
    return yaml_header + content


def handle_token_args(args):
    """
    Handle token-related command line arguments.
    """
    if args.reset_token:
        print("Resetting Hugging Face token...")
        logout()
        print("Token reset successfully.")
        return True  # Exit after reset
    
    if args.set_token:
        print("Setting Hugging Face token...")
        login(token=args.set_token)
        print("Token set successfully.")
        return True  # Exit after set
    
    return False  # Continue with normal flow


def quantize_model(
    model_name: str,
    quant_type_4: str,
    double_quant_4: bool,
    compute_type_4: str,
    quant_storage_4: str,
) -> tuple:
    """
    Quantize a model using BitsAndBytes.

    Args:
        model_name: HuggingFace model ID
        quant_type_4: Quantization type ('fp4' or 'nf4')
        double_quant_4: Whether to use double quantization
        compute_type_4: Compute dtype
        quant_storage_4: Storage dtype

    Returns:
        tuple: (quantized_model, original_memory_mb, dtype_name, hf_save_name)
    """
    print("Detecting native dtype and calculating original footprint...")

    # --- 1. DETECT NATIVE DTYPE & CALC ORIGINAL FOOTPRINT ---
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Check what the model's actual native dtype is
    native_dtype = getattr(config, "dtype", torch.float32)
    if native_dtype is None: native_dtype = torch.float32 # Default if not specified
    
    # Virtual load on 'meta' device using the native dtype
    with torch.device("meta"):
        is_masked = any(arch in str(config.architectures) for arch in ["Masked", "Bert", "Roberta"])
        model_class = AutoModelForMaskedLM if is_masked else AutoModelForCausalLM
        temp_model = model_class.from_config(config, dtype=native_dtype)
    
    orig_mem = temp_model.get_memory_footprint() / 1e6  # MB
    dtype_name = str(native_dtype).split('.')[-1].upper()
    print(f"📦 Original Footprint ({dtype_name}): {orig_mem:.2f} MB")

    # --- 2. CONFIGURATION & NAMING ---
    dq_init = "-dq" if double_quant_4 else ""
    type_init = f"-{quant_type_4}"
    # Generates name like: bert-base-uncased-bnb-4bit-nf4-dq
    hf_save_name = f"{model_name.split('/')[-1]}-bnb-4bit{type_init}{dq_init}"
    print(f"🏷️ Suggested HF Name: {hf_save_name}")

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type_4,
        bnb_4bit_use_double_quant=double_quant_4,
        bnb_4bit_quant_storage=DTYPE_MAPPING[quant_storage_4],
        bnb_4bit_compute_dtype=DTYPE_MAPPING[compute_type_4],
    )

    print("Loading quantized model...")

    # Load model
    model = model_class.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Quantization completed.")
    return model, orig_mem, dtype_name, hf_save_name


def save_model_locally(
    model,
    tokenizer,
    model_name: str,
    orig_mem: float,
    dtype_name: str,
    hf_save_name: str,
    quant_type_4: str,
    double_quant_4: str,
    compute_type_4: str,
    quant_storage_4: str,
    output_dir: str = "quantized_model",
) -> str:
    """
    Save the quantized model locally.

    Args:
        model: Quantized model
        tokenizer: Model tokenizer
        model_name: Original model name
        orig_mem: Original model memory in MB
        dtype_name: Original dtype name
        hf_save_name: Suggested name for Hugging Face Hub
        quant_type_4: Quantization type
        double_quant_4: Double quantization setting
        compute_type_4: Compute dtype
        quant_storage_4: Storage dtype
        output_dir: Directory to save the model

    Returns:
        str: Success message with local path info
    """
    # --- 4. FINAL PRECISION-AWARE REPORT ---
    quant_mem = model.get_memory_footprint() / 1e6
    reduction = ((orig_mem - quant_mem) / orig_mem) * 100

    print(f"\n📊 COMPARISON:")
    print(f"📦 Original ({dtype_name}):  {orig_mem:.2f} MB")
    print(f"✅ Quantized:           {quant_mem:.2f} MB")
    print(f"📉 Actual Reduction:    {reduction:.1f}%")

    print(f"\n💾 Saving to {output_dir}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model and tokenizer
    print("Saving model...")
    tokenizer.save_pretrained(output_dir, safe_serialization=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    # Create model card
    print("Creating model card...")
    model_card = create_model_card(
        model_name, quant_type_4, str(double_quant_4), compute_type_4, quant_storage_4, orig_mem, quant_mem, dtype_name, reduction
    )
    with open(os.path.join(output_dir, "README.md"), "w", encoding='utf-8') as f:
        f.write(model_card)

    print("Model saved successfully!")

    # Push to Hub option
    push_to_hub = get_yes_no("Do you want to push the model to Hugging Face Hub?", default=False)
    if push_to_hub:
        commit_msg = f"Upload 4-bit quantized version of {model_name} with {reduction:.1f}% memory reduction"
        print(f"Pushing to {hf_save_name}...")
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Copy files from output_dir to tmpdirname
            import shutil
            for file in os.listdir(output_dir):
                shutil.copy(os.path.join(output_dir, file), tmpdirname)
            
            # Push to Hub
            api = HfApi()
            user = api.whoami()
            username = user['name']
            repo_id = f"{username}/{hf_save_name}"
            api.create_repo(repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=tmpdirname,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg
            )
        
        print(f"Successfully pushed to https://huggingface.co/{repo_id}")

    return f"""
Quantization completed successfully!

Model saved to: {os.path.abspath(output_dir)}
"""


def get_user_input(prompt: str, options: list = None, default: str = None) -> str:
    """
    Get user input with optional options and default value.

    Args:
        prompt: The prompt to display
        options: List of valid options
        default: Default value if user presses enter

    Returns:
        str: User input
    """
    if options:
        prompt += "\n"
        for i, option in enumerate(options, 1):
            prompt += f"{i}. {option}\n"
        prompt += f"Enter choice (1-{len(options)}) or type the option"
    if default:
        prompt += f" [default: {default}]"
    prompt += ": "

    while True:
        user_input = input(prompt).strip()
        if not user_input and default:
            return default
        if options:
            # Check if it's a number
            try:
                choice = int(user_input)
                if 1 <= choice <= len(options):
                    return options[choice - 1]
            except ValueError:
                pass
            # Check if it's the full option
            if user_input in options:
                return user_input
            print(f"Invalid option. Please choose 1-{len(options)} or type one of: {', '.join(options)}")
        else:
            return user_input


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Get yes/no input from user.

    Args:
        prompt: The prompt to display
        default: Default value (True for yes, False for no)

    Returns:
        bool: True for yes, False for no
    """
    default_text = "Y/n" if default else "y/N"
    prompt += f" ({default_text}): "

    while True:
        user_input = input(prompt).strip().lower()
        if not user_input:
            return default
        if user_input in ['y', 'yes', 'true', '1']:
            return True
        if user_input in ['n', 'no', 'false', '0']:
            return False
        print("Please enter y/yes or n/no")


def main():
    parser = argparse.ArgumentParser(description="BitsAndBytes Model Quantizer")
    parser.add_argument("--set-token", type=str, help="Set Hugging Face token")
    parser.add_argument("--reset-token", action="store_true", help="Reset Hugging Face token")
    parser.add_argument("--default", action="store_true", help="Use default settings: microsoft/DialoGPT-medium, nf4, double quant, bfloat16 compute, uint8 storage, quantized_model dir, no local save, push to hub")
    
    args = parser.parse_args()
    
    # Handle token arguments
    if handle_token_args(args):
        return
    
    print("🤗 BitsAndBytes Model Quantizer")
    print("=" * 40)
    
    # Check if logged in to Hugging Face
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("Not logged in to Hugging Face. Please enter your token:")
        token = input("Token: ").strip()
        if token:
            login(token=token)
            print("Logged in successfully!")
        else:
            print("No token provided. You may need to log in later for pushing to Hub.")
    
    if args.default:
        # Use defaults
        model_name = "microsoft/DialoGPT-medium"
        quant_type = "nf4"
        double_quant = True
        compute_dtype = "bfloat16"
        storage_dtype = "uint8"
        output_dir = "quantized_model"
        save_locally = False
        push_to_hub = True
    else:
        # Get model name
        model_name = input("Enter the HuggingFace model name (e.g., microsoft/DialoGPT-medium): ").strip()
        if not model_name:
            print("Error: Model name is required.")
            sys.exit(1)

        # Get quantization type
        quant_type = get_user_input(
            "Select quantization type",
            options=["nf4", "fp4"],
            default="nf4"
        )

        # Get double quantization
        double_quant = get_yes_no("Enable double quantization", default=True)

        # Get compute dtype
        compute_dtype = get_user_input(
            "Select compute dtype",
            options=["float16", "bfloat16", "float32"],
            default="bfloat16"
        )

        # Get storage dtype
        storage_dtype = get_user_input(
            "Select storage dtype",
            options=["float16", "float32", "int8", "uint8", "bfloat16"],
            default="uint8"
        )

        # Get output directory
        output_dir = input("Enter output directory (default: quantized_model): ").strip()
        if not output_dir:
            output_dir = "quantized_model"

        save_locally = True  # Will ask later
        push_to_hub = False  # Will ask later

    print("\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Quantization type: {quant_type}")
    print(f"  Double quantization: {double_quant}")
    print(f"  Compute dtype: {compute_dtype}")
    print(f"  Storage dtype: {storage_dtype}")
    print(f"  Output directory: {output_dir}")
    if not args.default:
        print(f"  Save locally: {save_locally}")
        print(f"  Push to Hub: {push_to_hub}")
    print()

    # Confirm
    if not get_yes_no("Proceed with quantization", default=True):
        print("Operation cancelled.")
        sys.exit(0)

    try:
        # Quantize model
        quantized_model, orig_mem, dtype_name, hf_save_name = quantize_model(
            model_name,
            quant_type,
            double_quant,
            compute_dtype,
            storage_dtype,
        )

        # Calculate quantized memory
        quant_mem = quantized_model.get_memory_footprint() / 1e6
        reduction = ((orig_mem - quant_mem) / orig_mem) * 100

        print(f"\n📊 COMPARISON:")
        print(f"📦 Original ({dtype_name}):  {orig_mem:.2f} MB")
        print(f"✅ Quantized:           {quant_mem:.2f} MB")
        print(f"📉 Actual Reduction:    {reduction:.1f}%")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if not args.default:
            # Ask if save locally
            save_locally = get_yes_no("Do you want to save the model locally?", default=True)
        
        if save_locally:
            # Save locally
            result = save_model_locally(
                quantized_model,
                tokenizer,
                model_name,
                orig_mem,
                dtype_name,
                hf_save_name,
                quant_type,
                str(double_quant),
                compute_dtype,
                storage_dtype,
                output_dir,
            )
            print(result)
        else:
            print("Skipping local save.")

        if args.default:
            push_to_hub = True
        elif not args.default:
            # Push to Hub option
            push_to_hub = get_yes_no("Do you want to push the model to Hugging Face Hub?", default=False)
        
        if push_to_hub:
            # Calculate reduction for commit message and model card
            quant_mem = quantized_model.get_memory_footprint() / 1e6
            reduction = ((orig_mem - quant_mem) / orig_mem) * 100
            
            # Create model card for push
            model_card = create_model_card(
                model_name, quant_type, str(double_quant), compute_dtype, storage_dtype, orig_mem, quant_mem, dtype_name, reduction
            )
            
            commit_msg = f"Upload 4-bit quantized version of {model_name} with {reduction:.1f}% memory reduction"
            print(f"Pushing to {hf_save_name}...")
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Save model and tokenizer
                tokenizer.save_pretrained(tmpdirname, safe_serialization=True)
                quantized_model.save_pretrained(tmpdirname, safe_serialization=True)
                
                # Save README
                with open(os.path.join(tmpdirname, "README.md"), "w", encoding='utf-8') as f:
                    f.write(model_card)
                
                # Push to Hub
                api = HfApi()
                user = api.whoami()
                username = user['name']
                repo_id = f"{username}/{hf_save_name}"
                api.create_repo(repo_id, exist_ok=True)
                api.upload_folder(
                    folder_path=tmpdirname,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_msg
                )
            
            print(f"Successfully pushed to https://huggingface.co/{repo_id}")

        # Clean up
        del quantized_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()