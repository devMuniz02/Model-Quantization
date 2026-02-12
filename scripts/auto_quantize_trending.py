#!/usr/bin/env python3
"""
Automated Quantization Script for Trending Models

This script:
1. Fetches top 50 trending models from Hugging Face
2. Filters base and fine-tuned models with < 10B parameters
3. Checks against a JSON list of already quantized models
4. Quantizes new models using BitsAndBytes 4-bit quantization
5. Pushes quantized models to Hugging Face Hub
6. Updates the JSON list with newly quantized models
"""

import os
import sys
import json
import tempfile
from typing import List, Dict, Any
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi, login, ModelCard, ModelCardData, scan_cache_dir
import bitsandbytes.nn

# Skip certain model types and keywords
SKIP_MODEL_TYPES = {
    "qwen3_vl",
    "paddleocr_vl", 
    "vibevoice",      # unsupported in transformers right now
}

SKIP_KEYWORDS = ("vl", "vision", "multimodal", "asr", "ocr", "speech")

# DTYPE_MAPPING
DTYPE_MAPPING = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Path to quantized models JSON
QUANTIZED_MODELS_JSON = "data/quantized_models.json"

def get_trending_models() -> List[Dict[str, Any]]:
    """
    Fetch top 50 trending models and filter base/fine-tuned < 10B params.

    Returns:
        List of dicts with model info: {'id': str, 'params': float}
    """
    api = HfApi()
    trending = api.list_models(
        sort="trending_score",
        limit=100,
        expand=["safetensors", "cardData"]
    )

    filtered_models = []
    for model in trending:
        m_id = model.id
        tags = [t.lower() for t in (model.tags or [])]

        # Classification logic
        is_lora = "lora" in tags or "peft" in tags or "adapter" in m_id.lower() or "lora" in m_id.lower()
        is_gguf = "gguf" in tags or m_id.lower().endswith(".gguf") or "gguf" in m_id.lower()
        quant_tags = {"awq", "gptq", "exl2", "bitsandbytes", "quantized"}
        is_quant = any(q in tags for q in quant_tags) or "quant" in m_id.lower()
        base_model_pointer = getattr(model.cardData, 'base_model', None)

        if is_lora or is_gguf or is_quant:
            continue  # Skip these categories

        category = "Fine-tuned" if base_model_pointer else "Base"

        # Parameter extraction
        raw_params = 0
        if hasattr(model, 'safetensors') and model.safetensors:
            raw_params = model.safetensors.get('total', 0)

        if category in ["Base", "Fine-tuned"] and 0 < raw_params < 5e9:
            filtered_models.append({
                'id': m_id,
                'params': raw_params,
                'category': category
            })

    return filtered_models

def load_quantized_models() -> Dict[str, str]:
    """
    Load dict of model IDs with their status from JSON.

    Returns:
        Dict of model_id: status
    """
    if os.path.exists(QUANTIZED_MODELS_JSON):
        with open(QUANTIZED_MODELS_JSON, 'r') as f:
            data = json.load(f)
            return data.get('models', {})
    return {}

def save_quantized_models(quantized_models: Dict[str, str]):
    """
    Save updated dict of model IDs with status to JSON.
    """
    os.makedirs(os.path.dirname(QUANTIZED_MODELS_JSON), exist_ok=True)
    with open(QUANTIZED_MODELS_JSON, 'w') as f:
        json.dump({'models': quantized_models}, f, indent=2)

def quantize_model(
    model_name: str,
    quant_type_4: str = "nf4",
    double_quant_4: bool = True,
    compute_type_4: str = "bfloat16",
    quant_storage_4: str = "uint8",
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
        tuple: (quantized_model, tokenizer, original_memory_mb, dtype_name, hf_save_name)
    """
    print(f"Quantizing {model_name}...")

    # Detect native dtype
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    native_dtype = getattr(config, "dtype", torch.float32)
    if native_dtype is None:
        native_dtype = torch.float32

    # Check for skip conditions
    model_type = (getattr(config, "model_type", "") or "").lower()
    if model_type in SKIP_MODEL_TYPES:
        raise ValueError(f"Skipping {model_name}: unsupported model_type={model_type}")
    if any(k in model_name.lower() for k in SKIP_KEYWORDS):
        raise ValueError(f"Skipping {model_name}: contains skip keyword")

    with torch.device("meta"):
        # Try to create temp model for memory calculation
        try:
            temp_model = AutoModelForCausalLM.from_config(config, dtype=native_dtype, trust_remote_code=True)
        except Exception:
            temp_model = AutoModel.from_config(config, dtype=native_dtype, trust_remote_code=True)

    orig_mem = temp_model.get_memory_footprint() / 1e6  # MB
    dtype_name = str(native_dtype).split('.')[-1].upper()

    # Naming
    dq_init = "-dq" if double_quant_4 else ""
    type_init = f"-{quant_type_4}"
    hf_save_name = f"{model_name.split('/')[-1]}-bnb-4bit{type_init}{dq_init}"

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type_4,
        bnb_4bit_use_double_quant=double_quant_4,
        bnb_4bit_quant_storage=DTYPE_MAPPING[quant_storage_4],
        bnb_4bit_compute_dtype=DTYPE_MAPPING[compute_type_4],
    )

    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Quantization completed for {model_name}.")
    return model, tokenizer, orig_mem, dtype_name, hf_save_name

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
    username: str,
) -> str:
    """
    Create a model card for the quantized model.
    """
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
- **Quantized Footprint**: {quant_mem:.2f} MB ({quant_storage_4.upper()})
- **Memory Reduction**: {reduction:.1f}%

## Usage
```python
from transformers import AutoModel, AutoTokenizer

model_name = "{model_name.split('/')[-1]}-bnb-4bit-{quant_type_4}"
model = AutoModel.from_pretrained(
    "{username}/{model_name.split('/')[-1]}-bnb-4bit-{quant_type_4}",
)
tokenizer = AutoTokenizer.from_pretrained("{username}/{model_name.split('/')[-1]}-bnb-4bit-{quant_type_4}", use_fast=True)
```
"""

    return yaml_header + content

def push_to_hub(model, tokenizer, model_name: str, hf_save_name: str, orig_mem: float, dtype_name: str):
    """
    Push the quantized model to Hugging Face Hub.
    """
    quant_mem = model.get_memory_footprint() / 1e6
    reduction = ((orig_mem - quant_mem) / orig_mem) * 100

    api = HfApi()
    user = api.whoami()
    username = user['name']

    model_card = create_model_card(
        model_name, "nf4", "True", "bfloat16", "uint8", orig_mem, quant_mem, dtype_name, reduction, username
    )

    commit_msg = f"Upload 4-bit quantized version of {model_name} with {reduction:.1f}% memory reduction"
    print(f"Pushing {hf_save_name} to Hugging Face Hub...")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tokenizer.save_pretrained(tmpdirname, safe_serialization=True)
        model.save_pretrained(tmpdirname, safe_serialization=True)

        with open(os.path.join(tmpdirname, "README.md"), "w", encoding='utf-8') as f:
            f.write(model_card)

        repo_id = f"{username}/{hf_save_name}"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )

    print(f"Successfully pushed to https://huggingface.co/{repo_id}")

    # Clean entire cache to free space
    cache_info = scan_cache_dir()
    delete_strategy = cache_info.delete_revisions(*[
        revision.commit_hash 
        for repo in cache_info.repos 
        for revision in repo.revisions
    ])
    print(f"Cleaning cache...")
    print(f"Total space to be freed: {delete_strategy.expected_freed_size_str}")
    delete_strategy.execute()
    print("Model cache completely cleaned!")

    # Clean up
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Automated Quantization of Trending Models")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--max-models", type=int, default=100, help="Maximum number of models to quantize in this run")

    args = parser.parse_args()

    # Login to HF
    if args.hf_token:
        login(token=args.hf_token)
    else:
        try:
            api = HfApi()
            user = api.whoami()
            print(f"Logged in as: {user['name']}")
        except:
            print("Please provide --hf-token or login manually.")
            sys.exit(1)

    # Get trending models
    print("Fetching trending models...")
    trending_models = get_trending_models()
    print(f"Found {len(trending_models)} eligible models.")

    # Load already processed
    quantized_models = load_quantized_models()
    quantized_count = len([k for k, v in quantized_models.items() if v == "quantized"])
    print(f"Already quantized: {quantized_count} models.")

    # Find new models
    new_models = [m for m in trending_models if m['id'] not in quantized_models]
    # Sort by parameter count ascending (smallest first)
    new_models.sort(key=lambda x: x['params'])
    print(f"New models to quantize: {len(new_models)}")

    # Limit to max_models
    new_models = new_models[:args.max_models]

    for model_info in new_models:
        model_id = model_info['id']
        try:
            print(f"Processing {model_id}...")
            # Quantize
            model, tokenizer, orig_mem, dtype_name, hf_save_name = quantize_model(model_id)

            # Push
            push_to_hub(model, tokenizer, model_id, hf_save_name, orig_mem, dtype_name)

            # Update JSON
            quantized_models[model_id] = "quantized"
            save_quantized_models(quantized_models)

            print(f"Completed {model_id}")

        except Exception as e:
            error_str = str(e)
            if "custom code" in error_str.lower():
                print(f"Skipping {model_id}: Requires custom code not supported by this script.")
                quantized_models[model_id] = "skipped"
            else:
                print(f"Error processing {model_id}: {e}")
                quantized_models[model_id] = "failed"
            save_quantized_models(quantized_models)
            continue

    print("Automation completed.")

if __name__ == "__main__":
    main()