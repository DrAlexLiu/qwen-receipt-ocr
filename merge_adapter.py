#!/usr/bin/env python3
"""
Merge a LoRA / DoRA adapter into its base model.

The merged model is a plain bf16 checkpoint that vLLM and transformers
load directly. This is the way to use the adapters published in the
releases.

Usage:
    python merge_adapter.py \
        --base-model Qwen/Qwen3.5-9B \
        --adapter ./Qwen3.5-9B-receipt-dora \
        --output ./Qwen3.5-9B-receipt-merged
"""

import argparse
import os
import shutil

import torch
import transformers
from huggingface_hub import snapshot_download
from peft import PeftModel

# Files of the base model that training does not change and the merged
# model still needs (tokenizer, processor, chat template).
SKIP = (
    "config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
    ".gitattributes",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA / DoRA adapter into its base model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Local path or HuggingFace ID of the base model",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Directory with adapter_config.json and adapter weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write the merged model to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for merging (default: cuda:0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base = args.base_model
    if not os.path.isdir(base):
        base = snapshot_download(base)

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, device_map=args.device
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(
        args.output, safe_serialization=True, max_shard_size="5GB"
    )
    for name in os.listdir(base):
        src = os.path.join(base, name)
        if (
            os.path.isfile(src)
            and not name.endswith(".safetensors")
            and name not in SKIP
        ):
            shutil.copy(src, os.path.join(args.output, name))
    print(f"Merged model written to {args.output}")


if __name__ == "__main__":
    main()
