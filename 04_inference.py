#!/usr/bin/env python3
"""
Batch Inference for Fine-tuned Vision Language Models

Uses vLLM directly for fast batch inference.
Supports LoRA/DoRA fine-tuned models via merge-then-infer.

Usage:
    # With fine-tuned model (auto-find latest checkpoint)
    python 04_inference.py --model-dir ./output

    # With specific checkpoint
    python 04_inference.py --model-dir ./output/v1-xxx/checkpoint-250

    # With JSON schema for constrained decoding
    python 04_inference.py --model-dir ./output \
        --json-schema ./data/groundtruth_schema.json

    # Using base model (no fine-tuning)
    python 04_inference.py --use-base-model \
        --model-id ./models/Qwen3.5-9B
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# vLLM environment (must be before import)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_best_checkpoint(model_dir: str) -> Optional[str]:
    """Find the best/latest checkpoint in model directory."""
    model_path = Path(model_dir)

    if (model_path / "adapter_config.json").exists():
        return str(model_path)

    checkpoints = [
        p
        for p in model_path.glob("**/checkpoint-*")
        if (p / "adapter_config.json").exists()
    ]
    if not checkpoints:
        return None

    def get_num(p):
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return 0

    checkpoints.sort(key=get_num, reverse=True)
    return str(checkpoints[0])


def find_base_model(checkpoint_dir: str) -> Optional[str]:
    """Find base model path from checkpoint's args.json."""
    args_file = Path(checkpoint_dir) / "args.json"
    if args_file.exists():
        with open(args_file) as f:
            args = json.load(f)
            for key in ["model", "model_id_or_path", "model_name_or_path"]:
                if key in args and args[key]:
                    return args[key]
    return None


def merge_lora(checkpoint_dir: str) -> str:
    """Merge LoRA adapter into base model using SWIFT export.

    The merged model is cached next to the checkpoint as
    `<checkpoint>-merged`, so repeated inference runs reuse it.
    """
    merged_dir = checkpoint_dir.rstrip("/") + "-merged"
    if (Path(merged_dir) / "config.json").exists():
        print(f"  Using existing merged model: {merged_dir}")
        return merged_dir

    from swift import export_main

    print("  Running SWIFT export (merge_lora)...")
    export_main(
        [
            "--adapters",
            checkpoint_dir,
            "--merge_lora",
            "true",
            "--output_dir",
            merged_dir,
            "--use_hf",
            "true",
        ]
    )

    print(f"  Merged model saved to: {merged_dir}")
    return merged_dir


def build_vl_messages(sample: dict, data_dir: Path, max_pixels: int) -> list:
    """Convert SWIFT training format to Qwen VL message format."""
    messages = []
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            break
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            content = []
            for img_path in sample.get("images", []):
                full_path = str(data_dir / img_path)
                content.append(
                    {
                        "type": "image",
                        "image": f"file://{full_path}",
                        "max_pixels": max_pixels,
                    }
                )
            text = msg["content"].replace("<image>", "").strip()
            content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
    return messages


def run_inference(args):
    """Run batch inference using vLLM directly."""
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    test_data_path = data_dir / args.test_file

    if not test_data_path.exists():
        print(f"Error: Test data not found: {test_data_path}")
        sys.exit(1)

    with open(test_data_path) as f:
        test_data = json.load(f)
    if args.limit > 0:
        test_data = test_data[: args.limit]

    # Output setup
    output_dir = Path(args.output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = output_dir / f"inference-{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "results.jsonl"

    # Determine model path
    if args.use_base_model:
        model_id = args.model_id
        if os.path.exists(model_id):
            model_id = str(Path(model_id).resolve())
        model_path = model_id
        checkpoint_dir = None
    else:
        checkpoint_dir = find_best_checkpoint(args.model_dir)
        if not checkpoint_dir:
            print(f"Error: No checkpoint found in {args.model_dir}")
            sys.exit(1)

        base_model = find_base_model(checkpoint_dir) or args.model_id

        print("=" * 60)
        print("Merging LoRA Adapter")
        print("=" * 60)
        print(f"  Base Model:  {base_model}")
        print(f"  Checkpoint:  {checkpoint_dir}")

        merge_start = time.time()
        model_path = merge_lora(checkpoint_dir)
        print(f"  Merge time:  {time.time() - merge_start:.1f}s")
        if args.merge_only:
            result_dir.rmdir()
            return model_path

    num_gpus = args.tensor_parallel_size or torch.cuda.device_count()

    # Print config
    print("=" * 60)
    print("Batch Inference Configuration (vLLM)")
    print("=" * 60)
    print(f"Model:         {model_path}")
    if checkpoint_dir:
        print(f"Checkpoint:    {checkpoint_dir}")
        print("Mode:          Fine-tuned (LoRA merged)")
    else:
        print("Mode:          Base model")
    print(f"Test Samples:  {len(test_data)}")
    print(f"Result File:   {result_path}")
    print(f"GPUs:          tensor_parallel_size={num_gpus}")

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )

    print("Loading vLLM engine...")
    load_start = time.time()

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        # vLLM's custom P2P all-reduce can fail on some CUDA setups; the
        # NCCL fallback works.
        disable_custom_all_reduce=num_gpus > 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )

    print(f"Engine loaded in {time.time() - load_start:.1f}s")
    print("=" * 60)

    sampling_kwargs = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.json_schema:
        with open(args.json_schema) as f:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=json.load(f)
            )
        print(f"JSON Schema:   {args.json_schema}")
    sampling_params = SamplingParams(**sampling_kwargs)

    # Prepare all inputs
    print("Preparing inputs...")
    prompts = []
    for sample in test_data:
        vl_messages = build_vl_messages(sample, data_dir, args.max_pixels)

        # Training targets carry an empty <think></think> block, i.e.
        # the non-thinking mode of Qwen3.5/3.6; match it at inference.
        text = processor.apply_chat_template(
            vl_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        image_inputs, _ = process_vision_info(
            vl_messages,
            image_patch_size=processor.image_processor.patch_size,
        )

        prompts.append(
            {
                "prompt": text,
                "multi_modal_data": {"image": image_inputs},
            }
        )

    # Run batch inference
    print(f"Running inference on {len(prompts)} samples...")
    print("-" * 60)

    infer_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - infer_start

    print("-" * 60)
    per_sample = infer_time / len(prompts)
    print(f"Inference: {infer_time:.1f}s total, {per_sample:.2f}s/sample")

    # Save results (compatible with 05_evaluate.py)
    with open(result_path, "w") as f:
        for sample, output in zip(test_data, outputs):
            response = output.outputs[0].text

            gt = None
            for msg in sample["messages"]:
                if msg["role"] == "assistant":
                    gt = msg["content"]
                    break

            result = {
                "response": response,
                "labels": gt,
                "images": sample.get("images", []),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Save config
    config = {
        "model_path": model_path,
        "checkpoint": checkpoint_dir,
        "test_file": str(test_data_path),
        "n_samples": len(test_data),
        "max_length": args.max_length,
        "max_tokens": args.max_tokens,
        "max_pixels": args.max_pixels,
        "temperature": args.temperature,
        "json_schema": args.json_schema,
        "inference_time_seconds": round(infer_time, 1),
        "time_per_sample": round(infer_time / len(prompts), 2),
        "num_gpus": num_gpus,
        "timestamp": timestamp,
    }
    with open(result_dir / "inference_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    return result_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with vLLM for fine-tuned Vision LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing fine-tuned model",
    )
    model_group.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use base model (no LoRA)",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="Base model ID (fallback if not in args.json)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="conversations_test_swift_format.json",
        help="Test data filename",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge the adapter (run before vLLM in a "
        "separate process so merge memory is released)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N samples (0 = all)",
    )

    # Constrained decoding
    parser.add_argument(
        "--json-schema",
        type=str,
        default=None,
        help="Path to JSON schema for constrained decoding",
    )

    # Inference
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum output tokens"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=602112,
        help="Image resize cap; keep equal to training MAX_PIXELS",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )

    # vLLM
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (default: all visible GPUs)",
    )

    # GPU
    parser.add_argument(
        "--cuda-devices", type=str, default=None, help="CUDA visible devices"
    )

    args = parser.parse_args()

    if not args.use_base_model and not args.model_dir:
        parser.error(
            "Either --model-dir or --use-base-model must be specified"
        )

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    try:
        result_path = run_inference(args)
        print("\n" + "=" * 60)
        print("Batch inference completed!")
        print(f"Results: {result_path}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during inference: {e}")
        raise


if __name__ == "__main__":
    main()
