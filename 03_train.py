#!/usr/bin/env python3
"""
Fine-tune Vision Language Models with MS-SWIFT

This script fine-tunes VLMs using the ModelScope SWIFT framework.

Supported models:
- Qwen/Qwen3-VL-2B/4B/8B/32B-Instruct, Qwen/Qwen3-VL-30B-A3B-Instruct
- Qwen/Qwen3.5-2B/4B/9B/27B (dense), Qwen/Qwen3.5-35B-A3B (MoE)
- Qwen/Qwen3.6-27B (dense), Qwen/Qwen3.6-35B-A3B (MoE)

Usage:
    python 03_train.py
    python 03_train.py --model-id ./models/Qwen3.6-35B-A3B
    python 03_train.py --data-dir ./data --output-dir ./output
    python 03_train.py --num-epochs 4 --batch-size 4
"""

import argparse
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ============================================
# Model Type Mapping
# ============================================

# HF config.json `model_type` -> SWIFT model type
HF_MODEL_TYPE_MAP = {
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_moe": "qwen3_vl_moe",
    "qwen3_5": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
}


def get_model_type(model_id: str) -> str:
    """Get the SWIFT model type from config.json or the model ID."""
    config_path = Path(model_id) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            hf_type = json.load(f).get("model_type")
        if hf_type in HF_MODEL_TYPE_MAP:
            return HF_MODEL_TYPE_MAP[hf_type]

    model_lower = model_id.lower()
    is_moe = (
        "a3b" in model_lower or "a10b" in model_lower or "a22b" in model_lower
    )
    if "qwen3.5" in model_lower or "qwen3.6" in model_lower:
        return "qwen3_5_moe" if is_moe else "qwen3_5"
    if "qwen3-vl" in model_lower or "qwen3_vl" in model_lower:
        return "qwen3_vl_moe" if is_moe else "qwen3_vl"
    raise ValueError(
        f"Unknown model type for: {model_id}. Please specify --model-type"
    )


# DoRA on the Gated DeltaNet `in_proj_qkv` of Qwen3.5/3.6 produces NaN
# loss; every other linear layer is fine, and so is DoRA on Qwen3-VL.
# With DoRA, target all LLM linears except that one.
QWEN3_5_DORA_TARGET_REGEX = (
    r"^(?!.*visual)(?!.*lm_head).*\.("
    r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
    r"in_proj_a|in_proj_b|in_proj_z|out_proj)$"
)


def find_latest_checkpoint(output_dir: str) -> str:
    """Find the latest checkpoint in output directory."""
    checkpoint_path = Path(output_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = list(checkpoint_path.glob("**/checkpoint-*"))
    if not checkpoints:
        return None

    def get_checkpoint_num(p):
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return 0

    checkpoints.sort(key=get_checkpoint_num, reverse=True)
    return str(checkpoints[0])


def setup_environment(args):
    """Setup environment variables for training."""
    os.environ["SIZE_FACTOR"] = str(args.size_factor)
    os.environ["MAX_PIXELS"] = str(args.max_pixels)

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.nproc_per_node:
        os.environ["NPROC_PER_NODE"] = str(args.nproc_per_node)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir


def patch_mp_ddp_compat():
    """Let ms-swift's model-parallel + DDP patch run on transformers 5.

    swift forwards all `infer_auto_device_map` kwargs to
    `get_balanced_memory`, but transformers 5.16 added
    `no_placement_params` to the former only. Drop kwargs the wrapped
    function does not accept. swift binds the function when it patches,
    so this must run before `sft_main`.
    """
    try:
        from transformers.integrations import accelerate as tf_accelerate
    except ImportError:
        return
    orig = tf_accelerate.get_balanced_memory
    accepted = set(inspect.signature(orig).parameters)

    def get_balanced_memory(*args, **kwargs):
        return orig(
            *args, **{k: v for k, v in kwargs.items() if k in accepted}
        )

    tf_accelerate.get_balanced_memory = get_balanced_memory


def run_training(args):
    """Run SWIFT fine-tuning."""
    from swift import sft_main

    patch_mp_ddp_compat()

    # Prepare paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    train_data = data_dir / args.train_file
    val_data = data_dir / args.val_file

    if not train_data.exists():
        print(f"Error: Training data not found: {train_data}")
        sys.exit(1)
    if not val_data.exists():
        print(f"Error: Validation data not found: {val_data}")
        sys.exit(1)

    # Create version directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_dir = output_dir / f"v1-{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Get model type
    model_type = args.model_type or get_model_type(args.model_id)

    # Handle local model path
    model_id = args.model_id
    if os.path.exists(model_id):
        model_id = str(Path(model_id).resolve())
        print(f"Using local model: {model_id}")

    print("=" * 60)
    print("SWIFT Fine-tuning Configuration")
    print("=" * 60)
    print(f"Model ID:      {model_id}")
    print(f"Model Type:    {model_type}")
    print(f"Train Data:    {train_data}")
    print(f"Val Data:      {val_data}")
    print(f"Output Dir:    {version_dir}")
    print(f"Epochs:        {args.num_epochs}")
    max_steps = args.max_steps if args.max_steps > 0 else "unlimited"
    print(f"Max Steps:     {max_steps}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LoRA rank:     {args.lora_rank} (alpha {args.lora_alpha})")
    print(f"LoRA + DoRA:   {args.use_dora}")
    print(f"DeepSpeed:     {args.deepspeed or 'off'}")
    print("=" * 60)

    # Build SWIFT arguments
    argv = [
        "--model_type",
        model_type,
        "--model",
        model_id,
        "--model_revision",
        "main",
        "--tuner_type",
        "lora",
        "--use_dora",
        str(args.use_dora).lower(),
        "--output_dir",
        str(version_dir),
        "--max_length",
        str(args.max_length),
        "--dataset",
        args.train_file,
        "--val_dataset",
        args.val_file,
        "--save_steps",
        str(args.save_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--num_train_epochs",
        str(args.num_epochs),
        "--lora_dtype",
        "bfloat16",
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--per_device_eval_batch_size",
        "1",
        "--learning_rate",
        str(args.learning_rate),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--use_hf",
        "true",
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--save_total_limit",
        str(args.save_total_limit),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--freeze_vit",
        str(args.freeze_vit).lower(),
        "--freeze_llm",
        str(args.freeze_llm).lower(),
        "--freeze_aligner",
        str(args.freeze_aligner).lower(),
        "--attn_impl",
        args.attn_impl,
        "--ddp_find_unused_parameters",
        "false",
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--report_to",
        "tensorboard",
    ]

    if model_type.endswith("_moe"):
        # As in ms-swift's examples/models/qwen3_5/transformers.sh
        argv.extend(
            ["--experts_impl", "grouped_mm", "--router_aux_loss_coef", "1e-3"]
        )

    if args.use_dora and model_type in ("qwen3_5", "qwen3_5_moe"):
        argv.extend(["--target_regex", QWEN3_5_DORA_TARGET_REGEX])
    else:
        argv.extend(["--target_modules", "all-linear"])

    if args.deepspeed:
        argv.extend(["--deepspeed", args.deepspeed])

    if args.gradient_checkpointing:
        argv.extend(["--gradient_checkpointing", "true"])

    if args.max_steps > 0:
        argv.extend(["--max_steps", str(args.max_steps)])

    # Resume from checkpoint
    resume_checkpoint = args.resume_from
    if not resume_checkpoint and args.auto_resume:
        resume_checkpoint = find_latest_checkpoint(str(output_dir))

    if resume_checkpoint:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        argv.extend(["--resume_from_checkpoint", resume_checkpoint])

    print("\nStarting training...")
    print("-" * 60)

    # Change to data directory for relative paths
    original_dir = os.getcwd()
    os.chdir(data_dir)
    print(f"Working directory: {os.getcwd()}")

    try:
        result = sft_main(argv)
    finally:
        os.chdir(original_dir)

    print("-" * 60)
    print("Training completed!")

    if result and "best_model_checkpoint" in result:
        print(f"Best checkpoint: {result['best_model_checkpoint']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Vision LLMs with MS-SWIFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="SWIFT model type (auto-detected if not specified)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="conversations_train_swift_format.json",
        help="Training data filename",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="conversations_dev_swift_format.json",
        help="Validation data filename",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num-epochs", type=int, default=4, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for unlimited)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="Warmup ratio"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha"
    )
    parser.add_argument(
        "--use-dora", action="store_true", default=True, help="Use DoRA"
    )
    parser.add_argument(
        "--no-dora", action="store_false", dest="use_dora", help="Disable DoRA"
    )

    # Freezing
    parser.add_argument("--freeze-vit", action="store_true", default=True)
    parser.add_argument("--freeze-llm", action="store_true", default=False)
    parser.add_argument("--freeze-aligner", action="store_true", default=True)

    # Checkpoints
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--auto-resume", action="store_true", default=False)

    # Resources
    parser.add_argument("--size-factor", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--cuda-devices", type=str, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=False
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed preset (zero2, zero3) or config path; "
        "zero3 is needed when the model does not fit one GPU",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        help="Attention implementation (sdpa, flash_attn, eager)",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4)

    args = parser.parse_args()

    setup_environment(args)

    try:
        run_training(args)
        print("\n" + "=" * 60)
        print("Fine-tuning completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
