#!/usr/bin/env python3
"""
Create SWIFT-format Dataset from FATURA Raw Data

Converts the FATURA invoice dataset directly to the ModelScope SWIFT
format: reads the raw data and writes the training files.

Usage:
    python 02_create_dataset.py
    python 02_create_dataset.py --input-dir ./data-raw \
        --output-dir ./data

    # With data augmentation
    python 02_create_dataset.py --augment --aug-copies 2
"""

import argparse
import io
import json
import mimetypes
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from genson import SchemaBuilder
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

SYSTEM_PROMPT = "You are a document processing expert and assistant."

# ============================================
# Image Augmentation Functions
# ============================================


def augment_image(image: Image.Image) -> Image.Image:
    """Apply data augmentation to a document image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Random brightness (0.85-1.15)
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.85, 1.15)
        )

    # Random contrast (0.85-1.15)
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.85, 1.15)
        )

    # Random slight rotation (±3°)
    if random.random() < 0.3:
        image = image.rotate(
            random.uniform(-3, 3), fillcolor="white", expand=False
        )

    # Random Gaussian blur
    if random.random() < 0.15:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2))
        )

    # Random noise
    if random.random() < 0.15:
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, 8, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)

    return image


# ============================================
# Data Loading Functions
# ============================================


def load_fatura_split(
    csv_path: str,
    images_dir: str,
    annotations_dir: str,
    template_indices: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Load FATURA data from CSV and process into DataFrame."""
    df = pd.read_csv(csv_path)

    # Filter by template indices if specified (for Strategy 2)
    if template_indices:
        df = df[
            df["img_path"].apply(
                lambda x: (
                    int(x.split("_")[0].split("Template")[1])
                    in template_indices
                )
            )
        ]

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Loading"):
        try:
            img_path = os.path.join(images_dir, row["img_path"])
            annot_path = os.path.join(annotations_dir, row["annot_path"])

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            with open(annot_path, "r") as f:
                annotations = json.load(f)
                annotations.pop("OTHER", None)

            results.append(
                {
                    "filename": Path(img_path).name,
                    "filetype": mimetypes.guess_type(img_path)[0],
                    "target_data": json.dumps(annotations),
                    "doc_bytes": img_bytes,
                }
            )
        except Exception as e:
            print(f"    Warning: Error processing {row['img_path']}: {e}")

    return pd.DataFrame(results)


# ============================================
# SWIFT Format Conversion
# ============================================


def transform_annotations(data: Any, defaults: List[str] = None) -> Dict:
    """Transform annotations to clean format."""
    if isinstance(data, dict):
        if defaults:
            for k in defaults:
                if k not in data:
                    data[k] = None

        # Remove bbox fields
        data.pop("bbox", None)

        for key, value in list(data.items()):
            data[key] = transform_annotations(value)

        # Simplify single-value dicts
        if len(data) == 1:
            return list(data.values())[0]
        return data

    elif isinstance(data, list):
        return [
            transform_annotations(item) for item in data if item is not None
        ]

    return data


def create_swift_example(
    images: List[str], target_data: Dict, keys: List[str]
) -> Dict:
    """Create SWIFT-compatible training example."""
    image_tags = "Document pages: <image>" * len(images)
    user_prompt = (
        f"{image_tags} Process all document pages and extract "
        f"the following information in JSON format: {', '.join(keys)}"
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": json.dumps(
                    OrderedDict(sorted(target_data.items()))
                ),
            },
        ],
        "images": images,
    }


def process_split(
    df: pd.DataFrame,
    split_name: str,
    output_dir: str,
    keys: List[str],
    augment: bool = False,
    aug_copies: int = 2,
) -> int:
    """Process a data split and save to SWIFT format."""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    swift_data = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"  Processing {split_name}"
    ):
        # Save image
        filename = Path(row["filename"]).stem
        img_path = os.path.join(images_dir, f"{filename}.png")

        if not os.path.exists(img_path):
            img = Image.open(io.BytesIO(row["doc_bytes"]))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path)

        # Parse and transform annotations
        target_data = transform_annotations(
            json.loads(row["target_data"]), defaults=keys
        )

        # Create SWIFT example
        rel_path = f"images/{filename}.png"
        example = create_swift_example([rel_path], target_data, keys)
        swift_data.append(example)

        # Data augmentation (training only)
        if augment and split_name == "train":
            img = Image.open(img_path)
            for aug_idx in range(aug_copies):
                aug_img = augment_image(img)
                aug_filename = f"{filename}_aug{aug_idx}.png"
                aug_path = os.path.join(images_dir, aug_filename)
                aug_img.save(aug_path)

                aug_example = create_swift_example(
                    [f"images/{aug_filename}"], target_data, keys
                )
                swift_data.append(aug_example)

    # Save to JSON
    output_path = os.path.join(
        output_dir, f"conversations_{split_name}_swift_format.json"
    )
    with open(output_path, "w") as f:
        json.dump(swift_data, f, indent=2)

    return len(swift_data)


def collect_all_keys(dataframes: Dict[str, pd.DataFrame]) -> List[str]:
    """Collect all unique keys from all splits."""
    all_keys = set()
    for df in dataframes.values():
        for target in df["target_data"]:
            all_keys.update(json.loads(target).keys())
    return sorted(all_keys)


def generate_schema(output_dir: str) -> None:
    """Generate JSON schema from all data."""
    builder = SchemaBuilder()

    for split in ["train", "dev", "test"]:
        path = os.path.join(
            output_dir, f"conversations_{split}_swift_format.json"
        )
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                for item in data:
                    content = item["messages"][2]["content"]
                    builder.add_object(json.loads(content))

    schema = builder.to_schema()
    schema_path = os.path.join(output_dir, "groundtruth_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved: {schema_path}")


# ============================================
# Main
# ============================================


def main():
    parser = argparse.ArgumentParser(
        description="Create SWIFT dataset from FATURA"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data-raw",
        help="Input directory containing extracted FATURA dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for SWIFT format dataset",
    )
    parser.add_argument(
        "--strategy",
        type=int,
        default=2,
        choices=[1, 2],
        help="Split strategy: 1=random, 2=template-based (default: 2)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation for training set",
    )
    parser.add_argument(
        "--aug-copies",
        type=int,
        default=2,
        help="Number of augmented copies per sample (default: 2)",
    )
    args = parser.parse_args()

    # Paths
    dataset_dir = os.path.join(args.input_dir, "invoices_dataset_final")
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(
        dataset_dir, "Annotations", "Original_Format"
    )

    print("=" * 60)
    print("FATURA to SWIFT Dataset Converter")
    print("=" * 60)
    print(f"Input:  {dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Strategy: {args.strategy}")
    if args.augment:
        print(f"Augmentation: ENABLED ({args.aug_copies} copies)")
    print()

    # Check input exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset not found at {dataset_dir}")
        print("Please run 01_download_and_extract.sh first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Template indices for Strategy 2
    train_templates = {
        3,
        11,
        30,
        24,
        40,
        48,
        41,
        22,
        27,
        19,
        45,
        1,
        29,
        44,
        9,
        47,
        36,
        23,
        18,
        42,
        15,
        14,
        28,
        43,
        33,
        6,
        38,
        26,
        13,
        34,
        17,
        37,
        5,
        8,
        21,
        35,
        16,
        20,
        31,
        46,
    }
    dev_templates = {50, 7, 32, 39, 2, 12, 4, 49, 10, 25}

    # Load all splits
    splits = {
        "train": "strat1_train.csv",
        "dev": "strat1_dev.csv",
        "test": "strat1_test.csv",
    }
    dataframes = {}

    print("Loading data...")
    for split_name, csv_file in splits.items():
        csv_path = os.path.join(dataset_dir, csv_file)

        # Apply template filter for Strategy 2
        templates = None
        if args.strategy == 2:
            templates = (
                train_templates if split_name == "train" else dev_templates
            )

        dataframes[split_name] = load_fatura_split(
            csv_path, images_dir, annotations_dir, templates
        )
        print(f"  {split_name}: {len(dataframes[split_name])} samples")

    # Collect all keys
    keys = collect_all_keys(dataframes)
    print(f"\nFound {len(keys)} unique keys: {keys[:5]}...")

    # Process each split
    print("\nCreating SWIFT format dataset...")
    sample_counts = {}
    for split_name, df in dataframes.items():
        count = process_split(
            df,
            split_name,
            args.output_dir,
            keys,
            augment=args.augment,
            aug_copies=args.aug_copies,
        )
        sample_counts[split_name] = count

    # Generate schema
    print("\nGenerating schema...")
    generate_schema(args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nSample counts:")
    for split, count in sample_counts.items():
        print(f"  {split}: {count}")

    if args.augment:
        print(
            f"\nAugmentation: {sample_counts['train']} training samples "
            f"({args.aug_copies + 1}x original)"
        )

    print("\nNext step: python 03_train.py")


if __name__ == "__main__":
    main()
