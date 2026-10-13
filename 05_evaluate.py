#!/usr/bin/env python3
"""
Model Evaluation Script for Document-to-JSON Fine-tuned Models

This script evaluates inference results against ground truth.

Features:
- Multi-metric evaluation: Exact Match, CER, ROUGE, BLEU
- Per-entity analysis
- Visualization using matplotlib

Usage:
    python 05_evaluate.py --results ./output/inference-xxx/results.jsonl
    python 05_evaluate.py --results ./output/results.jsonl --save-plots
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

warnings.filterwarnings("ignore")

# Try to import optional dependencies
try:
    import regex

    regex.DEFAULT_VERSION = regex.VERSION1
    JSON_PATTERN = regex.compile(r'\{(?:[^{}"]|"(?:\\.|[^"\\])*"|(?R))*\}')
except ImportError:
    import re

    JSON_PATTERN = re.compile(r"\{[^{}]*\}")

try:
    from Levenshtein import distance as levenshtein_distance

    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("Warning: Levenshtein not installed.")

try:
    import evaluate

    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False


# ============================================
# Utility Functions
# ============================================


def find_and_parse_json(text: str) -> Dict:
    """Extract and parse JSON from text."""
    if not text or not isinstance(text, str):
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = JSON_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def parse_labels(labels_str: Any) -> Dict:
    """Parse labels string to dict."""
    if isinstance(labels_str, dict):
        return labels_str
    if isinstance(labels_str, str):
        try:
            result = json.loads(labels_str)
            if isinstance(result, str):
                result = json.loads(result)
            return result
        except (TypeError, ValueError):
            return {}
    return {}


# ============================================
# Metrics
# ============================================


def calculate_exact_match(pred: str, label: str) -> float:
    """Calculate exact match (1.0 if match, 0.0 otherwise)."""
    pred_str = str(pred).strip() if pred else ""
    label_str = str(label).strip() if label else ""
    return 1.0 if pred_str == label_str else 0.0


def calculate_cer(pred: str, label: str) -> float:
    """Calculate Character Error Rate."""
    if not HAS_LEVENSHTEIN:
        return -1.0
    pred_str = str(pred) if pred else ""
    label_str = str(label) if label else ""
    if not label_str:
        return 0.0 if not pred_str else 1.0
    return levenshtein_distance(pred_str, label_str) / len(label_str)


def evaluate_sample(response: Dict, labels: Dict, entities: List[str]) -> Dict:
    """Evaluate a single sample across all entities."""
    results = {}

    for entity in entities:
        pred_val = response.get(entity, None)
        label_val = labels.get(entity, None)

        pred_str = str(pred_val) if pred_val and pred_val != "None" else ""
        label_str = str(label_val) if label_val and label_val != "None" else ""

        if not label_str:
            results[entity] = {
                "exact_match": None,
                "cer": None,
                "pred": pred_str,
                "label": label_str,
            }
            continue

        results[entity] = {
            "exact_match": calculate_exact_match(pred_str, label_str),
            "cer": calculate_cer(pred_str, label_str),
            "pred": pred_str,
            "label": label_str,
        }

    return results


# ============================================
# Evaluation
# ============================================


def evaluate_all(df: pd.DataFrame, model_name: str = "fine-tuned") -> Dict:
    """Evaluate all samples and aggregate results."""

    print("Parsing responses and labels...")
    df["response_parsed"] = df["response"].apply(find_and_parse_json)
    df["labels_parsed"] = df["labels"].apply(parse_labels)

    # Get all entities
    all_entities = set()
    for labels in df["labels_parsed"]:
        if isinstance(labels, dict):
            all_entities.update(labels.keys())
    entities = sorted(list(all_entities))
    print(f"Found {len(entities)} entities: {entities}")

    # Evaluate each sample
    print("Evaluating samples...")
    all_results = []
    for idx, row in df.iterrows():
        sample_results = evaluate_sample(
            row["response_parsed"], row["labels_parsed"], entities
        )
        for entity, metrics in sample_results.items():
            metrics["sample_id"] = idx
            metrics["entity"] = entity
            metrics["model"] = model_name
            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    # Aggregate by entity
    print("\nPer-Entity Results:")
    print("-" * 80)

    entity_summary = []
    for entity in entities:
        entity_data = results_df[results_df["entity"] == entity]
        valid_data = entity_data[entity_data["exact_match"].notna()]

        if len(valid_data) == 0:
            print(f"  {entity}: No valid ground truth")
            continue

        exact_match = valid_data["exact_match"].mean()
        cer = valid_data["cer"].mean() if HAS_LEVENSHTEIN else None
        n_exact = int(valid_data["exact_match"].sum())
        n_total = len(valid_data)

        entity_summary.append(
            {
                "entity": entity,
                "exact_match": exact_match,
                "cer": cer,
                "n_exact": n_exact,
                "n_total": n_total,
            }
        )

        cer_str = f", CER: {cer:.4f}" if cer is not None and cer >= 0 else ""
        print(
            f"  {entity}: Exact Match: {exact_match:.2%} ({n_exact}/{n_total}){
                cer_str
            }"
        )

    entity_df = pd.DataFrame(entity_summary)

    # Overall metrics
    valid_results = results_df[results_df["exact_match"].notna()]
    overall_exact_match = valid_results["exact_match"].mean()
    overall_cer = valid_results["cer"].mean() if HAS_LEVENSHTEIN else None

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"  Model: {model_name}")
    print(f"  Total Samples: {len(df)}")
    print(f"  Total Entities: {len(entities)}")
    print(f"  Exact Match (avg): {overall_exact_match:.2%}")
    if overall_cer is not None and overall_cer >= 0:
        print(f"  CER (avg): {overall_cer:.4f}")

    # ROUGE and BLEU
    if HAS_EVALUATE:
        print("\nCalculating ROUGE and BLEU...")
        try:
            rouge = evaluate.load("rouge")
            bleu = evaluate.load("bleu")

            predictions = valid_results["pred"].tolist()
            references = valid_results["label"].tolist()

            valid_pairs = [
                (p, r) for p, r in zip(predictions, references) if r
            ]
            if valid_pairs:
                preds, refs = zip(*valid_pairs)

                rouge_scores = rouge.compute(
                    predictions=list(preds), references=list(refs)
                )
                print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
                print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
                print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")

                try:
                    bleu_scores = bleu.compute(
                        predictions=list(preds), references=[[r] for r in refs]
                    )
                    print(f"  BLEU: {bleu_scores['bleu']:.4f}")
                except Exception:
                    pass
        except Exception as e:
            print(f"  (ROUGE/BLEU failed: {e})")

    return {
        "results_df": results_df,
        "entity_df": entity_df,
        "overall_exact_match": overall_exact_match,
        "overall_cer": overall_cer,
        "model_name": model_name,
        "n_samples": len(df),
        "n_entities": len(entities),
    }


# ============================================
# Save Results
# ============================================


def save_results(eval_results: Dict, output_dir: str):
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_path / "evaluation_details.csv"
    eval_results["results_df"].to_csv(results_file, index=False)
    print(f"\nSaved detailed results to: {results_file}")

    # Save entity summary
    entity_file = output_path / "evaluation_by_entity.csv"
    eval_results["entity_df"].to_csv(entity_file, index=False)
    print(f"Saved entity summary to: {entity_file}")

    # Save overall summary
    summary = {
        "model": eval_results["model_name"],
        "n_samples": eval_results["n_samples"],
        "n_entities": eval_results["n_entities"],
        "overall_exact_match": eval_results["overall_exact_match"],
        "overall_cer": eval_results["overall_cer"],
        "timestamp": datetime.now().isoformat(),
    }
    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")


# ============================================
# Visualization (Matplotlib)
# ============================================


def create_visualizations(eval_results: Dict, output_dir: str):
    """Create visualization plots using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualizations.")
        return

    output_path = Path(output_dir)
    entity_df = eval_results["entity_df"].copy()

    if entity_df.empty:
        print("No data for visualization")
        return

    # Sort by exact match
    entity_df = entity_df.sort_values("exact_match", ascending=True)

    # 1. Horizontal bar chart - Exact Match by Entity
    fig, ax = plt.subplots(figsize=(10, max(6, len(entity_df) * 0.4)))

    colors = plt.cm.Blues(entity_df["exact_match"] * 0.7 + 0.3)
    bars = ax.barh(
        entity_df["entity"],
        entity_df["exact_match"],
        color=colors,
        edgecolor="navy",
        linewidth=0.5,
    )

    ax.set_xlabel("Exact Match Rate", fontsize=12)
    ax.set_ylabel("Entity", fontsize=12)
    ax.set_title(
        f"Exact Match by Entity - {eval_results['model_name']}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.05)

    # Add percentage labels
    for bar, val in zip(bars, entity_df["exact_match"]):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=9,
        )

    # Add grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Overall exact match line
    overall_em = eval_results["overall_exact_match"]
    ax.axvline(
        x=overall_em,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall_em:.1%}",
    )
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_file = output_path / "exact_match_by_entity.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {plot_file}")

    # 2. Pie chart - Overall Performance
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate overall correct/incorrect
    results_df = eval_results["results_df"]
    valid_results = results_df[results_df["exact_match"].notna()]
    n_correct = int(valid_results["exact_match"].sum())
    n_incorrect = len(valid_results) - n_correct

    sizes = [n_correct, n_incorrect]
    labels = [f"Correct\n({n_correct})", f"Incorrect\n({n_incorrect})"]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0.05, 0)

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title(
        f"Overall Extraction Accuracy - {eval_results['model_name']}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plot_file = output_path / "overall_accuracy_pie.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {plot_file}")

    # 3. CER distribution (if available)
    if HAS_LEVENSHTEIN and "cer" in entity_df.columns:
        valid_cer = entity_df[
            entity_df["cer"].notna() & (entity_df["cer"] >= 0)
        ]
        if not valid_cer.empty:
            fig, ax = plt.subplots(figsize=(10, max(6, len(valid_cer) * 0.4)))

            colors = plt.cm.Reds(valid_cer["cer"] * 0.7 + 0.1)
            bars = ax.barh(
                valid_cer["entity"],
                valid_cer["cer"],
                color=colors,
                edgecolor="darkred",
                linewidth=0.5,
            )

            ax.set_xlabel("Character Error Rate (CER)", fontsize=12)
            ax.set_ylabel("Entity", fontsize=12)
            ax.set_title(
                f"CER by Entity - {eval_results['model_name']}",
                fontsize=14,
                fontweight="bold",
            )

            for bar, val in zip(bars, valid_cer["cer"]):
                ax.text(
                    val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=9,
                )

            ax.xaxis.grid(True, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)

            plt.tight_layout()
            plot_file = output_path / "cer_by_entity.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to: {plot_file}")


# ============================================
# Main
# ============================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model inference results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to inference results (JSONL file)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fine-tuned",
        help="Name for the model being evaluated",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as results)",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save visualization plots"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = str(Path(args.results).parent)

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print(f"Model name: {args.model_name}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    # Load results
    print(f"Loading results from: {args.results}")
    df = pd.read_json(args.results, lines=True)
    print(f"Loaded {len(df)} samples")

    # Evaluate
    eval_results = evaluate_all(df, args.model_name)

    # Save results
    save_results(eval_results, args.output_dir)

    # Create visualizations
    if args.save_plots:
        create_visualizations(eval_results, args.output_dir)

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
