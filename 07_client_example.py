#!/usr/bin/env python3
"""
Document Extraction Client

Requirements:
    pip install requests

Usage:
    python 07_client_example.py --image invoice.png
    python 07_client_example.py --image doc.png --output result.json
"""

import argparse
import json
import time
from pathlib import Path

import requests

# ============================================================
# Configuration - Update this value for endpoint
# ============================================================
DEFAULT_API_URL = "http://localhost:58080/extract"


def main():
    parser = argparse.ArgumentParser(description="Document Extraction Client")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the document image"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save output to file"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help="URL of the /extract endpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["document", "receipt"],
        help="Prompt to use (default: the server's default)",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    print(f"Processing: {args.image}")
    print("-" * 50)

    try:
        start_time = time.time()

        # Upload image file
        with open(args.image, "rb") as f:
            files = {"image": (Path(args.image).name, f)}
            data = {"mode": args.mode} if args.mode else None
            response = requests.post(args.url, files=files, data=data)

        response.raise_for_status()
        elapsed_time = time.time() - start_time

        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nOutput saved to: {args.output}")

        print(f"\nProcessing Time: {elapsed_time:.2f}s")

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API server")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
