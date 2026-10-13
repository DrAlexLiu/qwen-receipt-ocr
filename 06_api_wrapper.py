#!/usr/bin/env python3
"""
Document Extraction API Wrapper

This wrapper receives image files and forwards them to the vLLM server
with the proper message format.

Requirements:
    pip install flask requests

Usage:
    python 06_api_wrapper.py --port 58080 \
        --vllm-url http://localhost:8080/v1/chat/completions

Images larger than --max-pixels are downscaled before they are sent.
The default, 2408448, is 4x the training value (602112); phone photos
of receipts need the extra pixels.
"""

import argparse
import base64
import io
import json
import math
import re

import requests
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# Configuration
VLLM_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "document-extraction"
MAX_PIXELS = 2408448
SYSTEM_PROMPT = "You are a document processing expert and assistant."

# "document" is the prompt the model was trained with for whole
# documents. "receipt" asks for fixed keys and the purchased items as a
# JSON list.
PROMPTS = {
    "document": (
        "Process all document pages and extract all information "
        "in JSON format."
    ),
    "receipt": (
        "Extract this receipt as JSON with the keys: merchant_name, "
        "address, phone, date, time, order_number, items (a list of "
        "objects with name, quantity, price), subtotal, tax, total, "
        "payment_method. Use null for anything not on the receipt. "
        "Output only the JSON."
    ),
}
DEFAULT_MODE = "document"

# A Markdown code fence around the whole response, e.g. ```json ... ```
CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```$", re.DOTALL)


def encode_image_to_base64(image_data: bytes) -> str:
    """Convert image bytes to a base64 PNG of at most MAX_PIXELS."""
    with Image.open(io.BytesIO(image_data)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        if width * height > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (width * height))
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(size, Image.Resampling.BICUBIC)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


@app.route("/extract", methods=["POST"])
def extract_document():
    """
    Extract document data from uploaded image.

    Accepts:
        - multipart/form-data with 'image' file
        - application/json with 'image' as base64 string
        - optional 'mode', form field or JSON key: "document" or
          "receipt"

    Returns:
        JSON with extracted document data
    """
    try:
        # Get image from request
        if (
            request.content_type
            and "multipart/form-data" in request.content_type
        ):
            if "image" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            mode = request.form.get("mode", DEFAULT_MODE)
            image_file = request.files["image"]
            image_data = image_file.read()
            image_base64 = encode_image_to_base64(image_data)
        elif request.is_json:
            data = request.get_json()
            if "image" not in data:
                return jsonify({"error": "No image provided"}), 400
            mode = data.get("mode", DEFAULT_MODE)
            # Base64, with or without a data: URL prefix
            image_data = base64.b64decode(data["image"].split(",", 1)[-1])
            image_base64 = encode_image_to_base64(image_data)
        else:
            return jsonify({"error": "Invalid content type"}), 400

        if mode not in PROMPTS:
            return jsonify({"error": f"Unknown mode: {mode}"}), 400

        # Build payload for vLLM
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Document pages: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64},
                        },
                        {"type": "text", "text": PROMPTS[mode]},
                    ],
                },
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        # Forward to vLLM
        response = requests.post(
            VLLM_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        # Extract content
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Some models wrap the JSON in a Markdown code fence
        fenced = CODE_FENCE.match(content.strip())
        if fenced:
            content = fenced.group(1)

        try:
            parsed = json.loads(content)
            return jsonify(parsed)
        except json.JSONDecodeError:
            return jsonify({"raw_text": content})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to vLLM server"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


def main():
    global VLLM_URL, MODEL_NAME, MAX_PIXELS, DEFAULT_MODE

    parser = argparse.ArgumentParser(
        description="Document Extraction API Wrapper"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=58080,
        help="Port to run wrapper (default: 58080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8080/v1/chat/completions",
        help="vLLM API URL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="document-extraction",
        help="Model name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=sorted(PROMPTS),
        help="Prompt used when a request names none (default: document)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=MAX_PIXELS,
        help="Downscale larger images to this many pixels (default: 2408448)",
    )

    args = parser.parse_args()

    VLLM_URL = args.vllm_url
    MODEL_NAME = args.model_name
    MAX_PIXELS = args.max_pixels
    DEFAULT_MODE = args.mode

    print(f"Starting wrapper server on http://{args.host}:{args.port}")
    print(f"Forwarding to vLLM: {VLLM_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max pixels: {MAX_PIXELS}")
    print(f"Default mode: {DEFAULT_MODE}")
    print("-" * 50)
    print("Endpoints:")
    print("  POST /extract  - Upload image and get extracted data")
    print("  GET  /health   - Health check")

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
