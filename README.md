# Vision LLM Fine-tuning Pipeline

Fine-tune Vision Language Models (VLMs) for document-to-JSON extraction using the MS-SWIFT framework.

## Overview

This pipeline fine-tunes VLMs to extract structured JSON data from document images (invoices, receipts, etc.). It trains on the FATURA invoice dataset and supports Qwen3.5 / Qwen3.6 (dense and MoE) as well as Qwen3-VL.

A fine-tuned Qwen3.5-9B adapter is published in the [releases](../../releases). On 26 phone photos of real receipts it returns every purchased item with the right price; see [Results](#results).

## Project Structure

```
├── 01_download_and_extract.sh    # Extract FATURA dataset
├── 02_create_dataset.py          # Create SWIFT-format dataset
├── 03_train.py                   # Fine-tuning script (LoRA / DoRA)
├── 04_inference.py               # Merge adapter + batch inference with vLLM
├── 05_evaluate.py                # Evaluation with metrics & plots
├── 06_api_wrapper.py             # Upload-an-image API in front of vLLM
├── 07_client_example.py          # Client example for API calls
├── merge_adapter.py              # Merge a LoRA / DoRA adapter into its base model
└── results/
    ├── RESULTS.md                # Full comparison
    └── *.tsv                     # Score tables
```

## Requirements

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.11 – 2.13, transformers 5.9 – 5.16, vLLM 0.21 – 0.28, MS-SWIFT 4.5.3 and peft 0.20. Qwen3.5 / Qwen3.6 need a recent transformers and vLLM; `flash-linear-attention` is the fast path for their linear attention.

## Supported Models

| Model | SWIFT type | Parameters |
|-------|------------|------------|
| Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B | qwen3_5 | dense |
| Qwen/Qwen3.6-27B | qwen3_5 | dense |
| Qwen/Qwen3.6-35B-A3B | qwen3_5_moe | 35B total / 3B active |
| Qwen/Qwen3-VL-2B/4B/8B-Instruct | qwen3_vl | dense |
| Qwen/Qwen3-VL-30B-A3B-Instruct | qwen3_vl_moe | 30B total / 3B active |

The SWIFT model type is read from the model's local `config.json`, so `--model-type` is rarely needed.

### Notes for Qwen3.5 / Qwen3.6

- **DoRA**: DoRA on the Gated DeltaNet `in_proj_qkv` projection produces NaN loss (every other linear layer is fine, and DoRA on Qwen3-VL is fine). `03_train.py` excludes that single projection when DoRA is used on these models; plain LoRA (`--no-dora`) still targets all linear layers.
- **Thinking mode**: training targets carry an empty `<think></think>` block (non-thinking mode). Inference and serving must pass `enable_thinking=False` to match; `04_inference.py` does, and `vllm serve` needs `--default-chat-template-kwargs '{"enable_thinking": false}'`.
- **Qwen3.6-35B-A3B memory**: the bf16 expert weights alone are about 64 GB and cannot be 4-bit quantized with bitsandbytes. It trains with model parallelism plus DDP: several processes, each spreading the model over 2 GPUs (`--nproc-per-node`).

## Use the Released Adapter

The release holds a DoRA adapter for `Qwen/Qwen3.5-9B` (about 40 MB). vLLM does not support DoRA, so merge it into the base model once, then serve the merged model.

```bash
# 1. Download and unpack the adapter from the releases page
tar -xzf Qwen3.5-9B-receipt-dora-checkpoint-900.tar.gz

# 2. Merge it into the base model (downloads the base model if needed)
python merge_adapter.py \
    --base-model Qwen/Qwen3.5-9B \
    --adapter ./Qwen3.5-9B-receipt-dora-checkpoint-900 \
    --output ./Qwen3.5-9B-receipt-merged

# 3. Serve the merged model (bf16, about 18 GB of GPU memory)
vllm serve ./Qwen3.5-9B-receipt-merged \
    --served-model-name document-extraction --port 8080 \
    --max-model-len 8192 \
    --default-chat-template-kwargs '{"enable_thinking": false}'

# 4. Start the upload API in front of it
python 06_api_wrapper.py --port 58080 --mode receipt \
    --vllm-url http://localhost:8080/v1/chat/completions

# 5. Send a receipt
curl -X POST http://localhost:58080/extract -F "image=@receipt.jpg"
```

The response has the keys `merchant_name`, `address`, `phone`, `date`, `time`, `order_number`, `items` (a list of `name`, `quantity`, `price`), `subtotal`, `tax`, `total` and `payment_method`.

## Quick Start

### 1. Download and Extract Dataset

1. Download FATURA2.zip from: https://zenodo.org/records/10371464/files/FATURA2.zip?download=1
2. Run extraction:

```bash
./01_download_and_extract.sh ./FATURA2.zip ./data-raw
```

### 2. Create Training Dataset

```bash
# Basic usage
python 02_create_dataset.py --input-dir ./data-raw --output-dir ./data

# With data augmentation (for training set)
python 02_create_dataset.py --input-dir ./data-raw --output-dir ./data --augment --aug-copies 2
```

Options:
- `--strategy 1`: Random split
- `--strategy 2`: Template-based split (default, better generalization)
- `--augment`: Enable data augmentation (brightness, contrast, rotation, blur, noise)
- `--aug-copies N`: Number of augmented copies per sample

### 3. Fine-tune Model

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/Qwen3.5-9B

python 03_train.py \
    --model-id ./models/Qwen3.5-9B \
    --data-dir ./data \
    --output-dir ./output \
    --num-epochs 4 \
    --batch-size 2
```

Key training parameters:
- `--num-epochs`: Training epochs (default: 4)
- `--batch-size`: Per-device batch size (default: 4, use 1–2 for larger models)
- `--gradient-accumulation-steps`: Use it to keep the global batch size when the GPU count changes
- `--learning-rate`: Learning rate (default: 1e-4)
- `--lora-rank` / `--lora-alpha`: LoRA rank and alpha (default: 8 / 32)
- `--use-dora / --no-dora`: Enable/disable DoRA (default: enabled)
- `--save-steps` / `--save-total-limit`: Checkpoint interval and how many to keep (default: 50 / 3)
- `--nproc-per-node`: Number of training processes (DDP) on multi-GPU machines
- `--deepspeed`: DeepSpeed preset or config path (e.g. `zero3`)

The published results use 4 epochs, a global batch of 16 (1500 steps on 6000 samples), DoRA rank 8 and learning rate 1e-4, with a checkpoint and a dev evaluation every 50 steps.

### 4. Run Inference

`04_inference.py` merges the LoRA/DoRA adapter into the base model (cached as `<checkpoint>-merged`) and runs batch inference with vLLM.

```bash
# With fine-tuned model (auto-finds latest checkpoint)
python 04_inference.py --model-dir ./output

# With specific checkpoint
python 04_inference.py --model-dir ./output/v1-xxx/checkpoint-900

# With JSON schema for constrained decoding
python 04_inference.py --model-dir ./output --json-schema ./data/groundtruth_schema.json

# With a model that is already merged, or a base model
python 04_inference.py --use-base-model --model-id ./Qwen3.5-9B-receipt-merged
```

### 5. Evaluate Results

```bash
# Basic evaluation
python 05_evaluate.py --results ./output/inference-xxx/results.jsonl

# With visualization plots
python 05_evaluate.py --results ./output/inference-xxx/results.jsonl --save-plots
```

Metrics:
- **Exact Match**: Percentage of perfectly extracted fields
- **CER**: Character Error Rate
- **ROUGE-1/2/L**: Text similarity scores
- **BLEU**: Machine translation metric

### 6. Deploy as API Server

Serve the merged model with vLLM, as in [Use the Released Adapter](#use-the-released-adapter). For your own checkpoint, the merged model is the `<checkpoint>-merged` folder that `04_inference.py` writes, or the output of `merge_adapter.py`.

**GPU Requirements (bf16):**

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| Qwen3.5-2B / 4B | ~6 / ~10GB | 1x any recent GPU |
| Qwen3.5-9B, Qwen3-VL-8B | ~20GB | 1x 24 GB or 32 GB GPU |
| Qwen3.6-35B-A3B | ~70GB | 2–4x 80 GB GPUs (`--tensor-parallel-size`) |

`--quantization fp8` lets vLLM quantize the weights on load (Qwen3.5-9B: 10.8 GiB instead of 17.7 GiB) at a small cost in accuracy; see [results/RESULTS.md](results/RESULTS.md).

**API wrapper (port 58080):**

```bash
python 06_api_wrapper.py --port 58080 --vllm-url http://localhost:8080/v1/chat/completions
```

The wrapper downscales images above `--max-pixels` (default 2408448, four times the training value) and decodes greedily. It has two prompts, chosen per request with `mode` or for all requests with `--mode`:

- `document` (default): the training-style "extract all information in JSON format" prompt, for invoices like FATURA.
- `receipt`: fixed keys plus the purchased items as a JSON list.

### 7. Call the API

**Using the client script:**

```bash
python 07_client_example.py --image ./invoice.png

# Save output to file
python 07_client_example.py --image ./invoice.png --output result.json

# Receipt prompt, another endpoint
python 07_client_example.py --image receipt.jpg --mode receipt \
    --url http://localhost:58080/extract
```

**Using curl:**

```bash
curl -X POST http://localhost:58080/extract \
    -F "image=@invoice.png"

# A receipt, with the purchased items as a JSON list
curl -X POST http://localhost:58080/extract \
    -F "image=@receipt.jpg" -F "mode=receipt"
```

**Using Python requests:**

```python
import requests

with open("receipt.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:58080/extract",
        files={"image": f},
        data={"mode": "receipt"},
    )

print(response.json())
```

## Data Format

### SWIFT Training Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a document processing expert."},
    {"role": "user", "content": "Document pages: <image> Extract: field1, field2..."},
    {"role": "assistant", "content": "{\"field1\": \"value1\", \"field2\": \"value2\"}"}
  ],
  "images": ["images/invoice_001.png"]
}
```

## Results

Full tables and notes: [results/RESULTS.md](results/RESULTS.md).

**FATURA test set** (250 invoices, 34 fields, test templates unseen in training). Each model was trained twice; cells are `checkpoint: exact match / CER` for the checkpoint with the lowest dev loss.

| Model | Run 1 | Run 2 | Train time (16x H100) |
|-------|-------|-------|-----------------------|
| Qwen3.5-4B | 750: 98.76% / 0.0070 | 700: **99.46% / 0.0001** | 22 min |
| Qwen3.5-9B | 450: 98.32% / 0.0140 | 1100: 98.98% / 0.0073 | 28 min |
| Qwen3.6-35B-A3B (MoE) | 900: 99.05% / 0.0061 | 1050: 98.95% / 0.0080 | 53 min |
| Qwen3.5-2B | 250: 98.79% / 0.0074 | 550: 97.84% / 0.0191 | 18 min |
| Qwen3-VL-8B (earlier run) | 250: 98.00% / 0.0072 | | |

Run-to-run variance is as large as the gaps between models.

**Real receipts** (26 phone photos, hand-typed ground truth, image size 4x the training value):

| Model | FATURA | Text transcribed | Items as JSON with the right price | Empty responses to the generic prompt |
|-------|-------:|-----------------:|-----------------------------------:|--------------------------------------:|
| Qwen3.5-9B, run 2, checkpoint-900 (released) | 98.79% | 97.4% | 41 / 41 | 0 / 52 |
| Qwen3-VL-8B, checkpoint-250 | 98.00% | 92.5% | 0 / 41 | 2 / 52 |
| Qwen3.5-4B, run 2, checkpoint-800 | 99.56% | 95.9% | 0 / 41 | 14 / 52 |

- The model with the best FATURA score is not the best on real receipts. Qwen3.5-9B is, and it is the only one that returns line items as JSON.
- Every FATURA label has `TABLE = [[{}]]`. Qwen3.5-4B and Qwen3-VL-8B learned to emit that placeholder for line items.
- Phone photos need more pixels than the training image size: transcription improves by 4 to 7 points at 4x.
- The released checkpoint was chosen on these test sets, so its scores are optimistic for new documents. It is not error-free: long codes on blurred or faded receipts are still misread.

## Reference

- **Dataset**: [FATURA - Multi-Layout Invoice Dataset](https://zenodo.org/records/10371464)
- **Framework**: [MS-SWIFT](https://github.com/modelscope/ms-swift)
- **Paper**: Limam, M., Dhiaf, M., & Kessentini, Y. (2023). FATURA: A Multi-Layout Invoice Image Dataset

## License

The code is released under the [MIT License](LICENSE). The FATURA dataset is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The released adapter is a derivative of `Qwen/Qwen3.5-9B` and is subject to that model's license.
