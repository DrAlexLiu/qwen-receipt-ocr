# Results

All numbers are for fine-tuned models. Thinking mode is off in training and
inference.

## Training setup

| Setting | Value |
|---------|-------|
| Data | FATURA, 6000 training invoices, template-based split (test templates are not seen in training) |
| Method | DoRA, rank 8, alpha 32, dropout 0.05, all linear layers of the language model |
| Schedule | 4 epochs, global batch 16 (1500 steps), learning rate 1e-4 |
| Checkpoints | every 50 steps, each with a dev-set evaluation |
| Image size | `max_pixels` 602112 |
| Hardware | 16 H100 GPUs; 18 to 53 minutes per model |

Each model was trained twice with identical settings (run 1 and run 2).
Run 2 kept all 30 checkpoints.

## FATURA test set

250 invoices, 34 fields each, scored with `05_evaluate.py`. Cells are
`checkpoint: exact match / CER`.

| Model | Run 1: checkpoint with lowest dev loss | Run 2: checkpoint with lowest dev loss | Run 2: best of 10 on test | Train time |
|-------|----------------------------------------|----------------------------------------|---------------------------|------------|
| Qwen3.5-4B | 750: 98.76% / 0.0070 | 700: **99.46% / 0.0001** | 600: **99.62% / 0.0000** | 22 min |
| Qwen3.5-9B | 450: 98.32% / 0.0140 | 1100: 98.98% / 0.0073 | 1100: 98.98% / 0.0073 | 28 min |
| Qwen3.6-35B-A3B (MoE) | 900: 99.05% / 0.0061 | 1050: 98.95% / 0.0080 | 700: 99.11% / 0.0073 | 53 min |
| Qwen3.5-2B | 250: 98.79% / 0.0074 | 550: 97.84% / 0.0191 | 1100: 98.95% / 0.0080 | 18 min |

The unbiased number is the checkpoint picked by dev loss. "Best of 10 on
test" takes the 10 checkpoints with the lowest dev loss and picks by test
score, so it is optimistic. All 40 scores: `fatura_test_run2_top10.tsv`.

Earlier Qwen3-VL fine-tunes, scored with the same script:

| Model | Checkpoint | Exact match | CER |
|-------|-----------|-------------|-----|
| Qwen3-VL-8B | 250 | 98.00% | 0.0072 |
| Qwen3-VL-30B-A3B (MoE, 4-bit QLoRA) | 650 | 97.40% | 0.0120 |
| Qwen3-VL-4B | 550 | 97.27% | 0.0056 |

What to read from this:

- Run-to-run variance is as large as the gaps between models. Most of it
  comes from one field: on one unseen template the models read the `GSTIN`
  value correctly but sometimes file it under `GSTIN_SELLER`, which counts as
  a miss. Whether a run does this changes between retrains (`GSTIN` exact
  match for Qwen3.5-4B: 56% in run 1, 100% in run 2).
- Dev loss is a weak selector: the 10 best checkpoints differ by about 0.001
  in dev loss, and for Qwen3.5-2B the dev-best checkpoint ranks 9th of 10 on
  the test set.
- Every checkpoint of Qwen3.5-4B and Qwen3.5-9B from both runs (66 in all)
  was also scored on the test set: `fatura_test_all_checkpoints_4b_9b.tsv`.

## Real receipts

FATURA is synthetic invoices. To see what the models do on real documents,
67 models (the 66 checkpoints above and Qwen3-VL-8B checkpoint-250) were run
on 26 phone photos of receipts: cafes, restaurants, airport shops, a train
ticket, a transit top-up, a museum, a bookstore; English, and Chinese with
English. The photos and the model outputs are private and are not part of
this repository; the scores are in `receipts_26.tsv`.

Each photo was run with five prompts at two image sizes: the training value
(`1x`, `max_pixels` 602112) and four times the pixels (`4x`, 2408448). Ground
truth was typed in by hand: 611 printed strings, 41 purchased items on 22
receipts, 116 fields. Matching ignores whitespace and case.

| Column | Meaning |
|--------|---------|
| `ocr` | share of the 611 strings found in a plain transcription |
| `api_text` | the same for the "extract all information in JSON format" prompt |
| `api_valid` | how many of the 52 responses to that prompt parse as JSON |
| `api_empty` | how many of those 52 hold under 20% of the strings |
| `items_json`, `items_text` | share of the 41 items returned with the right price, as a JSON list and as text lines |
| `fields` | date, subtotal, tax, total and seller name under the FATURA keys, share of 116 |

| Model | Checkpoint | FATURA | `ocr` 1x / 4x | `api_text` 1x / 4x | `api_valid` | `api_empty` | `items_json` 1x / 4x | `items_text` 1x / 4x | `fields` 1x / 4x |
|-------|-----------|-------:|---|---|---:|---:|---|---|---|
| Qwen3.5-9B | **run 2, 900** | 98.79% | 0.926 / 0.974 | 0.881 / 0.939 | 49 | 0 | 0.951 / 1.000 | 0.878 / 0.976 | 0.905 / 0.974 |
| Qwen3.5-9B | run 2, 1100 (lowest dev loss) | 98.98% | 0.923 / 0.975 | 0.884 / 0.962 | 51 | 0 | 0.951 / 1.000 | 0.951 / 0.976 | 0.888 / 0.940 |
| Qwen3-VL-8B | 250 | 98.00% | 0.885 / 0.925 | 0.827 / 0.892 | 46 | 2 | 0.000 / 0.000 | 0.854 / 0.951 | 0.802 / 0.862 |
| Qwen3.5-4B | run 2, 800 | 99.56% | 0.900 / 0.959 | 0.802 / 0.548 | 35 | 14 | 0.024 / 0.000 | 0.902 / 0.951 | 0.802 / 0.905 |
| Qwen3.5-4B | run 2, 700 (lowest dev loss) | 99.46% | 0.903 / 0.961 | 0.417 / 0.255 | 42 | 33 | 0.000 / 0.000 | 0.951 / 0.927 | 0.767 / 0.879 |
| Qwen3.5-4B | run 1, 1450 | 98.83% | 0.900 / 0.957 | 0.894 / 0.825 | 51 | 3 | 0.366 / 0.317 | 0.878 / 0.951 | 0.879 / 0.879 |

- **Qwen3.5-9B is the best on real receipts**, on every column, although it
  is not the best on FATURA. From checkpoint-200 on, every Qwen3.5-9B
  checkpoint returns at least 90% of the items as a JSON list.
- **The FATURA score does not predict behaviour on receipts.** Qwen3.5-4B has
  the highest FATURA scores and the most failures here: asked to "extract all
  information", its best FATURA checkpoints often return only
  `{"document_pages": [[{}]]}`.
- **`[[{}]]` is learned from the labels.** Every FATURA label has
  `TABLE = [[{}]]`, and Qwen3.5-4B and Qwen3-VL-8B emit that placeholder for
  line items. Training data with real table rows would remove it.
- **More pixels help transcription**: `ocr` rises by 4 to 7 points at 4x for
  every model. Phone photos need more than the training image size.
- No model is error-free. At 4x the released checkpoint still misses long
  codes on blurred or faded receipts.

## Released weights and serving

The released adapter is **Qwen3.5-9B, run 2, checkpoint-900**. It was chosen
on the FATURA test set and on these receipts, so its scores above are
optimistic for new documents.

Served with vLLM on one RTX 5090 behind `06_api_wrapper.py`, each of the 26
photos uploaded once per mode:

| Weights | GPU memory for weights | `receipt` mode: items with the right price | `receipt` mode: valid JSON | `receipt` mode: seconds per receipt | `document` mode: strings found | `document` mode: valid JSON |
|---------|-----------------------:|---:|---:|---:|---:|---:|
| bf16 (recommended) | 17.7 GiB | 41 / 41 | 26 / 26 | 3.3 | 0.930 | 25 / 26 |
| `--quantization fp8` | 10.8 GiB | 41 / 41 | 26 / 26 | 2.7 | 0.885 | 23 / 26 |

- In `receipt` mode with bf16 weights, `total` is right on 25 of 26 receipts,
  `date` on 24 of 24, `subtotal` on 21 of 21, `tax` on 20 of 20 and
  `merchant_name` on 25 of 25. Fields are counted only where the receipt
  prints them.
- `document` mode keeps most of the text, but the JSON layout changes from
  receipt to receipt. Use `receipt` mode for receipts.
- FP8 loses one field in `receipt` mode and is clearly worse in `document`
  mode (2 empty responses of 26).
- Merge the adapter before serving (`merge_adapter.py`). vLLM does not
  support DoRA, and dropping the DoRA magnitude vectors to get a plain LoRA
  adapter does not give the trained model.

## Notes for Qwen3.5 / Qwen3.6

- DoRA on the Gated DeltaNet `in_proj_qkv` projection gives NaN loss.
  `03_train.py` excludes that one projection when DoRA is used on these
  models.
- Qwen3.6-35B-A3B: the bf16 expert weights (about 64 GB) do not fit one
  H100, cannot be 4-bit quantized with bitsandbytes, and DeepSpeed ZeRO-3 runs
  out of memory. It trains with two processes per 4-GPU node, each spreading
  the model over 2 GPUs (`--nproc-per-node`).
