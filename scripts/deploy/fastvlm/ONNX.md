# ColFastVLM ONNX Deployment Guide

This document walks through exporting the merged ColFastVLM checkpoint to ONNX, validating the bundle, and loading the graphs from a WebGPU frontend such as `transformers.js`.

## 1. Environment

Install the `deploy` extras to pull in ONNX, ONNX Runtime, and the conversion helpers:

```bash
uv pip install -e ".[deploy]"
```

> **Note:** The project pins `numpy<2` because current ONNX Runtime wheels are not yet compatible with NumPy 2.x.

## 2. Export the ONNX graphs

Run `export_simple_pytorch_onnx.py` against the merged model directory. The command below creates fp32 and fp16 variants for each component and copies the tokenizer / processor assets into the `onnx/` subdir.

```bash
python scripts/deploy/fastvlm/export_simple_pytorch_onnx.py \
  /Users/nate/models/fastvlm/colfastvlm-0.5b-v0.1m \
  --components vision projector text-backbone text-projection text-encoder \
  --variants fp32 fp16 \
  --sequence-length 128 \
  --verbose
```

Artifacts will be written to `</path/to/model>/onnx/`, with the fp16 graphs named `*_fp16.onnx`.

### Naming convention

- `vision_encoder[_fp16].onnx`: MobileCLIP image tower
- `mm_projector[_fp16].onnx`: MLP projector that maps vision features to the language space
- `text_backbone[_fp16].onnx`: Qwen2 decoder stack (text-only path)
- `text_projection[_fp16].onnx`: ColBERT projection head (896 → 128)
- `text_encoder[_fp16].onnx`: Combined text backbone + projection, emitting normalized embeddings

All fp16 graphs have been rewritten so `Cast` nodes stay in half precision, avoiding ONNX Runtime type mismatches.

## 3. Validate the bundle

Use the included smoke test to ensure the fp16 graphs load and run with dummy inputs:

```bash
python scripts/deploy/fastvlm/validate_webgpu_bundle.py \
  /Users/nate/models/fastvlm/colfastvlm-0.5b-v0.1m/onnx
```

The script checks the presence of tokenizer assets, validates input/output dtypes, and feeds random tensors through each graph via ONNX Runtime.

## 4. Integrate with `transformers.js`

When loading the bundle from a WebGPU app (e.g., the `fastvlm-webgpu` example):

1. Upload the `onnx/` directory contents to your model hosting location (Hugging Face or static storage).
2. Point the frontend to the model repo and specify fp16 dtypes for the components you load. Example snippet:

```ts
const model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
  device: "webgpu",
  dtype: {
    vision_encoder: "fp16",
    mm_projector: "fp16",
    text_encoder: "fp16",
  },
});
```

3. If you prefer to run the backbone + projection separately (for pure retrieval embeddings), load `text_backbone_fp16.onnx` and `text_projection_fp16.onnx` individually and wire them through your WebGPU execution layer.

## 5. Optional variants

Currently the tool produces fp32 and fp16 exports. If you need int8/int4 quantisation for smaller downloads, extend `_generate_variants` in `export_simple_pytorch_onnx.py` with the desired quantisation pass (e.g., weight-only Q4). Keep the validation script as the gate to ensure the new graphs still execute correctly.

## 6. Troubleshooting

- **ONNX Runtime import errors:** Confirm the virtualenv is using NumPy < 2 and reinstall with `uv pip install -e ".[deploy]"`.
- **Validation failure:** Re-run the exporter; check that the merged model directory contains `custom_text_proj.pt` and the Hugging Face tokenizer files.
- **WebGPU precision mismatch:** Ensure the `dtype` map in the frontend matches the suffix of the ONNX files you upload (e.g., use `vision_encoder_fp16.onnx` when you set `vision_encoder: "fp16"`).

With these steps you should be able to export, validate, and deploy the ColFastVLM retriever in a browser-based WebGPU environment.
