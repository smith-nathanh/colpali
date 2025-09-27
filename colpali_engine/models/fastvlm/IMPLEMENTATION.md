# ColFastVLM Implementation Notes

This document records the current understanding of the ColFastVLM stack as used in the `colfastvlm-0.5b-v0.1m` release. The goal is to capture architectural details, training hooks, and the deployment/validation workflow so future changes can be audited quickly.

## Backbone Overview
- Base model: `apple/FastVLM-0.5B`, a LLaVA-Qwen2 derivative (`colpali_engine/models/fastvlm/llava_qwen.py`).
- Text hidden width: `896` (`colpali_engine/models/fastvlm/config.json:15`).
- Vision tower: MobileCLIP FastViT-HD, ingesting 1024×1024 RGB images with 64×64 patches -> 16×16 grid (`256` patches) of `3072`-D features (`colpali_engine/models/fastvlm/llava_qwen.py:1425`, `colpali_engine/models/fastvlm/llava_qwen.py:1591-1600`).
- Multimodal projector: `mlp2x_gelu` mapping `3072 → 896` before fusion into the language stack (`colpali_engine/models/fastvlm/config.json:22-26`, `colpali_engine/models/fastvlm/llava_qwen.py:1615-1628`).

## Embedding Pathways
### Document / Image Encodes
1. Images are normalized to the tower’s expected 1024² resolution; the vision encoder emits a `(batch, 256, 3072)` tensor (`llava_qwen.encode_images`).
2. The MLP projector reduces this to `(batch, 256, 896)` so the features align with the decoder hidden size (same references as above).
3. During retrieval forward passes, these vision tokens are concatenated to the decoder hidden states and optionally masked to zero out prompt text, leaving the image patches as the only non-zero embeddings (`colpali_engine/models/fastvlm/modeling_fastvlm.py:165-196`).
4. `custom_text_proj`, a learned linear layer, converts every token to unit-normalised `128`-D vectors (`colpali_engine/models/fastvlm/modeling_fastvlm.py:20-198`). With `mask_non_image_embeddings=True`, a single document image therefore yields `256` active vectors at 128 dims.

### Query / Text Encodes
- Text queries traverse the same decoder backbone and projection head without vision inputs. Token length is preserved, and each token is projected from `896` → `128` and L2-normalised (`colpali_engine/models/fastvlm/modeling_fastvlm.py:165-196`).
- The shared projection guarantees that text and image embeddings live in the same ColBERT space, enabling late-interaction scoring.

## LoRA Training Hooks
- LoRA adapters target the Qwen2 decoder blocks along with the multimodal projector; the `custom_text_proj` layer is always listed in `modules_to_save` so the trained projection weights survive merge/export (`scripts/configs/fastvlm/train_colfastvlm-500-base.yaml:66-79`).
- The repository wrapper surfaces `custom_text_proj` explicitly (see `ColFastVLM` in `modeling_fastvlm.py`) so PEFT checkpoints can update it alongside the base model.

## Merge and Export Pipeline
1. **Merge LoRA adapters** with `scripts/deploy/fastvlm/merge_colfastvlm.py`. This script:
   - Loads the base FastVLM checkpoint (pulling overrides from `adapter_config.json` when present).
   - Attaches and merges the LoRA weights back into the base modules (`PeftModel.merge_and_unload`).
   - Saves the merged model plus tokenizer assets and `custom_text_proj.pt` so downstream tooling can reload the projection layer.
2. **Export ONNX graphs** via `scripts/deploy/fastvlm/export_simple_pytorch_onnx.py`. Key component shapes:
   - `vision_encoder[_fp16].onnx`: `(batch, 256, 3072)` outputs.
   - `mm_projector[_fp16].onnx`: `(batch, 256, 896)` outputs.
   - `text_backbone[_fp16].onnx`: `(batch, sequence, 896)` outputs (sequence length is the dummy `--sequence-length`).
   - `text_projection[_fp16].onnx` and `text_encoder[_fp16].onnx`: `(batch, sequence, 128)` outputs.
3. Auxiliary assets (`tokenizer.json`, `custom_text_proj.pt`, etc.) are copied into the `onnx/` directory for WebGPU and transformers.js consumption.

## Validation (2024-09-27)
- Environment prepared with `uv pip install -e ".[deploy]" && uv pip install numpy onnx onnxruntime onnxconverter-common`.
- Bundle validated using:
  ```bash
  uv run python scripts/deploy/fastvlm/validate_webgpu_bundle.py \
    ~/models/fastvlm/colfastvlm-0.5b-v0.1m/onnx
  ```
- Reported shapes (fp16 graphs):
  - `vision_encoder_fp16.onnx`: `1 × 256 × 3072`
  - `mm_projector_fp16.onnx`: `1 × 256 × 896`
  - `text_backbone_fp16.onnx`: `1 × 128 × 896`
  - `text_projection_fp16.onnx`: `1 × 128 × 128`
  - `text_encoder_fp16.onnx`: `1 × 128 × 128`
- All auxiliary assets were present in `~/models/fastvlm/colfastvlm-0.5b-v0.1m/onnx`, confirming the bundle is ready for WebGPU / ONNX Runtime deployment.

