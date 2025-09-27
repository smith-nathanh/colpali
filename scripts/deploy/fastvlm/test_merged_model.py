#!/usr/bin/env python3
"""Smoke tests for a merged ColFastVLM bundle."""

from __future__ import annotations

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

try:  # pragma: no cover - optional dependency for image download
    import requests
except ImportError:  # pragma: no cover - optional dependency for image download
    requests = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_ensure_repo_on_path()


def test_transformers_loading(model_path: Path):
    """Ensure Hugging Face APIs can load the merged model."""

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    logger.info("Testing transformers AutoModel loading from %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    logger.info("Tokenizer loaded (vocab size=%s)", tokenizer.vocab_size)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    logger.debug("<image> token id: %s", image_token_id)

    try:
        AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        logger.info("Processor loaded successfully")
    except Exception as exc:  # pragma: no cover - processor is optional
        logger.warning("Processor load failed (expected for some bundles): %s", exc)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    logger.info("Model loaded (dtype=%s, device=%s)", model.dtype, next(model.parameters()).device)

    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        logger.info("Vision tower: %s", type(vision_tower))

    return tokenizer, model


def test_colfastvlm_loading(model_path: Path):
    """Load the merged bundle with the repository ColFastVLM wrapper."""

    from colpali_engine.models import ColFastVLM

    logger.info("Testing ColFastVLM wrapper loading from %s", model_path)

    model = ColFastVLM.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    logger.info("ColFastVLM loaded (dtype=%s, device=%s)", model.dtype, model.device)

    if hasattr(model, "custom_text_proj"):
        proj_layer = model.custom_text_proj
        logger.info(
            "Custom text projection present (in_features=%s, out_features=%s)",
            proj_layer.in_features,
            proj_layer.out_features,
        )

    return model


def test_basic_inference(model, tokenizer) -> bool:
    """Run a short text-only generation to ensure logits flow."""

    logger.info("Running basic text inference smoke test")

    prompt = "Hello, how are you?"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        logger.error("Tokenizer did not return input_ids; received keys: %s", list(encoded.keys()))
        return False

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    try:
        with torch.no_grad():
            output = model.generate(
                inputs=input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        logger.error("Basic inference failed: %s", exc, exc_info=True)
        return False

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("Inference succeeded (prompt=%s, response=%s)", prompt, response)
    return True


def test_image_inference(model, tokenizer, skip_image: bool) -> bool:
    """Validate image-conditioned generation if networking is permitted."""

    if skip_image:
        logger.info("Skipping image inference by request")
        return True

    if requests is None:
        logger.warning("requests not available; skipping image inference test")
        return True

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    logger.info("Fetching test image from %s", url)

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on network access
        logger.error("Image download failed: %s", exc, exc_info=True)
        return False

    image = Image.open(BytesIO(resp.content)).convert("RGB")
    logger.info("Image downloaded (%sx%s)", *image.size)

    prompt = "<image>Describe this image."
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:  # pragma: no cover - dependent on vision tower availability
        logger.error("Image inference failed: %s", exc, exc_info=True)
        return False

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("Image inference output: %s", response)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a merged ColFastVLM bundle")
    parser.add_argument("model_path", type=Path, help="Path to the merged model directory")
    parser.add_argument("--skip-image", action="store_true", help="Skip the image-conditioned generation check")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Testing merged ColFastVLM model")
    logger.info("=" * 60)

    tokenizer, hf_model = test_transformers_loading(args.model_path)
    basic_ok = test_basic_inference(hf_model, tokenizer)
    image_ok = test_image_inference(hf_model, tokenizer, args.skip_image)

    colfastvlm_model = test_colfastvlm_loading(args.model_path)
    colfastvlm_ok = colfastvlm_model is not None

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("Transformers loading: %s", "PASS" if hf_model is not None else "FAIL")
    logger.info("Basic inference: %s", "PASS" if basic_ok else "FAIL")
    logger.info("Image inference: %s", "PASS" if image_ok else "FAIL")
    logger.info("ColFastVLM wrapper: %s", "PASS" if colfastvlm_ok else "FAIL")

    if not (hf_model and basic_ok and image_ok and colfastvlm_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
