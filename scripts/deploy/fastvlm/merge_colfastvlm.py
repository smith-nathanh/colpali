#!/usr/bin/env python3
"""Merge ColFastVLM LoRA adapters with the base FastVLM model.

This utility mirrors the manual workflow used during validation: it loads the
base ColFastVLM wrapper from this repository, attaches the PEFT checkpoint, and
exports a fully-merged Hugging Face model bundle (including the custom text
projection weights and tokenizer artefacts).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from importlib import machinery
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _stub_wandb() -> None:
    if "wandb" in sys.modules:
        return
    module = types.ModuleType("wandb")
    module.__spec__ = machinery.ModuleSpec("wandb", loader=None)
    module.init = lambda *_, **__: None
    module.log = lambda *_, **__: None
    module.finish = lambda *_, **__: None
    sys.modules["wandb"] = module


_stub_wandb()


from colpali_engine.models import ColFastVLM


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


logger = logging.getLogger(__name__)


def _read_adapter_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_path / "adapter_config.json"
    if not config_path.exists():
        logger.debug("No adapter_config.json found under %s", config_path)
        return {}
    logger.debug("Loading adapter_config.json from %s", config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_base_model_path(args_base: str | None, adapter_config: dict[str, Any]) -> str:
    if args_base:
        return args_base
    base = adapter_config.get("base_model_name_or_path")
    if base:
        logger.info("Found base model path in adapter config: %s", base)
        return base
    logger.warning("Falling back to default FastVLM base model path")
    return "apple/FastVLM-0.5B"


def _load_wrapper(base_model_path: str, dtype: torch.dtype | str, device_map: str | None) -> ColFastVLM:
    logger.info("Loading base ColFastVLM from %s", base_model_path)
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": base_model_path,
        "trust_remote_code": True,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    model = ColFastVLM.from_pretrained(**model_kwargs)
    logger.info("Base model loaded (dtype=%s, device=%s)", model.dtype, model.device)
    return model


def _merge_lora(wrapper: ColFastVLM, checkpoint_path: Path) -> ColFastVLM:
    logger.info("Attaching PEFT adapter from %s", checkpoint_path)
    peft_model = PeftModel.from_pretrained(wrapper, str(checkpoint_path), is_trainable=False)
    logger.info("Merging LoRA weights back into the base model")
    merged: ColFastVLM = peft_model.merge_and_unload()
    merged.to(torch.device("cpu"))
    return merged


def _save_tokenizer(tokenizer_path: str, output_path: Path) -> None:
    logger.info("Saving tokenizer assets from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)


def merge_colfastvlm(
    checkpoint: Path, output: Path, base_model: str | None, dtype: str, device_map: str | None
) -> None:
    logger.info("=" * 60)
    logger.info("Starting ColFastVLM merge")
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Output dir: %s", output)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    adapter_config = _read_adapter_config(checkpoint)
    base_model_path = _resolve_base_model_path(base_model, adapter_config)

    torch_dtype: torch.dtype | str = dtype
    if dtype != "auto":
        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported torch dtype '{dtype}'") from exc

    wrapper = _load_wrapper(base_model_path, torch_dtype, device_map)
    merged = _merge_lora(wrapper, checkpoint)

    logger.info("Ensuring output directory exists at %s", output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Serialising merged model")
    merged.save_pretrained(str(output))

    tokenizer_source = base_model_path if base_model_path else str(checkpoint)
    _save_tokenizer(tokenizer_source, output)

    saved_files = sorted(p.name for p in output.iterdir())
    logger.info("Saved files: %s", saved_files)
    logger.info("ColFastVLM merge completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a ColFastVLM LoRA checkpoint into a standalone model bundle")
    parser.add_argument("checkpoint", type=Path, help="Path to the PEFT checkpoint directory (e.g. checkpoint-xxxxx)")
    parser.add_argument("output", type=Path, help="Destination directory for the merged model")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model path (defaults to adapter_config.json or apple/FastVLM-0.5B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Torch dtype to load the base model with (e.g. float16, bfloat16, auto)",
    )
    parser.add_argument(
        "--device-map", type=str, default=None, help="Optional device map to use during loading (e.g. auto)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.verbose)
    merge_colfastvlm(
        checkpoint=args.checkpoint,
        output=args.output,
        base_model=args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
