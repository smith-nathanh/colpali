#!/usr/bin/env python3
"""Smoke-test ColFastVLM ONNX bundle for WebGPU deployment.

This utility validates that the fp16 ONNX graphs exported for ColFastVLM
have the expected inputs/outputs and can execute a dummy forward pass using
ONNX Runtime (CPU). Although WebGPU execution cannot be exercised here, the
CPU check ensures graph structure and dtypes match what the browser runtimes
expect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import onnx
import onnxruntime as ort

EXPECTED_COMPONENTS: dict[str, dict[str, object]] = {
    "vision_encoder": {
        "file": "vision_encoder_fp16.onnx",
        "inputs": {"pixel_values": "tensor(float16)"},
        "outputs": {"vision_features": "tensor(float16)"},
        "dummy": lambda: {
            "pixel_values": np.random.randn(1, 3, 1024, 1024).astype(np.float16)
        },
    },
    "mm_projector": {
        "file": "mm_projector_fp16.onnx",
        "inputs": {"vision_features": "tensor(float16)"},
        "outputs": {"projected_features": "tensor(float16)"},
        "dummy": lambda: {
            "vision_features": np.random.randn(1, 256, 3072).astype(np.float16)
        },
    },
    "text_backbone": {
        "file": "text_backbone_fp16.onnx",
        "inputs": {
            "input_ids": "tensor(int64)",
            "attention_mask": "tensor(int64)",
        },
        "outputs": {"hidden_states": "tensor(float16)"},
        "dummy": lambda: {
            "input_ids": np.random.randint(0, 151936, size=(1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        },
    },
    "text_projection": {
        "file": "text_projection_fp16.onnx",
        "inputs": {"hidden_states": "tensor(float16)"},
        "outputs": {"embeddings": "tensor(float16)"},
        "dummy": lambda: {
            "hidden_states": np.random.randn(1, 128, 896).astype(np.float16)
        },
    },
    "text_encoder": {
        "file": "text_encoder_fp16.onnx",
        "inputs": {
            "input_ids": "tensor(int64)",
            "attention_mask": "tensor(int64)",
        },
        "outputs": {"colbert_embeddings": "tensor(float16)"},
        "dummy": lambda: {
            "input_ids": np.random.randint(0, 151936, size=(1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        },
    },
}

ASSET_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "custom_text_proj.pt",
)


def _assert_io(session: ort.InferenceSession, schema: Mapping[str, Mapping[str, str]]) -> None:
    inputs = {tensor.name: tensor.type for tensor in session.get_inputs()}
    outputs = {tensor.name: tensor.type for tensor in session.get_outputs()}
    expected_inputs = schema["inputs"]
    expected_outputs = schema["outputs"]
    if inputs != expected_inputs:
        raise AssertionError(f"Input mismatch: expected {expected_inputs}, got {inputs}")
    if outputs != expected_outputs:
        raise AssertionError(f"Output mismatch: expected {expected_outputs}, got {outputs}")


def _run_dummy_forward(session: ort.InferenceSession, builder) -> dict[str, np.ndarray]:
    feeds = builder()
    return session.run(None, feeds)


def _load_onnx_metadata(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    meta = {}
    if model.metadata_props:
        meta = {prop.key: prop.value for prop in model.metadata_props}
    return meta


def validate_bundle(bundle_dir: Path, dump_metadata: bool) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    ort.set_default_logger_severity(3)

    for component, info in EXPECTED_COMPONENTS.items():
        filename = info["file"]
        path = bundle_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX file for {component}: {path}")

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _assert_io(session, info)
        outputs = _run_dummy_forward(session, info["dummy"])
        summary[component] = {
            "file": str(path.relative_to(bundle_dir)),
            "output_shapes": [tuple(out.shape) for out in outputs],
            "output_dtypes": [str(out.dtype) for out in outputs],
            "metadata": _load_onnx_metadata(path) if dump_metadata else {},
        }

    missing_assets = [name for name in ASSET_FILES if not (bundle_dir / name).exists()]
    if missing_assets:
        raise FileNotFoundError(f"Missing auxiliary assets: {missing_assets}")

    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ColFastVLM ONNX bundle for WebGPU deployment")
    parser.add_argument("bundle", type=Path, help="Path to the ONNX bundle directory")
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Print model metadata (from ONNX Graph metadata props)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_path = args.bundle.expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    summary = validate_bundle(bundle_path, dump_metadata=args.metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
