#!/usr/bin/env python3
"""Export a merged ColFastVLM bundle to ONNX artifacts."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import types
from importlib import machinery
from pathlib import Path
from typing import Iterable, Mapping

import onnx
import torch
import torch.nn.functional as nnf
from onnxconverter_common import float16


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

from colpali_engine.models import ColFastVLM  # noqa: E402

torch.set_grad_enabled(False)


LOGGER = logging.getLogger("export_colfastvlm_onnx")


class TextBackboneForONNX(torch.nn.Module):
    """Thin wrapper around the LlavaQwen2 text backbone to stabilise ONNX export."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=False,
            use_cache=False,
        )
        hidden_states = outputs[0]
        return hidden_states


class TextEncoderForONNX(torch.nn.Module):
    """Full text encoder that mirrors ColFastVLM's text-only forward path."""

    def __init__(self, backbone: torch.nn.Module, projection: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=False,
            use_cache=False,
        )
        hidden_states = outputs[0]
        projected = self.projection(hidden_states)
        projected = nnf.normalize(projected, p=2.0, dim=-1)
        if attention_mask is not None:
            mask = attention_mask.to(projected.dtype).unsqueeze(-1)
            projected = projected * mask
        return projected


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _export_module(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: Iterable[str],
    output_names: Iterable[str],
    dynamic_axes: Mapping[str, Mapping[int, str]],
    opset_version: int,
) -> None:
    LOGGER.info("Exporting %s", output_path.name)
    torch.onnx.export(
        module,
        inputs,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes={k: dict(v) for k, v in dynamic_axes.items()},
    )
    LOGGER.info("Saved %s", output_path)
    return output_path


def _convert_to_fp16(source: Path, target: Path) -> None:
    LOGGER.info("Creating FP16 variant %s", target.name)
    model = onnx.load(str(source))
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=set(),
    )
    for node in converted.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 1:  # float32
                    attr.i = 10  # float16
    onnx.save(converted, str(target))


def _generate_variants(base_path: Path, variants: set[str]) -> None:
    if "fp16" in variants:
        target = base_path.with_name(f"{base_path.stem}_fp16.onnx")
        _convert_to_fp16(base_path, target)


def _prepare_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    return module.to(device=torch.device("cpu"), dtype=torch.float32)


def export_vision_encoder(wrapper: ColFastVLM, output_dir: Path, opset_version: int) -> tuple[Path, int, int]:
    vision = wrapper.model.get_vision_tower()
    if hasattr(vision, "load_model") and not getattr(vision, "is_loaded", False):
        vision.load_model()
    vision = _prepare_module(vision)

    vision_cfg = getattr(vision, "config", {})
    image_cfg = vision_cfg.get("image_cfg", {})
    image_size = int(image_cfg.get("image_size", 336))
    num_patches = int((image_size // int(image_cfg.get("patch_size", 32))) ** 2)

    dummy_images = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    output_path = output_dir / "vision_encoder.onnx"
    base_path = _export_module(
        vision,
        (dummy_images,),
        output_path,
        input_names=["pixel_values"],
        output_names=["vision_features"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "vision_features": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
    )

    embed_dim = int(image_cfg.get("embed_dim", vision_cfg.get("embed_dim", 3072)))
    return base_path, num_patches, embed_dim


def export_mm_projector(
    wrapper: ColFastVLM,
    output_dir: Path,
    opset_version: int,
    *,
    sequence_length: int,
    feature_dim: int,
) -> Path:
    mm_projector = wrapper.model.get_model().mm_projector
    mm_projector = _prepare_module(mm_projector)

    dummy_features = torch.zeros(1, sequence_length, feature_dim, dtype=torch.float32)
    output_path = output_dir / "mm_projector.onnx"
    base_path = _export_module(
        mm_projector,
        (dummy_features,),
        output_path,
        input_names=["vision_features"],
        output_names=["projected_features"],
        dynamic_axes={
            "vision_features": {0: "batch", 1: "sequence"},
            "projected_features": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
    )
    return base_path


def export_text_backbone(
    wrapper: ColFastVLM,
    output_dir: Path,
    opset_version: int,
    *,
    sequence_length: int,
) -> Path:
    backbone = wrapper.model.model
    backbone = _prepare_module(backbone)
    onnx_wrapper = TextBackboneForONNX(backbone)

    vocab_size = int(wrapper.model.config.vocab_size)
    dummy_input_ids = torch.randint(0, vocab_size, (1, sequence_length), dtype=torch.long)
    dummy_attention = torch.ones(1, sequence_length, dtype=torch.long)
    output_path = output_dir / "text_backbone.onnx"
    base_path = _export_module(
        onnx_wrapper,
        (dummy_input_ids, dummy_attention),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "hidden_states": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
    )
    return base_path


def export_text_projection(
    wrapper: ColFastVLM,
    output_dir: Path,
    opset_version: int,
    *,
    sequence_length: int,
) -> Path:
    projection = _prepare_module(wrapper.custom_text_proj)
    hidden_dim = projection.in_features
    dummy_hidden = torch.zeros(1, sequence_length, hidden_dim, dtype=torch.float32)
    output_path = output_dir / "text_projection.onnx"
    base_path = _export_module(
        projection,
        (dummy_hidden,),
        output_path,
        input_names=["hidden_states"],
        output_names=["embeddings"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
    )
    return base_path


def export_text_encoder(
    wrapper: ColFastVLM,
    output_dir: Path,
    opset_version: int,
    *,
    sequence_length: int,
) -> Path:
    backbone = wrapper.model.model
    backbone = _prepare_module(backbone)
    projection = _prepare_module(wrapper.custom_text_proj)
    encoder = TextEncoderForONNX(backbone, projection)

    vocab_size = int(wrapper.model.config.vocab_size)
    dummy_input_ids = torch.randint(0, vocab_size, (1, sequence_length), dtype=torch.long)
    dummy_attention = torch.ones(1, sequence_length, dtype=torch.long)
    output_path = output_dir / "text_encoder.onnx"
    base_path = _export_module(
        encoder,
        (dummy_input_ids, dummy_attention),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["colbert_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "colbert_embeddings": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
    )
    return base_path


DEFAULT_ASSET_NAMES = (
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


def _copy_assets(source: Path, destination: Path, assets: Iterable[str]) -> None:
    for name in assets:
        src_path = source / name
        if not src_path.exists():
            LOGGER.debug("Skipping missing asset %s", src_path.name)
            continue
        dst_path = destination / name
        if src_path.is_file():
            LOGGER.info("Copying %s", src_path.name)
            shutil.copy2(src_path, dst_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a merged ColFastVLM bundle to ONNX")
    parser.add_argument("model_path", type=Path, help="Path to the merged model directory")
    parser.add_argument("--output", type=Path, help="Output directory (defaults to <model_path>/onnx)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Dummy sequence length used for text exports (controls the initial graph shape)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["vision", "projector", "text-backbone", "text-projection", "text-encoder", "all"],
        default=["all"],
        help="Subset of components to export",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["fp32", "fp16"],
        default=["fp32", "fp16"],
        help="Precision variants to generate for each exported graph",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--copy-assets",
        nargs="*",
        default=list(DEFAULT_ASSET_NAMES),
        help="Model asset files to copy into the ONNX directory (pass an empty list to skip)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.verbose)

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        LOGGER.error("Model path %s does not exist", model_path)
        return 1

    output_dir = (args.output or (model_path / "onnx")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading ColFastVLM from %s", model_path)
    wrapper = ColFastVLM.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
    )
    LOGGER.info("Model loaded")

    selected = set(args.components)
    if "all" in selected:
        selected = {"vision", "projector", "text-backbone", "text-projection", "text-encoder"}

    seq_len = args.sequence_length
    image_sequence = None
    feature_dim = None
    variant_set = set(args.variants)
    keep_fp32 = "fp32" in variant_set
    conversion_variants = variant_set - {"fp32"}

    if "vision" in selected:
        vision_path, image_sequence, feature_dim = export_vision_encoder(wrapper, output_dir, args.opset)
        if conversion_variants:
            _generate_variants(vision_path, conversion_variants)
        if not keep_fp32:
            vision_path.unlink()
    if "projector" in selected:
        if image_sequence is None or feature_dim is None:
            vision_path, image_sequence, feature_dim = export_vision_encoder(wrapper, output_dir, args.opset)
            if conversion_variants:
                _generate_variants(vision_path, conversion_variants)
            if not keep_fp32:
                vision_path.unlink()
        projector_path = export_mm_projector(
            wrapper,
            output_dir,
            args.opset,
            sequence_length=image_sequence,
            feature_dim=feature_dim,
        )
        if conversion_variants:
            _generate_variants(projector_path, conversion_variants)
        if not keep_fp32:
            projector_path.unlink()
    if "text-backbone" in selected:
        backbone_path = export_text_backbone(wrapper, output_dir, args.opset, sequence_length=seq_len)
        if conversion_variants:
            _generate_variants(backbone_path, conversion_variants)
        if not keep_fp32:
            backbone_path.unlink()
    if "text-projection" in selected:
        projection_path = export_text_projection(wrapper, output_dir, args.opset, sequence_length=seq_len)
        if conversion_variants:
            _generate_variants(projection_path, conversion_variants)
        if not keep_fp32:
            projection_path.unlink()
    if "text-encoder" in selected:
        encoder_path = export_text_encoder(wrapper, output_dir, args.opset, sequence_length=seq_len)
        if conversion_variants:
            _generate_variants(encoder_path, conversion_variants)
        if not keep_fp32:
            encoder_path.unlink()

    if args.copy_assets:
        _copy_assets(model_path, output_dir, args.copy_assets)

    LOGGER.info("Export complete. Files written to %s", output_dir)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_DISABLED", "true")
    raise SystemExit(main())
