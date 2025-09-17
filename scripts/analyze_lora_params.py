import argparse
import importlib
from pathlib import Path
from typing import Optional

import torch
from torch import nn

try:
    from peft import get_peft_model
except ImportError as e:  # pragma: no cover
    raise SystemExit("peft package required. Install with `pip install peft`. ") from e

try:
    from transformers import AutoModel
except ImportError as e:  # pragma: no cover
    raise SystemExit("transformers package required. Install with `pip install transformers`. ") from e


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def has_peft_config(model: nn.Module) -> bool:
    """Check if model already has PEFT configuration applied."""
    return hasattr(model, "peft_config") and model.peft_config is not None


def print_trainable_params_summary(model: nn.Module) -> None:
    """Print summary of trainable parameters."""
    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    pct = 100.0 * trainable_params / total_params if total_params else 0.0

    print("\nTrainable Parameters Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Percentage trainable: {pct:.2f}%")


def analyze_lora_modules(model: nn.Module) -> None:
    """Analyze which modules have LoRA adapters applied and reverse engineer the regex."""
    if not has_peft_config(model):
        print("No PEFT configuration found in model.")
        return

    print("\nLoRA Module Analysis:")
    print("=" * 80)

    # Get all module names that have LoRA adapters
    lora_modules = []
    all_modules = []

    for name, module in model.named_modules():
        all_modules.append(name)
        # Check if this module has LoRA adapters
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_modules.append(name)

    print(f"Total modules in model: {len(all_modules)}")
    print(f"Modules with LoRA adapters: {len(lora_modules)}")
    print("\nModules with LoRA adapters:")
    for module_name in sorted(lora_modules):
        print(f"  - {module_name}")

    # Try to reverse engineer the regex pattern
    print("\nReverse Engineering LoRA Target Regex:")
    print("-" * 50)

    if lora_modules:
        # Analyze patterns in the LoRA module names
        text_model_modules = [m for m in lora_modules if "text_model" in m]
        custom_proj_modules = [m for m in lora_modules if "custom_text_proj" in m or "linear" in m]

        print(f"Text model modules with LoRA: {len(text_model_modules)}")
        print(f"Custom projection modules with LoRA: {len(custom_proj_modules)}")

        # Show some examples
        if text_model_modules:
            print("\nExample text model LoRA modules:")
            for module in text_model_modules[:5]:  # Show first 5
                print(f"  - {module}")
            if len(text_model_modules) > 5:
                print(f"  ... and {len(text_model_modules) - 5} more")

        if custom_proj_modules:
            print("\nCustom projection LoRA modules:")
            for module in custom_proj_modules:
                print(f"  - {module}")

        # Analyze what projection types are targeted
        proj_types = set()
        for module in lora_modules:
            if any(
                proj in module for proj in ["down_proj", "gate_proj", "up_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
            ):
                for proj in ["down_proj", "gate_proj", "up_proj", "k_proj", "q_proj", "v_proj", "o_proj"]:
                    if proj in module:
                        proj_types.add(proj)

        print(f"\nProjection types targeted: {sorted(proj_types)}")

        # Try to reconstruct the regex
        if text_model_modules and proj_types:
            proj_pattern = "|".join(sorted(proj_types))
            reconstructed_regex = f"(.*(model.text_model).*({proj_pattern}).*$"
            if custom_proj_modules:
                custom_terms = []
                for module in custom_proj_modules:
                    if "linear" in module:
                        custom_terms.append("linear")
                    if "custom_text_proj" in module:
                        custom_terms.append("custom_text_proj")
                if custom_terms:
                    custom_pattern = "|".join(set(custom_terms))
                    reconstructed_regex += f"|.*({custom_pattern}).*$"
            reconstructed_regex += ")"

            print("\nReconstructed regex pattern:")
            print(f"  {reconstructed_regex}")

            # Compare with config if available
            if hasattr(model, "peft_config") and model.peft_config:
                config_key = list(model.peft_config.keys())[0]
                peft_config = model.peft_config[config_key]
                if hasattr(peft_config, "target_modules"):
                    print("\nOriginal target_modules in config:")
                    print(f"  {peft_config.target_modules}")

                    # Check if they match
                    if str(peft_config.target_modules) == reconstructed_regex:
                        print("  ✅ Reconstructed regex matches original!")
                    else:
                        print("  ⚠️  Reconstructed regex differs from original")
    else:
        print("No LoRA modules found to analyze.")


def load_peft_config_from_yaml(config_file: str):
    """Load LoRA config from a training config YAML file."""
    import configue

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load the config using configue
    config = configue.load(config_path, sub_path="config")

    if not hasattr(config, "peft_config") or config.peft_config is None:
        # Check if train_only_linear is set
        if hasattr(config, "train_only_linear") and config.train_only_linear:
            raise ValueError(
                f"No peft_config found in {config_file}.\n"
                f"Note: This config uses 'train_only_linear: true' which trains only the linear projection layer "
                f"without LoRA/PEFT. This is a valid training configuration but doesn't use LoRA adapters."
            )
        else:
            raise ValueError(f"No peft_config found in {config_file}")

    return config.peft_config


def dynamic_import(dotted: str):
    mod_path, attr = dotted.rsplit(".", 1)
    module = importlib.import_module(mod_path)
    return getattr(module, attr)


def load_model(wrapper: Optional[str], model_name: str, device: str):
    """Load model without any PEFT configuration."""
    # Set device_map for CPU loading if needed
    device_map = "cpu" if device == "cpu" else "auto"

    if wrapper:
        cls = dynamic_import(wrapper)
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(
                model_name, device_map=device_map, torch_dtype=torch.float32 if device == "cpu" else torch.float16
            )
        else:
            model = cls(model_name)
    else:
        model = AutoModel.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.float32 if device == "cpu" else torch.float16
        )

    # Only move to device if not using device_map
    if device_map == "cpu":
        model.eval()
    else:
        model.to(device)
        model.eval()
    return model


def clear_model_from_memory(model):
    """Clear model from GPU memory."""
    import gc

    import torch

    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def print_lora_analysis(model, stage_name=""):
    """Print LoRA analysis for a model."""
    if stage_name:
        print(f"\n{'=' * 60}")
        print(f"LoRA Analysis - {stage_name}")
        print("=" * 60)

    if not has_peft_config(model):
        print("No PEFT configuration found in model.")
        return False

    print("\nLoRA Layers:")
    print("=" * 80)
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_modules.append(name)

    if not lora_modules:
        print("  No LoRA modules found (model may not be initialized with adapters yet)")
        return False

    # Group by layer pattern for condensed output
    import re
    from collections import defaultdict

    grouped = defaultdict(list)

    for module_name in sorted(lora_modules):
        # Extract pattern by removing layer numbers
        pattern = module_name
        # Replace layer numbers with {N}
        pattern = re.sub(r"\.layers\.\d+\.", ".layers.{N}.", pattern)
        grouped[pattern].append(module_name)

    print("Condensed view (layer patterns):")
    for pattern, modules in sorted(grouped.items()):
        layer_nums = []
        for module in modules:
            match = re.search(r"\.layers\.(\d+)\.", module)
            if match:
                layer_nums.append(int(match.group(1)))

        if layer_nums:
            if len(set(layer_nums)) > 1:
                layer_range = f"{min(layer_nums)}-{max(layer_nums)}"
            else:
                layer_range = str(layer_nums[0])
            print(f"  {pattern} (layers {layer_range}, {len(modules)} modules)")
        else:
            print(f"  {pattern} ({len(modules)} modules)")

    print(f"\nTotal LoRA modules: {len(lora_modules)}")

    # Show first few actual module names for reference
    print("\nFirst few actual module names:")
    for module_name in sorted(lora_modules)[:5]:
        print(f"  {module_name}")
    if len(lora_modules) > 5:
        print(f"  ... and {len(lora_modules) - 5} more")

    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze model layers and LoRA configurations.")
    parser.add_argument(
        "--model",
        required=True,
        help="HF model name or local path",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to load model on (cuda/cpu)"
    )
    parser.add_argument("--cpu", action="store_true", help="Force loading model on CPU")
    parser.add_argument(
        "--wrapper",
        default=None,
        help="Optional dotted path to wrapped model class (e.g. colpali_engine.models.ColIdefics3)",
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="Print the full model structure with layer names and dimensions",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file containing peft_config to apply",
    )

    args = parser.parse_args()

    has_lora = False
    model_to_use = None

    # Override device if --cpu flag is set
    if args.cpu:
        args.device = "cpu"
        print("Forcing CPU mode due to --cpu flag")

    # If wrapper is provided, use wrapper directly
    if args.wrapper:
        print(f"Loading model with wrapper: {args.wrapper} on {args.device}")
        try:
            model_to_use = load_model(args.wrapper, args.model, args.device)
        except Exception as e:
            print(f"Failed to load with wrapper: {e}")
            return
    else:
        # Only try standard transformers if no wrapper provided
        print(f"Loading model with standard transformers: {args.model} on {args.device}")
        try:
            # Use consistent loading approach
            device_map = "cpu" if args.device == "cpu" else "auto"
            model_to_use = AutoModel.from_pretrained(
                args.model, device_map=device_map, torch_dtype=torch.float32 if args.device == "cpu" else torch.float16
            )
            if device_map != "cpu":
                model_to_use.to(args.device)
            model_to_use.eval()
        except Exception as e:
            print(f"Failed to load with standard transformers: {e}")
            return

    # If config provided, apply it
    if args.config:
        print(f"\nApplying config from: {args.config}")

        try:
            # First check if this is a train_only_linear config
            import io
            import sys

            import configue

            config_path = Path(args.config)

            # Temporarily suppress output from config loading
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            config = configue.load(config_path, sub_path="config")
            sys.stdout = old_stdout

            if (
                hasattr(config, "train_only_linear")
                and config.train_only_linear
                and (not hasattr(config, "peft_config") or config.peft_config is None)
            ):
                print("\nConfig uses 'train_only_linear: true' mode")
                print("This trains only the linear projection layer without LoRA/PEFT.")

                # Apply the configuration to get accurate trainable params
                # First freeze all parameters
                for param in model_to_use.parameters():
                    param.requires_grad = False

                # Then unfreeze only the output projection layer
                # Check for both naming conventions: 'linear' (ColIdefics3) and 'custom_text_proj' (most others)
                projection_layer = None
                projection_name = None

                if hasattr(model_to_use, "linear"):
                    projection_layer = model_to_use.linear
                    projection_name = "linear"
                elif hasattr(model_to_use, "custom_text_proj"):
                    projection_layer = model_to_use.custom_text_proj
                    projection_name = "custom_text_proj"

                if projection_layer is not None:
                    for param in projection_layer.parameters():
                        param.requires_grad = True
                    param_count = sum(p.numel() for p in projection_layer.parameters())
                    print(f"Enabled training for '{projection_name}' projection layer with {param_count:,} parameters")
                else:
                    print("Warning: No output projection layer ('linear' or 'custom_text_proj') found in model.")

                # Use standard trainable params summary
                print_trainable_params_summary(model_to_use)
            else:
                # Try to load and apply PEFT config
                peft_config = load_peft_config_from_yaml(args.config)
                model_with_peft = get_peft_model(model_to_use, peft_config)
                print("PEFT configuration applied successfully.")
                print_trainable_params_summary(model_with_peft)

                # Check for LoRA after applying config
                has_lora = print_lora_analysis(model_with_peft, "Model with Applied PEFT Config")

                if has_lora:
                    print("\nSuccessfully applied and analyzed PEFT config.")
                    model_to_use = model_with_peft  # Update for print_model
                else:
                    print("\nNo LoRA adapters found even after applying config.")

        except ValueError as e:
            # Check if this is the train_only_linear case from load_peft_config_from_yaml
            if "train_only_linear" in str(e):
                print(f"\n{e}")
            else:
                print(f"Error with config: {e}")
        except Exception as e:
            print(f"Error applying config: {e}")

    else:
        # No config provided, just analyze the model as-is
        print_trainable_params_summary(model_to_use)
        has_lora = print_lora_analysis(model_to_use, "Model Analysis")
        if not has_lora:
            print("\nNo LoRA adapters found.")

    # Always show model structure if requested
    if args.print_model and model_to_use is not None:
        print("\nFull Model Structure:")
        print("=" * 80)
        print(model_to_use)


if __name__ == "__main__":
    main()
