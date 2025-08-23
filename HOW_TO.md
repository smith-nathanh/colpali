# HOW TO

## Contents
- [Training](#training)
  - [Configs and Datasets](#configs-and-datasets)
  - [Custom Trainer and Loss](#custom-trainer-and-loss)
  - [Launching with SkyPilot](#launching-with-skypilot)
  - [Local GPU Launch](#local-gpu-launch)
- [Evaluation (Vidore Benchmark)](#evaluation-vidore-benchmark)

---

## Training

### Configs and Datasets

Training and validation inputs for ColVision (ColPali) models are organized via YAML configs in:
```
scripts/configs/<model_name>/
```

From a config you can trace:
- Model name and architecture
- Training / evaluation dataset sources
- Hyperparameters
- LoRA configuration
- Loss function

Datasets can be declared in the config (preferred) instead of hardcoding inside `colpali_engine.utils.dataset_transformation`.

Example config snippet:
```yaml
train_dataset:
  (): colpali_engine.utils.dataset_transformation.load_train_set
  dataset_path: smith-nathanh/finance-dataset
eval_dataset:
  (): colpali_engine.utils.dataset_transformation.load_eval_set
  dataset_path: smith-nathanh/finance-dataset
```

Underlying loader functions:
```python
def load_train_set(dataset_path: str = "smith-nathanh/finance-dataset") -> ColPaliEngineDataset:
    dataset = load_dataset(dataset_path, split="train")
    return ColPaliEngineDataset(dataset, pos_target_column_name="image")

def load_eval_set(dataset_path: str = "smith-nathanh/finance-dataset") -> ColPaliEngineDataset:
    dataset = load_dataset(dataset_path, split="validation")
    return ColPaliEngineDataset(dataset, pos_target_column_name="image")
```

Finance dataset reference:
```
https://huggingface.co/datasets/smith-nathanh/finance-dataset
```

Hard negative mining datasets are saved on my GCS.

### Customizing Trainer and Loss

The Trainer objects are in:
```
colpali_engine/trainer/contrastive_trainer.py
```

Trainer class cannot currently be set via config. You must specify which trainer class to instantiate:
```
colpali_engine/trainer/colmodel_training.py
```

Loss functions:
```
colpali_engine/loss/late_interaction_losses.py
```
Loss choice is specified in the YAML config (e.g. `loss_func:` a callable reference).

### Launching with SkyPilot

Environment is managed with `uv` (see `pyproject.toml` for `skypilot[gcp]` dependency).

SkyPilot configs (example H100):
```
skypilot/H100/*.yaml
```

Launch (will pick cheapest eligible H100 across your configured regions):
```bash
sky launch -c h100 skypilot/H100/config_accel_1gpu.yaml
```

In the SkyPilot YAML you can set a run block, for example:
```bash
run: |
  echo "Starting ColBERT training..."
  cd ~/sky_workdir
  source .env
  accelerate launch scripts/train/train_colbert.py scripts/configs/qwen2_5/config_hardneg_1.yml
```

### Local GPU Launch

If already on a GPU machine:
```bash
source .env  # sets WANDB / HF tokens if present
# export HF_TOKEN="..."      # optional if private access is needed
# export WANDB_API_KEY="..." # if logging to Weights & Biases
accelerate launch scripts/train/train_colbert.py scripts/configs/qwen2_5/config_hardneg_1.yml
```

---

## Evaluation (Vidore Benchmark)

Use separate repo:
```
https://github.com/smith-nathanh/vidore-benchmark
```

It also has SkyPilot configs. For local evaluation (~24 GB VRAM might work but possibly ~40GB):

Clone and install:
```bash
uv pip install -e ".[all]"
```

Evaluate a pushed model:
```bash
vidore-benchmark evaluate-retriever \
  --model-class colqwen2.5 \
  --model-name smith-nathanh/fincolqwen0.1 \
  --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d \
  --dataset-format qa
```

