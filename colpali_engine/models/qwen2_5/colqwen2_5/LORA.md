LoRA Configuration Strategies for ColBERT‑Style Multimodal Retrieval
===================================================================

Goal
----
Optimize adaptation for late‑interaction (ColBERT) retrieval where we embed:
* Text queries (short, purely linguistic)
* Visual documents (rasterized pages → patch tokens)
into a shared low‑dimensional (e.g. 128d) space scored via MaxSim over token embeddings.


Current (Baseline) Style (Broad Attention + MLP)
-----------------------------------------------
**Location:** scripts/configs/qwen2_5/lora/vidore_v0_2.yml
**Pattern:** 


```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
    # target_modules: '(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)
```
```bash
# vidore v0.2
uv run python scripts/analyze_lora_params.py \
--model vidore/colqwen2.5-base \
--config scripts/configs/qwen2_5/lora/vidore_v0_2.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,814,822,016
#   Trainable parameters: 59,936,768
#   Percentage trainable: 1.57%
```

The model we will be finetuning does not already have lora configurations which is at `Qwen/Qwen2.5-VL-3B-Instruct`. We use the ColQwen2_5 wrapper which changes the module names and adds a custom_text_proj to project the outputs to 128 dim.

```bash
❯ uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
Forcing CPU mode due to --cpu flag
Loading model with wrapper: colpali_engine.models.ColQwen2_5 on cpu

Trainable Parameters Summary:
  Total parameters: 3,754,885,248
  Trainable parameters: 3,754,885,248
  Percentage trainable: 100.00%

============================================================
LoRA Analysis - Model Analysis
============================================================
No PEFT configuration found in model.

No LoRA adapters found.

Full Model Structure:
================================================================================
ColQwen2_5(
  (visual): Qwen2_5_VisionTransformerPretrainedModel(
    (patch_embed): Qwen2_5_VisionPatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2_5_VLVisionBlock(
        (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
        (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
        (attn): Qwen2_5_VLVisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): Qwen2_5_VLMLP(
          (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
          (act_fn): SiLU()
        )
      )
    )
    (merger): Qwen2_5_VLPatchMerger(
      (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=2048, bias=True)
      )
    )
  )
  (model): Qwen2_5_VLModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-35): 36 x Qwen2_5_VLDecoderLayer(
        (self_attn): Qwen2_5_VLSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=256, bias=True)
          (v_proj): Linear(in_features=2048, out_features=256, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2_5_VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2_5_VLRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
  (custom_text_proj): Linear(in_features=2048, out_features=128, bias=True)
)

```

Module Structure for LoRA Targeting
-----------------------------------
When using base model `Qwen/Qwen2.5-VL-3B-Instruct` with `ColQwen2_5` wrapper, the module naming follows this structure:

### Core Components
- **Final projection head:** `custom_text_proj` (added by ColQwen2_5 wrapper, 2048→128 dimensions)
- **Vision-language merger:** `visual.merger.mlp` - Sequential with 3 layers:
  - `visual.merger.mlp.0`: Linear(5120→5120) - **LoRA targetable**
  - `visual.merger.mlp.1`: GELU activation - **Not LoRA targetable**
  - `visual.merger.mlp.2`: Linear(5120→2048) - **LoRA targetable**

### Language Model (36 layers, indexed 0-35)
- **Path pattern:** `model.layers.{0-35}.{component}`
- **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP projections:** `gate_proj`, `up_proj`, `down_proj`
- **Example:** `model.layers.0.self_attn.q_proj`

### Vision Model (32 blocks, indexed 0-31)
- **Path pattern:** `visual.blocks.{0-31}.{component}`
- **Attention component:** `attn.qkv` (combined), `attn.proj`
- **MLP components:** `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
- **Example:** `visual.blocks.0.attn.proj`

### Important Notes
- **Vision attention structure:** Uses combined `qkv` projection unlike separate q/k/v in language model
- **Layer numbering:** Language layers 0-35 (36 total), Vision blocks 0-31 (32 total)
- **Merger targeting:** Only `visual.merger.mlp.0` and `visual.merger.mlp.2` are Linear layers (LoRA-compatible)
- **Critical for targeting:** Consider `custom_text_proj` and `visual.merger.mlp.{0,2}` for vision-language alignment

Core Adaptation Principles
--------------------------
1. Attention projection weights (q/k/v/o) chiefly govern token‑token similarity geometry → high priority.
2. Early decoder blocks interface with modality mapping (connector output → text latent) → useful for vision alignment.
3. Late decoder blocks shape semantic abstraction + retrieval space projection → useful for contrastive separation.
4. MLP layers less critical globally; restrict to top or sandwich bands to save params.
5. `connector.modality_projection.proj` for vision-language alignment often high‐leverage with modest rank.
6. Separate adaptation paths for queries vs documents can reduce compromise between short vs long / multimodal sequences.



Configurations
---------------
Always train the linear projection w/out LoRA
- Linear projection
- Linear projection + vision connector
- Linear projection + text_model (attention)
- Linear projection + text_model (attention) + vision_model (attention)
- Linear projection + text_model (attention) + connector + vision_model (attention) 
- Linear projection + all linear layers




Configuration Details
----------------------
Each variant includes a complete YAML configuration snippet:

### 1. Base (modification of Vidore model)
**Intent:** Fully train custom_text_projection

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--wrapper colpali_engine.models.ColQwen2_5 \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-base.yaml \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,815,014,656
#   Trainable parameters: 60,129,408
#   Percentage trainable: 1.58%

```

### 2. Linear projection
**Intent:** Only the custom_text_proj projection

```yaml
# no peft_config
# uses train_only_linear: true
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-linear.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,754,885,248
#   Trainable parameters: 262,272
#   Percentage trainable: 0.01%
```

### 3. Linear projection + vision merger
**Intent:** Only the custom_text_proj projection + the vision merger MLP

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(visual\.merger\.mlp\.(0|2)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-merger.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,755,704,576
#   Trainable parameters: 819,328
#   Percentage trainable: 0.02%
```

### 4. Linear projection + language_model (attention)
**Intent:** Custom projection + language model attention alignment 

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-language.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,769,893,120
#   Trainable parameters: 15,007,872
#   Percentage trainable: 0.40%
```


### 5. Linear projection + language_model (attention) + vision_model (attention)
**Intent:** Custom projection + language model and vision attention layers

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*visual\.blocks\..*\.attn\.(qkv|proj)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-language-vision.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,777,757,440
#   Trainable parameters: 22,872,192
#   Percentage trainable: 0.61%
```

### 6. Linear projection + language_model (attention) + merger + vision_model (attention) 
**Intent:** Practically everything except MLP
```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*visual\.blocks\..*\.attn\.(qkv|proj)$|visual\.merger\.mlp\.(0|2)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-language-merger-vision.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,778,314,496
#   Trainable parameters: 23,429,248
#   Percentage trainable: 0.62%
```

### 7. Linear projection + all linear layers
**Intent:** Upper bound on LoRA training
```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: "all-linear"
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--config scripts/configs/qwen2_5/lora/train_qwen2_5-all.yaml \
--wrapper colpali_engine.models.ColQwen2_5 \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 3,837,874,432
#   Trainable parameters: 82,989,184
#   Percentage trainable: 2.16%
```


## Important Regex Pattern Notes

### Module Path Structure
- **Language attention:** `model.layers.{layer_num}.self_attn.{q_proj|k_proj|v_proj|o_proj}`
- **Language MLP:** `model.layers.{layer_num}.mlp.{gate_proj|up_proj|down_proj}`
- **Vision attention:** `visual.blocks.{block_num}.attn.{qkv|proj}`
- **Vision MLP:** `visual.blocks.{block_num}.mlp.{gate_proj|up_proj|down_proj}`
- **Merger (LoRA-compatible only):** `visual.merger.mlp.{0|2}` (Linear layers only, skip .1 which is GELU)
- **Final projection:** `custom_text_proj`

Implementation Notes
--------------------
* Verify actual internal module name paths (`model.layers`) in the loaded architecture; adjust regex if prefix differs. Use `scripts/analyze_lora_params.py --model <name> --wrapper <wrapper> --print_model` to print the concrete module names before finalizing a pattern.
* For mixed ranks (different r per module group) either:
	- Use multiple adapters and optionally merge after training.
	- Patch PEFT to accept structured per‑module rank (advanced).
* Always set `remove_unused_columns=False` (already done) to avoid stripping necessary fields.
* Monitor W&B or logging backend for diagnostics at specified logging steps to avoid overhead every batch.
