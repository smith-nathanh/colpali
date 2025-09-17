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
**Location:** scripts/configs/idefics/train_colsmol-500-base.yaml
**Pattern:** All text model attention and MLP projections + final linear projection
**Cons:** Possibly redundant MLP adaptation; higher memory usage
NOTE: the incorrect custom_text_proj which should actually have been called `linear` based on ColIdefics3 wrapper. 

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(model.text_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
```
```bash
# vidore base
uv run python scripts/analyze_lora_params.py \
--model vidore/ColSmolVLM-Instruct-500M-base \
--config scripts/configs/idefics/train_colsmol-500-base.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
#Trainable Parameters Summary:
#  Total parameters: 477,663,552
#  Trainable parameters: 17,367,040
#  Percentage trainable: 3.64%
# NOTE: misses final linear layer
#      )
#      (linear): Linear(in_features=960, out_features=128, bias=True)
#    )
```

The model we will be finetuning does not already have lora configurations which is at `HuggingFaceTB/SmolVLM-256M-Instruct`. We use the ColIdefics3 wrapper which changes the module names.

Note: Both ColSmol base models (`HuggingFaceTB/SmolVLM-500M-Instruct` for training, `vidore/ColSmolVLM-Instruct-500M-base` for reference) use the ColIdefics3 wrapper which adds a `linear` layer for final projection and has `connector.modality_projection.proj` for vision-language alignment.


```
# Loading model: vidore/ColSmolVLM-Instruct-500M-base (which is HuggingFaceTB/SmolVLM-500M-Instruct wrapped with ColIdefics3)

Trainable Parameters Summary:
  Total parameters: 460,296,512
  Trainable parameters: 460,296,512
  Percentage trainable: 100.00%

Full Model Structure:
================================================================================
ColIdefics3(
  (model): Idefics3Model(
    (vision_model): Idefics3VisionTransformer(
      (embeddings): Idefics3VisionEmbeddings(
        (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=valid)
        (position_embedding): Embedding(1024, 768)
      )
      (encoder): Idefics3Encoder(
        (layers): ModuleList(
          (0-11): 12 x Idefics3EncoderLayer(
            (self_attn): Idefics3VisionAttention(
              (k_proj): Linear(in_features=768, out_features=768, bias=True)
              (v_proj): Linear(in_features=768, out_features=768, bias=True)
              (q_proj): Linear(in_features=768, out_features=768, bias=True)
              (out_proj): Linear(in_features=768, out_features=768, bias=True)
            )
            (layer_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            (mlp): Idefics3VisionMLP(
              (activation_fn): PytorchGELUTanh()
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
            )
            (layer_norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    )
    (connector): Idefics3Connector(
      (modality_projection): Idefics3SimpleMLP(
        (proj): Linear(in_features=12288, out_features=960, bias=False)
      )
    )
    (text_model): LlamaModel(
      (embed_tokens): Embedding(49280, 960, padding_idx=2)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=960, out_features=960, bias=False)
            (k_proj): Linear(in_features=960, out_features=320, bias=False)
            (v_proj): Linear(in_features=960, out_features=320, bias=False)
            (o_proj): Linear(in_features=960, out_features=960, bias=False)
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=960, out_features=2560, bias=False)
            (up_proj): Linear(in_features=960, out_features=2560, bias=False)
            (down_proj): Linear(in_features=2560, out_features=960, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm((960,), eps=1e-05)
          (post_attention_layernorm): LlamaRMSNorm((960,), eps=1e-05)
        )
      )
      (norm): LlamaRMSNorm((960,), eps=1e-05)
      (rotary_emb): LlamaRotaryEmbedding()
    )
  )
  (linear): Linear(in_features=960, out_features=128, bias=True)
)
```

Module Structure for LoRA Targeting
-----------------------------------
When using base model `HuggingFaceTB/SmolVLM-500M-Instruct` with `ColIdefics3` wrapper, the module naming follows this structure:

### Core Components
- **Final projection head:** `linear` (added by ColIdefics3 wrapper, 960→128 dimensions)
- **Vision-language connector:** `model.connector.modality_projection.proj` (12288→960 dimensions)

### Text Model (32 layers, indexed 0-31)
- **Path pattern:** `model.text_model.layers.{0-31}.{component}`
- **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP projections:** `gate_proj`, `up_proj`, `down_proj`
- **Example:** `model.text_model.layers.0.self_attn.q_proj`

### Vision Model (12 layers, indexed 0-11)
- **Path pattern:** `model.vision_model.encoder.layers.{0-11}.{component}`
- **Attention projections:** `q_proj`, `k_proj`, `v_proj`, `out_proj` ⚠️
- **MLP components:** `fc1`, `fc2` 
- **Example:** `model.vision_model.encoder.layers.0.self_attn.out_proj`

### Important Notes
- **Vision vs Text attention naming:** Vision model uses `out_proj` while text model uses `o_proj`
- **Layer numbering:** Text layers 0-31 (32 total), Vision layers 0-11 (12 total)
- **Critical for targeting:** Consider `linear` and `connector.modality_projection.proj` for projection adaptation

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

### 1. Base (correction of Vidore model)
**Intent:** Fix the incorrect name linear projection

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(model.text_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(linear).*$)'
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-ns-base.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 477,786,560
#   Trainable parameters: 17,490,048
#   Percentage trainable: 3.66%
```

### 2. Linear projection
**Intent:** Only the linear projection

```yaml
# no peft_config
# uses train_only_linear: true
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-linear.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 460,296,512
#   Trainable parameters: 123,008
#   Percentage trainable: 0.03%
```

### 3. Linear projection + vision connector
**Intent:** Only the linear projection for embedding output + the vision connector

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    # target_modules: '(.*(linear)$|.*(connector\.modality_projection\.proj)$)'
    target_modules: '(.*(connector\.modality_projection\.proj)$)'
    modules_to_save: ["linear"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-connector.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 460,843,456
#   Trainable parameters: 546,944
#   Percentage trainable: 0.12%
```

### 4. Linear projection + text_model (attention)
**Intent:** Linear + text alignment 

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*model\.text_model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$)'
    modules_to_save: ["linear"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-text.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 466,973,120
#   Trainable parameters: 6,676,608
#   Percentage trainable: 1.43%
```


### 5. Linear projection + text_model (attention) + vision_model (attention)
**Intent:** Linear and text model and vision attention layers

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*model\.text_model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*model\.vision_model\.encoder\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$)'
    modules_to_save: ["linear"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-text-vision.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 469,332,416
#   Trainable parameters: 9,035,904
#   Percentage trainable: 1.93%
```

### 6. Linear projection + text_model (attention) + connector + vision_model (attention) 
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
    target_modules: '(.*model\.text_model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*model\.vision_model\.encoder\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$|.*(connector\.modality_projection\.proj)$)'
    modules_to_save: ["linear"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-text-connector-vision.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 469,756,352
#   Trainable parameters: 9,459,840
#   Percentage trainable: 2.01%
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
    modules_to_save: ["linear"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model HuggingFaceTB/SmolVLM-500M-Instruct \
--config scripts/configs/idefics/lora/train_colsmol-500-all.yaml \
--wrapper colpali_engine.models.ColIdefics3 \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 483,518,912
#   Trainable parameters: 23,222,400
#   Percentage trainable: 4.80%
```




## Important Regex Pattern Notes

### Module Path Structure
- **Text attention:** `model.text_model.layers.{layer_num}.self_attn.{q_proj|k_proj|v_proj|o_proj}`
- **Text MLP:** `model.text_model.layers.{layer_num}.mlp.{gate_proj|up_proj|down_proj}`
- **Vision attention:** `model.vision_model.encoder.layers.{layer_num}.self_attn.{q_proj|k_proj|v_proj|out_proj}`
- **Vision MLP:** `model.vision_model.encoder.layers.{layer_num}.mlp.{fc1|fc2}`
- **Connector:** `model.connector.modality_projection.proj`
- **Final projection:** `linear` (not `custom_text_proj`)

### Common Regex Mistakes to Avoid
1. ❌ Using `custom_text_proj` - doesn't exist in ColSmol
2. ❌ Using `o_proj` for vision model - should be `out_proj`
3. ❌ Using `.*$` at the end - unnecessary and can cause issues
4. ❌ Missing escaping for dots in module paths
5. ✅ Always end patterns with `$` to match exact module names
6. ✅ Use proper path structure: `model.text_model.layers.` not just `text_model.layers.`

Implementation Notes
--------------------
* Verify actual internal module name paths (`text_model.layers`) in the loaded architecture; adjust regex if prefix differs (e.g. `model.text_model` vs `language_model`). Use `scripts/analyze_lora_params.py --model <name> --wrapper <wrapper> --print_model` to print the concrete module names before finalizing a pattern.
