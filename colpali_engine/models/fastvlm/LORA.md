FAstVLM
===================================================================

Goal
----
Optimize adaptation for late‑interaction (ColBERT) retrieval where we embed:
* Text queries (short, purely linguistic)
* Visual documents (rasterized pages → patch tokens)
into a shared low‑dimensional (e.g. 128d) space scored via MaxSim over token embeddings.


```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(model\.model\.layers)\..*\.(self_attn|mlp)\.(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(model\.model\.mm_projector)\.(0|2).*$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
# vidore v0.2
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--wrapper colpali_engine.models.ColFastVLM \
--config scripts/configs/fastvlm/train_colfastvlm-500-base.yaml \
--cpu \
--print_model
```

The model we will be finetuning does not already have lora configurations which is at `apple/FastVLM-0.5B`. We use the ColFastVLM wrapper which changes the module names and adds a custom_text_proj to project the outputs to 128 dim.

```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--wrapper colpali_engine.models.ColFastVLM \
--print_model
Loading model with wrapper: colpali_engine.models.ColFastVLM on cuda
The repository for apple/FastVLM-0.5B contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/apple/FastVLM-0.5B.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.

Do you wish to run the custom code? [y/N] y
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.

Trainable Parameters Summary:
  Total parameters: 622,778,208
  Trainable parameters: 622,778,208
  Percentage trainable: 100.00%

============================================================
LoRA Analysis - Model Analysis
============================================================
No PEFT configuration found in model.

No LoRA adapters found.

Full Model Structure:
================================================================================
ColFastVLM(
  (model): LlavaQwen2ForCausalLM(
    (model): LlavaQwen2Model(
      (embed_tokens): Embedding(151936, 896)
      (layers): ModuleList(
        (0-23): 24 x Qwen2DecoderLayer(
          (self_attn): Qwen2Attention(
            (q_proj): Linear(in_features=896, out_features=896, bias=True)
            (k_proj): Linear(in_features=896, out_features=128, bias=True)
            (v_proj): Linear(in_features=896, out_features=128, bias=True)
            (o_proj): Linear(in_features=896, out_features=896, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
            (up_proj): Linear(in_features=896, out_features=4864, bias=False)
            (down_proj): Linear(in_features=4864, out_features=896, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((896,), eps=1e-06)
      (rotary_emb): Qwen2RotaryEmbedding()
      (vision_tower): MobileCLIPVisionTower(
        (vision_tower): MCi(
          (model): FastViT(
            (patch_embed): Sequential(
              (0): MobileOneBlock(
                (se): Identity()
                (activation): GELU(approximate='none')
                (reparam_conv): Conv2d(3, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
              )
              (1): MobileOneBlock(
                (se): Identity()
                (activation): GELU(approximate='none')
                (reparam_conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96)
              )
              (2): MobileOneBlock(
                (se): Identity()
                (activation): GELU(approximate='none')
                (reparam_conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (network): ModuleList(
              (0): Sequential(
                (0): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96, bias=False)
                      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (1): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96, bias=False)
                      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
              )
              (1): PatchEmbed(
                (proj): Sequential(
                  (0): ReparamLargeKernelConv(
                    (activation): GELU(approximate='none')
                    (se): Identity()
                    (lkb_reparam): Conv2d(96, 192, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), groups=96)
                  )
                  (1): MobileOneBlock(
                    (se): Identity()
                    (activation): GELU(approximate='none')
                    (reparam_conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
              )
              (2): Sequential(
                (0): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (1): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (2): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (3): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (4): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (5): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (6): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (7): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (8): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (9): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (10): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (11): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192, bias=False)
                      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
              )
              (3): PatchEmbed(
                (proj): Sequential(
                  (0): ReparamLargeKernelConv(
                    (activation): GELU(approximate='none')
                    (se): Identity()
                    (lkb_reparam): Conv2d(192, 384, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), groups=192)
                  )
                  (1): MobileOneBlock(
                    (se): Identity()
                    (activation): GELU(approximate='none')
                    (reparam_conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
              )
              (4): Sequential(
                (0): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (1): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (2): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (3): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (4): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (5): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (6): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (7): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (8): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (9): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (10): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (11): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (12): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (13): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (14): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (15): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (16): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (17): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (18): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (19): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (20): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (21): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (22): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (23): RepMixerBlock(
                  (token_mixer): RepMixer(
                    (reparam_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384, bias=False)
                      (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(1536, 384, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
              )
              (5): PatchEmbed(
                (proj): Sequential(
                  (0): ReparamLargeKernelConv(
                    (activation): GELU(approximate='none')
                    (se): Identity()
                    (lkb_reparam): Conv2d(384, 768, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), groups=384)
                  )
                  (1): MobileOneBlock(
                    (se): Identity()
                    (activation): GELU(approximate='none')
                    (reparam_conv): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
              )
              (6): RepCPE(
                (reparam_conv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)
              )
              (7): Sequential(
                (0): AttentionBlock(
                  (norm): LayerNormChannel()
                  (token_mixer): MHSA(
                    (qkv): Linear(in_features=768, out_features=2304, bias=False)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768, bias=False)
                      (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(768, 3072, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (1): AttentionBlock(
                  (norm): LayerNormChannel()
                  (token_mixer): MHSA(
                    (qkv): Linear(in_features=768, out_features=2304, bias=False)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768, bias=False)
                      (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(768, 3072, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (2): AttentionBlock(
                  (norm): LayerNormChannel()
                  (token_mixer): MHSA(
                    (qkv): Linear(in_features=768, out_features=2304, bias=False)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768, bias=False)
                      (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(768, 3072, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (3): AttentionBlock(
                  (norm): LayerNormChannel()
                  (token_mixer): MHSA(
                    (qkv): Linear(in_features=768, out_features=2304, bias=False)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768, bias=False)
                      (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(768, 3072, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
              )
              (8): PatchEmbed(
                (proj): Sequential(
                  (0): ReparamLargeKernelConv(
                    (activation): GELU(approximate='none')
                    (se): Identity()
                    (lkb_reparam): Conv2d(768, 1536, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), groups=768)
                  )
                  (1): MobileOneBlock(
                    (se): Identity()
                    (activation): GELU(approximate='none')
                    (reparam_conv): Conv2d(1536, 1536, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
              )
              (9): RepCPE(
                (reparam_conv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536)
              )
              (10): Sequential(
                (0): AttentionBlock(
                  (norm): LayerNormChannel()
                  (token_mixer): MHSA(
                    (qkv): Linear(in_features=1536, out_features=4608, bias=False)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=1536, out_features=1536, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536, bias=False)
                      (bn): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(1536, 6144, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(6144, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
                (1): AttentionBlock(
                  (norm): LayerNormChannel()
                  (token_mixer): MHSA(
                    (qkv): Linear(in_features=1536, out_features=4608, bias=False)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=1536, out_features=1536, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                  )
                  (convffn): ConvFFN(
                    (conv): Sequential(
                      (conv): Conv2d(1536, 1536, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1536, bias=False)
                      (bn): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    )
                    (fc1): Conv2d(1536, 6144, kernel_size=(1, 1), stride=(1, 1))
                    (act): GELU(approximate='none')
                    (fc2): Conv2d(6144, 1536, kernel_size=(1, 1), stride=(1, 1))
                    (drop): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path): Identity()
                )
              )
            )
            (conv_exp): MobileOneBlock(
              (se): SEBlock(
                (reduce): Conv2d(3072, 192, kernel_size=(1, 1), stride=(1, 1))
                (expand): Conv2d(192, 3072, kernel_size=(1, 1), stride=(1, 1))
              )
              (activation): GELU(approximate='none')
              (reparam_conv): Conv2d(1536, 3072, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
            )
            (head): GlobalPool2D()
          )
        )
      )
      (mm_projector): Sequential(
        (0): Linear(in_features=3072, out_features=896, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=896, out_features=896, bias=True)
      )
    )
    (lm_head): Linear(in_features=896, out_features=151936, bias=False)
  )
  (custom_text_proj): Linear(in_features=896, out_features=128, bias=True)
)
```

Module Structure for LoRA Targeting
-----------------------------------
When using base model `apple/FastVLM-0.5B` with `ColFastVLM` wrapper, the module naming follows this structure:

### Core Components
- **Final projection head:** `custom_text_proj` (added by ColFastVLM wrapper, 896→128 dimensions) - **Fully trained via modules_to_save**
- **Vision-language projector:** `model.model.mm_projector` - Sequential with 3 layers:
  - `model.model.mm_projector.0`: Linear(3072→896) - **LoRA targetable**  
  - `model.model.mm_projector.1`: GELU activation - **Not LoRA targetable**
  - `model.model.mm_projector.2`: Linear(896→896) - **LoRA targetable**

### Language Model (24 layers, indexed 0-23) 
- **Path pattern:** `model.model.layers.{0-23}.{component}`
- **Attention projections:** `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`
- **MLP projections:** `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
- **Example:** `model.model.layers.0.self_attn.q_proj`

### Vision Model (MobileCLIP - Complex ConvNet Architecture)
- **Path pattern:** `model.model.vision_tower.vision_tower.model.{component}`
- **Architecture:** FastViT with RepMixer blocks, ConvFFN, and AttentionBlocks
- **Not typically LoRA targetable:** Complex convolutional architecture with BatchNorm and specialized blocks
- **Example components:** RepMixerBlock, ConvFFN, AttentionBlock with MHSA
- **Note:** Vision tower is typically frozen during ColPali-style training

### Important Notes
- **Language model structure:** Qwen2DecoderLayer with separate q/k/v/o projections
- **Layer numbering:** Language layers 0-23 (24 total), Vision tower has complex nested structure
- **MM projector targeting:** Only `model.model.mm_projector.0` and `model.model.mm_projector.2` are Linear layers (LoRA-compatible)
- **Critical for targeting:** Focus on `custom_text_proj` (fully trained) and `model.model.mm_projector.{0,2}` for vision-language alignment
- **Vision tower:** Complex MobileCLIP architecture typically left frozen (not LoRA adapted)

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

### 1. Base
**Intent:** Fully train custom_text_projection
  1. Language model attention (q/k/v/o): Critical for token similarity geometry in ColBERT-style retrieval
  2. Language model MLP: Adapts semantic processing for retrieval task
  3. MM projector: Aligns vision features with language embeddings - crucial for multimodal retrieval
  4. Custom projection (full training): The final 896→128 projection learns the retrieval embedding space
  5. Vision tower (frozen): Pre-trained visual features are sufficient; saves parameters and training time

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(model\.model\.layers)\..*\.(self_attn|mlp)\.(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(model\.model\.mm_projector)\.(0|2).*$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--wrapper colpali_engine.models.ColFastVLM \
--config scripts/configs/fastvlm/train_colfastvlm-500-base.yaml \
--cpu \
--print_model
# Trainable Parameters Summary:
#   Total parameters: 640,673,760
#   Trainable parameters: 17,895,552
#   Percentage trainable: 2.79%
```

### 2. Linear projection
**Intent:** Only the custom_text_proj projection

```yaml
# no peft_config
# uses train_only_linear: true
```
```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--config scripts/configs/fastvlm/train_colfastvlm-linear.yaml \
--wrapper colpali_engine.models.ColFastVLM \
--cpu \
--print_model

```

### 3. Linear projection + vision-language projector
**Intent:** Only the custom_text_proj projection + the MM projector Linear layers

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(model\.model\.mm_projector\.(0|2)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--config scripts/configs/fastvlm/train_colfastvlm-projector.yaml \
--wrapper colpali_engine.models.ColFastVLM \
--cpu \
--print_model

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
    target_modules: '(.*model\.model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--config scripts/configs/fastvlm/train_colfastvlm-language.yaml \
--wrapper colpali_engine.models.ColFastVLM \
--cpu \
--print_model

```


### 5. Linear projection + language_model (attention) + vision_model (attention)
**Intent:** Custom projection + language model attention + vision tower attention (Note: FastVLM vision tower uses Conv architecture, not standard attention)

```yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*model\.model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*model\.model\.vision_tower\.vision_tower\.model\.network\.7\..*\.token_mixer\.(qkv|proj)$|.*model\.model\.vision_tower\.vision_tower\.model\.network\.10\..*\.token_mixer\.(qkv|proj)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--config scripts/configs/fastvlm/train_colfastvlm-language-vision.yaml \
--wrapper colpali_engine.models.ColFastVLM \
--cpu \
--print_model

```

### 6. Linear projection + language_model (attention) + mm_projector + vision_model (attention) 
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
    target_modules: '(.*model\.model\.layers\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$|.*model\.model\.vision_tower\.vision_tower\.model\.network\.7\..*\.token_mixer\.(qkv|proj)$|.*model\.model\.vision_tower\.vision_tower\.model\.network\.10\..*\.token_mixer\.(qkv|proj)$|model\.model\.mm_projector\.(0|2)$)'
    modules_to_save: ["custom_text_proj"]
```
```bash
uv run python scripts/analyze_lora_params.py \
--model apple/FastVLM-0.5B \
--config scripts/configs/fastvlm/train_colfastvlm-language-projector-vision.yaml \
--wrapper colpali_engine.models.ColFastVLM \
--cpu \
--print_model
```

### 7. Linear projection + all linear layers
**Intent:** Upper bound on LoRA training (Note: FastVLM vision has many Conv layers, this targets all Linear layers)
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
--model apple/FastVLM-0.5B \
--config scripts/configs/fastvlm/train_colfastvlm-all.yaml \
--wrapper colpali_engine.models.ColFastVLM \
--cpu \
--print_model

```


## Important Regex Pattern Notes

### Module Path Structure
- **Language attention:** `model.model.layers.{layer_num}.self_attn.{q_proj|k_proj|v_proj|o_proj}`
- **Language MLP:** `model.model.layers.{layer_num}.mlp.{gate_proj|up_proj|down_proj}`
- **Vision tower MHSA blocks:** `model.model.vision_tower.vision_tower.model.network.{7|10}.{block_num}.token_mixer.{qkv|proj}` (Only blocks 7 and 10 have AttentionBlocks with MHSA)
- **MM Projector (LoRA-compatible only):** `model.model.mm_projector.{0|2}` (Linear layers only, skip .1 which is GELU)
- **Final projection:** `custom_text_proj`

### Vision Tower Architecture Notes
FastVLM uses a MobileCLIP vision tower with a FastViT architecture that is primarily convolutional:
- Most vision processing uses RepMixerBlocks with Conv2d operations (not LoRA-targetable)
- Only network blocks 7 and 10 contain AttentionBlocks with MHSA (Multi-Head Self Attention) that have Linear projections
- The vision tower is typically frozen during ColPali-style training to preserve pre-trained visual features

Implementation Notes
--------------------
* Verify actual internal module name paths (`model.layers`) in the loaded architecture; adjust regex if prefix differs. Use `scripts/analyze_lora_params.py --model <name> --wrapper <wrapper> --print_model` to print the concrete module names before finalizing a pattern.
* For mixed ranks (different r per module group) either:
	- Use multiple adapters and optionally merge after training.
	- Patch PEFT to accept structured per‑module rank (advanced).
* Always set `remove_unused_columns=False` (already done) to avoid stripping necessary fields.
* Monitor W&B or logging backend for diagnostics at specified logging steps to avoid overhead every batch.
