# FastVLM Hyperparameter Tuning Plan

This document summarizes the proposed approach for extending hyperparameter exploration around the best-performing `train_colfastvlm-500-base.yaml` configuration.

## 1. Baseline Scope
- Anchor on `scripts/configs/fastvlm/train_colfastvlm-500-base.yaml`, keeping dataset and model blocks fixed (processor, model constructor, dataset paths).
- Expose the following knobs for search:
  - `tr_args.learning_rate`
  - `tr_args.per_device_train_batch_size`
  - `tr_args.gradient_accumulation_steps`
  - `tr_args.warmup_steps`
  - `config.doc_block_size`
  - `peft_config` parameters: `r`, `lora_alpha`, `lora_dropout`
- Maintain retrieval-specific overrides (`mask_non_image_embeddings`, `fuse_in_decoder`) and LoRA `target_modules` unless a later sweep shows projector-only variants outperforming.

## 2. Search Space
- Learning rate: log-uniform in [5e-5, 4e-4]; pair warmup with LR (5–15% of total steps).
- Per-device batch size: {16, 24, 32}; combine with gradient accumulation {1, 2, 4} to keep effective batch mostly stable.
- Warmup steps: {50, 100, 200}.
- Doc block size: {32, 48, 64} to trade retrieval fidelity vs. VRAM.
- LoRA:
  - `r`: {16, 32, 48}
  - `lora_alpha`: {16, 32, 64} (scaled roughly with `r`)
  - `lora_dropout`: {0.05, 0.1, 0.15}

## 3. Early-Stopping Framework
- Use **Ray Tune** with the **Async HyperBand (ASHA)** scheduler to avoid running every trial to completion.
- Metric: primary retrieval score (e.g., MRR or Recall@k) emitted every `eval_steps` via W&B integration.
- Scheduler parameters:
  - `mode="max"`
  - `max_t`: total planned training steps (2 epochs of the current dataset)
  - `grace_period`: ~1/6 of `max_t` to allow warm-up before pruning
  - `reduction_factor`: 3 for aggressive culling

## 4. Trial Runtime Shortcuts
- Allow Ray to override `num_train_epochs` or pass a `max_steps` argument for truncated runs (e.g., 0.25–0.5 epochs) during the sweep.
- Temporarily disable `full_corpus_eval` and `compute_retrieval_metrics` until trials reach the top rung; re-enable for promoted trials.
- Optionally point `load_train_set` to a deterministic subset sampler so comparisons stay stable and faster.

## 5. Operational Flow
1. Build a lightweight wrapper that loads the base YAML, applies Ray-provided overrides, and launches `accelerate` with those settings.
2. Provision a SkyPilot Ray cluster on H100 (reuse `skypilot/H100/config_accel_1gpu.yaml` as the node definition).
3. Submit the Ray Tune experiment: 20–30 trials with ASHA, monitoring via W&B.
4. Rerun the top-performing configuration to full convergence (full epochs, full corpus evaluation) and optionally run a narrower sweep around the winning LR/batch combination.

## 6. Next Steps Checklist
- [ ] Parameterize the training entry point to accept Ray-provided overrides.
- [ ] Dry-run a single truncated trial on the SkyPilot cluster to validate logging and checkpointing.
- [ ] Launch the full Ray Tune sweep and schedule a review of top trials before promoting to long-form training.
