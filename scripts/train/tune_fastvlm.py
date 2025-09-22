"""Ray Tune orchestrator for FastVLM hyperparameter sweeps.

This script launches Ray Tune trials that reuse the existing ColPali training
stack. Each trial loads the baseline YAML config, applies Ray-provided
overrides, and runs a truncated training loop while reporting retrieval metrics
back to Ray for early stopping via ASHA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import configue
import ray
import typer
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from transformers import TrainerCallback

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer

DEFAULT_METRIC_KEYS = ("eval_ndcg@5", "eval_mrr", "eval_recall@5")

app = typer.Typer(pretty_exceptions_enable=False)


class RayTuneReportCallback(TrainerCallback):
    """Streams selected metrics from the HF Trainer logs to Ray Tune."""

    def __init__(self, metric_keys: Iterable[str]) -> None:
        self.metric_keys = set(metric_keys)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or not session.get_session():
            return

        payload: Dict[str, Any] = {key: logs[key] for key in self.metric_keys if key in logs}

        if not payload:
            return

        payload["training_iteration"] = state.global_step
        session.report(payload)


def load_training_config(config_file: Path) -> ColModelTrainingConfig:
    config = configue.load(config_file, sub_path="config")
    if not isinstance(config, ColModelTrainingConfig):
        raise TypeError("YAML must resolve to ColModelTrainingConfig")
    return config


def apply_trial_overrides(
    config: ColModelTrainingConfig,
    overrides: Dict[str, Any],
) -> None:
    tr_args = config.tr_args
    if "learning_rate" in overrides:
        tr_args.learning_rate = float(overrides["learning_rate"])
    if "per_device_train_batch_size" in overrides:
        tr_args.per_device_train_batch_size = int(overrides["per_device_train_batch_size"])
    if "gradient_accumulation_steps" in overrides:
        tr_args.gradient_accumulation_steps = int(overrides["gradient_accumulation_steps"])
    if "warmup_ratio" in overrides:
        tr_args.warmup_ratio = float(overrides["warmup_ratio"])
        # Clear any step-based warmup from the base config so the ratio governs the schedule.
        tr_args.warmup_steps = 0
    elif "warmup_steps" in overrides:
        tr_args.warmup_steps = int(overrides["warmup_steps"])
    if config.peft_config is not None:
        if "lora_r" in overrides:
            config.peft_config.r = int(overrides["lora_r"])
        if "lora_alpha" in overrides:
            config.peft_config.lora_alpha = int(overrides["lora_alpha"])
        if "lora_dropout" in overrides:
            config.peft_config.lora_dropout = float(overrides["lora_dropout"])
    if "num_train_epochs" in overrides:
        tr_args.num_train_epochs = float(overrides["num_train_epochs"])
    if "max_steps" in overrides and overrides["max_steps"] is not None:
        tr_args.max_steps = int(overrides["max_steps"])
    if "eval_steps" in overrides and overrides["eval_steps"] is not None:
        tr_args.eval_steps = int(overrides["eval_steps"])


def run_trial(
    trial_config: Dict[str, Any],
    base_config_path: Path,
    metric_keys: Iterable[str],
    static_overrides: Dict[str, Any],
    skip_save: bool,
) -> None:
    config = load_training_config(base_config_path)

    ray_session = session.get_session()
    trial_dir = Path(session.get_trial_dir()) if ray_session else Path("./fastvlm_trial")
    trial_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = trial_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    config.output_dir = str(artifact_dir)
    if config.tr_args.output_dir is None:
        config.tr_args.output_dir = str(artifact_dir)
    else:
        config.tr_args.output_dir = str(artifact_dir)

    if ray_session:
        run_suffix = f"ray-{ray_session.trial_id}"
        base_run_name = config.tr_args.run_name or "colfastvlm"
        config.tr_args.run_name = f"{base_run_name}-{run_suffix}"

    if static_overrides.get("num_train_epochs") is not None:
        config.tr_args.num_train_epochs = float(static_overrides["num_train_epochs"])
    if static_overrides.get("max_steps") is not None:
        config.tr_args.max_steps = int(static_overrides["max_steps"])
    if static_overrides.get("full_corpus_eval") is not None:
        config.full_corpus_eval = bool(static_overrides["full_corpus_eval"])
    if static_overrides.get("compute_retrieval_metrics") is not None:
        config.compute_retrieval_metrics = bool(static_overrides["compute_retrieval_metrics"])

    apply_trial_overrides(config, trial_config)

    training_app = ColModelTraining(config)
    trainer = ContrastiveTrainer(
        model=training_app.model,
        train_dataset=training_app.train_dataset,
        eval_dataset=training_app.eval_dataset,
        args=config.tr_args,
        data_collator=training_app.collator,
        loss_func=config.loss_func,
        is_vision_model=config.processor is not None,
        full_corpus_eval=config.full_corpus_eval,
        doc_block_size=config.doc_block_size,
        compute_retrieval_metrics=config.compute_retrieval_metrics,
    )
    trainer.add_callback(RayTuneReportCallback(metric_keys))

    trainer.train(resume_from_checkpoint=config.tr_args.resume_from_checkpoint)

    if config.run_eval and training_app.eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        if session.get_session():
            eval_metrics["training_iteration"] = trainer.state.global_step
            session.report(eval_metrics)

    if not skip_save:
        training_app.save()


def default_search_space() -> Dict[str, Any]:
    return {
        "learning_rate": tune.loguniform(5e-5, 5e-4),
        "per_device_train_batch_size": tune.choice([16, 32]),
        "warmup_ratio": tune.choice([0.02, 0.05, 0.1]),
        # "lora_r": tune.choice([16, 32, 48]),
        # "lora_alpha": tune.choice([16, 32, 64]),
        # "lora_dropout": tune.choice([0.05, 0.1, 0.15]),
    }


@app.command()
def main(
    config_file: Path = typer.Argument(
        ..., help="Path to the baseline FastVLM YAML (e.g. train_colfastvlm-500-base.yaml)."
    ),
    num_samples: int = typer.Option(20, "--num-samples", "-n", help="Number of Ray Tune trials."),
    metric: str = typer.Option("eval_ndcg@5", "--metric", help="Metric to optimize."),
    mode: str = typer.Option("max", "--mode", help="Optimization mode for the metric."),
    local_dir: Path = typer.Option(
        Path("./ray_results"),
        "--local-dir",
        help="Ray output directory (passed to RunConfig.storage_path).",
    ),
    experiment_name: str = typer.Option(
        "fastvlm_ray_tune",
        "--experiment-name",
        help="Name assigned to the Ray Tune experiment (becomes the folder under --local-dir).",
    ),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray address (use 'auto' for cluster)."),
    cpus_per_trial: int = typer.Option(8, "--cpus-per-trial", help="CPUs reserved per trial."),
    gpus_per_trial: float = typer.Option(1.0, "--gpus-per-trial", help="GPUs reserved per trial."),
    max_concurrent_trials: Optional[int] = typer.Option(
        None, "--max-concurrent-trials", help="Limit on parallel Ray Tune trials."
    ),
    scheduler_max_t: int = typer.Option(1000, "--scheduler-max-t", help="ASHA max_t (in training iterations)."),
    scheduler_grace_period: int = typer.Option(150, "--scheduler-grace-period", help="ASHA grace period."),
    reduction_factor: int = typer.Option(3, "--reduction-factor", help="ASHA reduction factor."),
    train_epochs: Optional[float] = typer.Option(
        None,
        "--train-epochs",
        help="Override number of epochs per trial (omit or set to 'none' to keep YAML value).",
    ),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Optional max training steps per trial."),
    full_corpus_eval: bool = typer.Option(
        False,
        "--full-corpus-eval/--no-full-corpus-eval",
        help="Run retrieval metrics against the entire eval corpus (slower).",
    ),
    retrieval_metrics: bool = typer.Option(
        True,
        "--retrieval-metrics/--no-retrieval-metrics",
        help="Compute retrieval metrics during tuning.",
    ),
    skip_save: bool = typer.Option(
        True,
        "--skip-save/--save-artifacts",
        help="Skip saving the trained weights to reduce I/O during sweeps.",
    ),
    eval_steps_override: Optional[int] = typer.Option(
        10,
        "--eval-steps",
        help="Override evaluation cadence during tuning (set null to keep YAML value).",
    ),
) -> None:
    """Launch a Ray Tune sweep for FastVLM hyperparameters."""

    try:
        config_file = config_file.resolve(strict=True)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Config file '{config_file}' does not exist.") from exc

    local_dir = local_dir.expanduser().resolve()

    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init()

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        max_t=scheduler_max_t,
        grace_period=scheduler_grace_period,
        reduction_factor=reduction_factor,
    )

    static_overrides = {
        "num_train_epochs": train_epochs,
        "max_steps": max_steps,
        "full_corpus_eval": full_corpus_eval,
        "compute_retrieval_metrics": retrieval_metrics,
        "eval_steps": eval_steps_override,
    }

    trainable = tune.with_parameters(
        run_trial,
        base_config_path=config_file,
        metric_keys=DEFAULT_METRIC_KEYS,
        static_overrides=static_overrides,
        skip_save=skip_save,
    )

    resources = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    tune_config = tune.TuneConfig(
        scheduler=scheduler,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent_trials,
    )

    run_config = air.RunConfig(
        storage_path=str(local_dir),
        name=experiment_name,
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        tune_config=tune_config,
        run_config=run_config,
        param_space=default_search_space(),
    )

    tuner.fit()


if __name__ == "__main__":
    app()
