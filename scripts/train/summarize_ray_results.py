#!/usr/bin/env python
"""Summarize Ray Tune experiment results stored in the configured output directory.

Usage example::

    python scripts/train/summarize_ray_results.py \
        /gcs/colqwen-ns-models/ray_results \
        --experiment fastvlm_ray_tune \
        --metric eval_ndcg@5

The script prints a ranked table of trials, their status, and the best metric
value observed. It also highlights the overall best configuration.

When `results_root` points to a `/gcs/` mount, the chosen experiment directory is
first copied into a local cache (default `./ray_results`) containing only
lightweight metadata files. If the target cache directory already contains that
experiment name, the script aborts to avoid silently overwriting local data.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@dataclass
class TrialSummary:
    trial_id: str
    trial_name: str
    status: str
    best_metric: Optional[float]
    best_step: Optional[int]
    last_report: Optional[int]
    config: Dict[str, Any]
    result_path: Path

    @property
    def has_metric(self) -> bool:
        return self.best_metric is not None and not math.isnan(self.best_metric)


def _discover_experiment_dir(results_root: Path, experiment: Optional[str]) -> Path:
    if experiment:
        target = results_root / experiment
        if not target.exists():
            raise typer.BadParameter(f"Experiment directory '{target}' does not exist.")
        return target

    experiments = [p for p in results_root.iterdir() if p.is_dir()]
    if not experiments:
        raise typer.Exit(f"No subdirectories found under {results_root}.")
    latest = max(experiments, key=lambda p: p.stat().st_mtime)
    typer.echo(
        f"No experiment specified; using most recently modified directory: {latest.name}",
        err=True,
    )
    return latest


def _load_experiment_state(exp_dir: Path) -> Dict[str, Any]:
    state_path = exp_dir / "experiment_state_latest.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError as exc:
        typer.echo(f"Warning: could not parse {state_path}: {exc}", err=True)
        return {}


def _iter_trial_dirs(exp_dir: Path) -> Iterable[Path]:
    for path in sorted(exp_dir.iterdir()):
        if path.is_dir() and (path / "result.json").exists():
            yield path


def _load_trial_summary(
    trial_dir: Path,
    metric: str,
    maximize: bool,
    trial_state_lookup: Dict[str, Dict[str, Any]],
) -> TrialSummary:
    params_path = trial_dir / "params.json"
    params: Dict[str, Any] = {}
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
        except json.JSONDecodeError as exc:
            typer.echo(f"Warning: could not parse {params_path}: {exc}", err=True)
    config = params.get("config", {})
    trial_id = params.get("trial_id", trial_dir.name)
    trial_name = params.get("trial_name", trial_dir.name)

    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    last_report: Optional[int] = None

    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            with result_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if metric in record and isinstance(record[metric], (int, float)):
                        value = float(record[metric])
                        step = int(record.get("training_iteration") or record.get("step") or 0)
                        if best_metric is None:
                            best_metric, best_step = value, step
                        else:
                            better = value > best_metric if maximize else value < best_metric
                            if better:
                                best_metric, best_step = value, step
                    if "training_iteration" in record and isinstance(record["training_iteration"], int):
                        last_report = record["training_iteration"]
        except json.JSONDecodeError as exc:
            typer.echo(f"Warning: could not parse {result_path}: {exc}", err=True)

    state_info = trial_state_lookup.get(trial_id, {})
    status = state_info.get("status", "UNKNOWN")

    # If Ray tracked metric analysis, prefer its best score
    metric_analysis = state_info.get("metric_analysis", {}).get(metric)
    if isinstance(metric_analysis, dict):
        best_from_state = metric_analysis.get("max" if maximize else "min")
        if isinstance(best_from_state, (int, float)):
            best_metric = float(best_from_state)
            best_step = metric_analysis.get("last_step") or best_step

    return TrialSummary(
        trial_id=trial_id,
        trial_name=trial_name,
        status=status,
        best_metric=best_metric,
        best_step=best_step,
        last_report=last_report,
        config=config,
        result_path=result_path,
    )


def _build_trial_lookup(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    trials = state.get("trials")
    if not isinstance(trials, list):
        return {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in trials:
        if not isinstance(entry, dict):
            continue
        trial_id = entry.get("trial_id") or entry.get("id")
        if not isinstance(trial_id, str):
            continue
        lookup[trial_id] = entry
    return lookup


def _is_remote_path(path: Path) -> bool:
    return str(path).startswith("/gcs/")


def _sync_experiment_dir(remote_exp_dir: Path, local_cache: Path) -> Path:
    """Copy select trial metadata from a remote experiment directory into a local cache."""
    local_cache = local_cache.expanduser().resolve()
    local_cache.mkdir(parents=True, exist_ok=True)

    local_exp_dir = local_cache / remote_exp_dir.name
    if local_exp_dir.exists():
        raise typer.BadParameter(
            f"Experiment directory '{local_exp_dir}' already exists locally; remove it or choose a different --local-cache.")

    try:
        _copy_experiment_metadata(remote_exp_dir, local_exp_dir)
    except Exception:
        if local_exp_dir.exists():
            shutil.rmtree(local_exp_dir, ignore_errors=True)
        raise

    return local_exp_dir


def _copy_experiment_metadata(remote_exp_dir: Path, local_exp_dir: Path) -> None:
    """Copy only the small tracking files required for scoring summaries."""
    local_exp_dir.mkdir(parents=True, exist_ok=False)

    for state_file in remote_exp_dir.glob("experiment_state*.json"):
        if state_file.is_file():
            shutil.copy2(state_file, local_exp_dir / state_file.name)

    for variant_file in remote_exp_dir.glob("basic-variant-state*.json"):
        if variant_file.is_file():
            shutil.copy2(variant_file, local_exp_dir / variant_file.name)

    tuner_file = remote_exp_dir / "tuner.pkl"
    if tuner_file.exists() and tuner_file.is_file():
        shutil.copy2(tuner_file, local_exp_dir / tuner_file.name)

    trial_files = {"result.json", "progress.csv", "params.json", "params.pkl", "error.txt"}

    for trial_dir in sorted(p for p in remote_exp_dir.iterdir() if p.is_dir()):
        files_to_copy = [f for f in trial_files if (trial_dir / f).exists()]
        if not files_to_copy:
            continue
        local_trial_dir = local_exp_dir / trial_dir.name
        local_trial_dir.mkdir(parents=False, exist_ok=False)
        for filename in files_to_copy:
            shutil.copy2(trial_dir / filename, local_trial_dir / filename)

@app.command()
def summarize(
    results_root: Path = typer.Argument(..., help="Root directory containing Ray Tune experiment folders."),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Experiment subdirectory name (defaults to latest modified).",
    ),
    metric: str = typer.Option(
        "eval_ndcg@5",
        "--metric",
        "-m",
        help="Metric key to optimize when ranking trials.",
    ),
    mode: str = typer.Option(
        "max",
        "--mode",
        help="Optimization mode for the metric ('max' or 'min').",
    ),
    top_k: int = typer.Option(5, "--top-k", help="Number of top trials to display."),
    show_config: bool = typer.Option(
        True,
        "--show-config/--hide-config",
        help="Display the best trial configuration at the end.",
    ),
    local_cache: Path = typer.Option(
        Path("./ray_results"),
        "--local-cache",
        help="Directory where remote experiments (e.g. /gcs/ paths) are cached; existing experiment names cause an error.",
    ),
) -> None:
    """Print a summary table for a Ray Tune experiment."""

    results_root = results_root.expanduser().resolve()
    if not results_root.exists():
        raise typer.BadParameter(f"Results root '{results_root}' does not exist.")

    exp_dir = _discover_experiment_dir(results_root, experiment)

    if _is_remote_path(results_root):
        exp_dir = _sync_experiment_dir(exp_dir, local_cache)
        typer.echo(f"Synced remote experiment to {exp_dir}", err=True)

    maximize = mode.lower() == "max"
    if mode.lower() not in {"max", "min"}:
        raise typer.BadParameter("--mode must be 'max' or 'min'.")

    state = _load_experiment_state(exp_dir)
    trial_state_lookup = _build_trial_lookup(state)

    trial_summaries: List[TrialSummary] = []
    for trial_dir in _iter_trial_dirs(exp_dir):
        summary = _load_trial_summary(trial_dir, metric, maximize, trial_state_lookup)
        trial_summaries.append(summary)

    if not trial_summaries:
        raise typer.Exit(f"No trials with result.json found under {exp_dir}.")

    def sort_key(trial: TrialSummary) -> tuple[int, float]:
        if not trial.has_metric:
            return (0, float("-inf"))
        score = trial.best_metric if trial.best_metric is not None else float("nan")
        return (1, score if maximize else -score)

    trial_summaries.sort(key=sort_key, reverse=True)

    header = f"Experiment: {exp_dir.name} | Metric: {metric} ({mode}) | Trials: {len(trial_summaries)}"
    typer.echo(header)
    typer.echo("-" * len(header))
    typer.echo(f"{'Rank':<5} {'Trial ID':<15} {'Status':<12} {'Best':>12} {'Step':>8} {'Last It':>8}")

    for idx, trial in enumerate(trial_summaries[:top_k], start=1):
        best_display = f"{trial.best_metric:.6f}" if trial.best_metric is not None else "-"
        step_display = str(trial.best_step) if trial.best_step is not None else "-"
        last_display = str(trial.last_report) if trial.last_report is not None else "-"
        typer.echo(
            f"{idx:<5} {trial.trial_id:<15} {trial.status:<12} {best_display:>12} {step_display:>8} {last_display:>8}"
        )

    if show_config:
        best_trial = next((t for t in trial_summaries if t.has_metric), trial_summaries[0])
        typer.echo()
        typer.echo(f"Best trial: {best_trial.trial_id} ({best_trial.status})")
        if best_trial.best_metric is not None:
            typer.echo(f"Best {metric}: {best_trial.best_metric:.6f} at step {best_trial.best_step}")
        typer.echo("Config:")
        typer.echo(json.dumps(best_trial.config, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
