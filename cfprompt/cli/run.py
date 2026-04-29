"""Run command — execute a Study from a YAML config."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer

from cfprompt import set_log_level
from cfprompt.cli.config import StudyConfig, load_yaml
from cfprompt.models.hf import HFModel
from cfprompt.models.openai import OpenAIModel
from cfprompt.study import Study


def _resolve_callable(spec: str):
    if ":" not in spec:
        raise ValueError(f"expected module:function, got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _build_model(cfg: dict) -> Any:
    type_name = cfg.pop("type")
    if type_name == "HFModel":
        if "dtype" in cfg and isinstance(cfg["dtype"], str):
            cfg["dtype"] = _DTYPE_MAP[cfg["dtype"]]
        return HFModel(**cfg)
    if type_name == "OpenAIModel":
        return OpenAIModel(**cfg)
    if ":" in type_name:
        cls = _resolve_callable(type_name)
        return cls(**cfg)
    raise ValueError(f"unknown Model type: {type_name!r}")


def run_command(
    config_path: Path,
    output: Path | None,
    cache_dir: Path | None,
    dry_run: bool,
    log_level: str,
) -> int:
    set_log_level(log_level)
    if not config_path.exists():
        typer.echo(f"config file not found: {config_path}", err=True)
        return 1
    try:
        raw = load_yaml(config_path)
        cfg = StudyConfig.model_validate(raw)
    except Exception as e:
        typer.echo(f"config error: {e}", err=True)
        return 1

    if dry_run:
        typer.echo(f"OK: {config_path} parses and validates (dry-run).")
        return 0

    target_perturbation = _resolve_callable(cfg.target_perturbation)
    extract_label = _resolve_callable(cfg.extract_label) if cfg.extract_label else None

    target_model = _build_model(cfg.target_model.model_dump())
    paraphrase_model = _build_model(cfg.paraphrase_model.model_dump())

    data_path = Path(cfg.data)
    if data_path.suffix.lower() == ".csv":
        data = pd.read_csv(data_path)
    elif data_path.suffix.lower() in (".parquet", ".pq"):
        data = pd.read_parquet(data_path)
    else:
        typer.echo(f"unsupported data extension: {data_path.suffix}", err=True)
        return 1

    s = Study(
        data=data,
        perturb_column=cfg.perturb_column,
        target_perturbation=target_perturbation,
        prompt_template=cfg.prompt_template,
        target_model=target_model,
        paraphrase_model=paraphrase_model,
        classes=cfg.classes,
        extract_label=extract_label,
        direction_column=cfg.direction_column,
        outcome_class_column=cfg.outcome_class_column,
        alternative=cfg.alternative,
        tolerance=cfg.tolerance,
        max_retries=cfg.max_retries,
        cache_dir=cache_dir if cache_dir else cfg.cache_dir,
        seed=cfg.seed,
        n_bootstrap=cfg.n_bootstrap,
    )

    report = s.run_all(metrics=cfg.metrics, regression_model=cfg.regression_model)
    out_path = Path(output) if output else Path(cfg.output)
    if out_path.suffix.lower() == ".xlsx":
        report.to_excel(out_path)
    elif out_path.suffix.lower() == ".json":
        report.to_json(out_path)
    else:
        typer.echo(f"unsupported output extension: {out_path.suffix}", err=True)
        return 1
    typer.echo(f"wrote {out_path}")
    return 0
