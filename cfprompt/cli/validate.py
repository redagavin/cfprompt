"""Validate command — schema-only check on a YAML config."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import StudyConfig, load_yaml


def validate_command(
    config_path: Path,
    no_import: bool = False,
) -> int:
    """Schema-only validation; on --no-import, skips module:function imports
    and any network calls."""
    if not config_path.exists():
        typer.echo(f"config file not found: {config_path}", err=True)
        return 1
    try:
        raw = load_yaml(config_path)
    except Exception as e:
        typer.echo(f"YAML parse error: {e}", err=True)
        return 1
    try:
        cfg = StudyConfig.model_validate(raw)
    except Exception as e:
        typer.echo(f"schema error: {e}", err=True)
        return 1
    typer.echo(f"OK: {config_path} parses and validates.")
    if not no_import:
        typer.echo(f"  target_perturbation = {cfg.target_perturbation}")
        if cfg.extract_label:
            typer.echo(f"  extract_label = {cfg.extract_label}")
    return 0
