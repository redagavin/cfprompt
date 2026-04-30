"""Validate command — schema-only check on a YAML config."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import StudyConfig, load_yaml
from .imports import resolve_callable


def validate_command(
    config_path: Path,
    no_import: bool = False,
) -> int:
    """Validate a YAML config: parse, schema-check, and (unless no_import) import
    the referenced module:function callables. SECURITY: imports execute the user's
    Python code; pass no_import=True for untrusted YAML."""
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
        try:
            resolve_callable(cfg.target_perturbation)
        except (ImportError, AttributeError, ValueError) as e:
            typer.echo(f"target_perturbation resolution failed: {e}", err=True)
            return 1
        typer.echo(f"  target_perturbation = {cfg.target_perturbation} (resolved)")
        if cfg.extract_label:
            try:
                resolve_callable(cfg.extract_label)
            except (ImportError, AttributeError, ValueError) as e:
                typer.echo(f"extract_label resolution failed: {e}", err=True)
                return 1
            typer.echo(f"  extract_label = {cfg.extract_label} (resolved)")
    return 0
