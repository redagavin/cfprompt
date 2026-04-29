"""Init command — scaffolds a starter cfprompt study directory."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

TEMPLATES = Path(__file__).parent / "templates"


def init_command(directory: Path, mode: str) -> int:
    if mode not in {"classification", "freeform"}:
        typer.echo(f"--mode must be 'classification' or 'freeform'; got {mode!r}", err=True)
        return 1
    directory.mkdir(parents=True, exist_ok=True)
    src_yaml = TEMPLATES / f"config_{mode}.yaml"
    shutil.copy(src_yaml, directory / "config.yaml")
    shutil.copy(TEMPLATES / "perturbations_template.py", directory / "perturbations.py")
    shutil.copy(TEMPLATES / "README_template.md", directory / "README.md")
    (directory / "data").mkdir(exist_ok=True)
    (directory / ".gitignore").write_text("cache/\n*.xlsx\n*.json\n")
    typer.echo(f"Scaffolded a cfprompt study in {directory}")
    return 0
