"""Init command — scaffolds a starter cfprompt study directory."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

TEMPLATES = Path(__file__).parent / "templates"


def init_command(directory: Path, mode: str, directional: bool = False) -> int:
    if mode not in {"classification", "freeform"}:
        typer.echo(f"--mode must be 'classification' or 'freeform'; got {mode!r}", err=True)
        return 1
    directory.mkdir(parents=True, exist_ok=True)
    targets = [
        directory / "config.yaml",
        directory / "perturbations.py",
        directory / "README.md",
        directory / ".gitignore",
    ]
    existing = [p for p in targets if p.exists()]
    if existing:
        names = [p.name for p in existing]
        typer.echo(
            f"refusing to overwrite existing files in {directory}: {names}. "
            f"Move or delete them before re-running cfprompt init.",
            err=True,
        )
        return 1
    src_yaml = TEMPLATES / f"config_{mode}.yaml"
    shutil.copy(src_yaml, directory / "config.yaml")
    shutil.copy(TEMPLATES / "perturbations_template.py", directory / "perturbations.py")
    shutil.copy(TEMPLATES / "README_template.md", directory / "README.md")
    (directory / "data").mkdir(exist_ok=True)
    (directory / ".gitignore").write_text("cache/\n*.xlsx\n*.json\n")
    if directional:
        with open(directory / "config.yaml", "a", encoding="utf-8") as f:
            f.write("\n# Directional regression fields:\n")
            f.write("direction_column: my_direction_col\n")
            f.write("outcome_class_column: my_outcome_col\n")
            f.write('alternative: "greater"\n')
    typer.echo(f"Scaffolded a cfprompt study in {directory}")
    return 0
