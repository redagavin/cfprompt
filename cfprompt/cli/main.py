"""Top-level Typer app for `cfprompt` CLI."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="cfprompt — counterfactual prompting CLI")


@app.command()
def validate(
    config_path: Path = typer.Argument(..., help="Path to YAML config"),
    no_import: bool = typer.Option(
        False, "--no-import", help="Skip module:function imports (offline-safe)"
    ),
) -> None:
    """Schema-only check; no expensive calls."""
    from .validate import validate_command

    raise typer.Exit(code=validate_command(config_path, no_import=no_import))


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to scaffold the starter dir"),
    mode: str = typer.Option("classification", help="classification | freeform"),
    directional: bool = typer.Option(
        False,
        "--directional",
        help="Include directional regression fields in the scaffold",
    ),
) -> None:
    """Scaffold a starter directory."""
    from .init import init_command

    raise typer.Exit(code=init_command(directory, mode=mode, directional=directional))


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to YAML config"),
    output: Path = typer.Option(None, "--output", "-o"),
    cache_dir: Path = typer.Option(None, "--cache-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Execute a study from a YAML config."""
    from .run import run_command

    raise typer.Exit(
        code=run_command(
            config_path=config_path,
            output=output,
            cache_dir=cache_dir,
            dry_run=dry_run,
            log_level=log_level,
        )
    )


if __name__ == "__main__":
    app()
