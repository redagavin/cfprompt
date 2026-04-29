"""Pydantic schema for cfprompt YAML configs + safe loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str
    perturb_column: str
    target_perturbation: str
    prompt_template: str
    target_model: ModelConfig
    paraphrase_model: ModelConfig

    classes: list[str] | None = None
    extract_label: str | None = None

    direction_column: str | None = None
    outcome_class_column: str | None = None
    alternative: Literal["greater", "less"] | None = None

    tolerance: float = 0.5
    max_retries: int = 50

    cache_dir: str | None = None
    seed: int = 42
    n_bootstrap: int = 1000

    metrics: list[str] = Field(default_factory=list)
    regression_model: Literal["difference", "level"] = "difference"

    output: str

    @model_validator(mode="after")
    def _validate_mode_and_directional(self) -> StudyConfig:
        if self.classes is not None and self.extract_label is not None:
            raise ValueError(
                "Mode is mutually exclusive: set 'classes' OR 'extract_label', not both."
            )
        if self.classes is None and self.extract_label is None:
            raise ValueError(
                "Mode required: set 'classes' (classification) or 'extract_label' (free-form)."
            )
        directional = {
            self.direction_column,
            self.outcome_class_column,
            self.alternative,
        }
        non_null = sum(v is not None for v in directional)
        if 0 < non_null < 3:
            raise ValueError(
                "Directional analysis requires all of "
                "{direction_column, outcome_class_column, alternative}."
            )
        return self


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file via yaml.safe_load (rejects !!python/* tags)."""
    with open(path) as f:
        return yaml.safe_load(f)
