"""Study orchestrator — owns state, stages, cache, save/load."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .cache import DiskCache
from .exceptions import ClassificationModeError, ConfigError, StageNotRunError
from .models.base import Model, Tokenizer

_logger = logging.getLogger("cfprompt")


def _init_drop_counts() -> dict[str, int]:
    return {
        "zero_edit": 0,
        "tokenization_failed": 0,
        "extraction_returned_none": 0,
        "extraction_raised": 0,
        "openai_missing_class": 0,
        "regression_nonfinite_logp": 0,
    }


@dataclass
class _StubTokenizer:
    """Internal tokenizer stub used by `Study.load()` when no model is
    re-supplied. Intentionally minimal — exposes only the Tokenizer Protocol
    surface plus a static cache_id that the stage methods would never
    consume in test-only flows."""

    saved_cache_id: str

    def encode(self, text: str) -> list[int]:
        raise StageNotRunError(
            "Cannot encode tokens: study was loaded without a target_model. "
            "Re-supply target_model in Study.load() to call generate_baselines/"
            "run_inference."
        )

    @property
    def cache_id(self) -> str:
        return self.saved_cache_id


@dataclass
class _StubModel:
    """Internal model stub used by `Study.load()` to carry the saved
    cache_id without depending on `unittest.mock`. Stage methods that call
    `target_model.score_classes(...)` etc. on this stub raise
    StageNotRunError so the user is told to re-supply a real model."""

    saved_cache_id: str

    @property
    def cache_id(self) -> str:
        return self.saved_cache_id

    @property
    def tokenizer(self) -> Tokenizer:
        return _StubTokenizer(saved_cache_id=f"{self.saved_cache_id}|stub-tok")

    def generate(self, prompts, per_prompt_seeds=None):
        raise StageNotRunError(
            "Cannot generate: study was loaded without a target_model. "
            "Re-supply target_model in Study.load() to call run_inference."
        )

    def score_classes(self, prompts, classes, per_prompt_seeds=None):
        raise StageNotRunError(
            "Cannot score classes: study was loaded without a target_model. "
            "Re-supply target_model in Study.load() to call run_inference."
        )

    def close(self) -> None:
        pass


class Study:
    """Top-level orchestrator. See spec §5.1 for the full API."""

    def __init__(
        self,
        data: pd.DataFrame,
        perturb_column: str,
        target_perturbation: Callable[[str], str],
        prompt_template: str,
        target_model: Model,
        paraphrase_model: Model,
        classes: list[str] | None = None,
        extract_label: Callable[[str], str | None] | None = None,
        direction_column: str | None = None,
        outcome_class_column: str | None = None,
        alternative: Literal["greater", "less"] | None = None,
        tolerance: float = 0.5,
        max_retries: int = 50,
        cache_dir: str | Path | None = None,
        seed: int = 42,
        n_bootstrap: int = 1000,
    ) -> None:
        if classes is not None and extract_label is not None:
            raise ConfigError(
                "Mode is mutually exclusive: set classes=[...] for "
                "classification OR extract_label=fn for free-form, not both."
            )
        if classes is None and extract_label is None:
            raise ConfigError(
                "Mode required: pass classes=[...] (classification) or "
                "extract_label=fn (free-form)."
            )

        self.mode: Literal["classification", "free_form"] = (
            "classification" if classes is not None else "free_form"
        )

        if perturb_column not in data.columns:
            raise ConfigError(
                f"perturb_column={perturb_column!r} not found in data; "
                f"columns are: {list(data.columns)}"
            )

        directional_kwargs = {
            "direction_column": direction_column,
            "outcome_class_column": outcome_class_column,
            "alternative": alternative,
        }
        set_directional = {k for k, v in directional_kwargs.items() if v is not None}
        if set_directional and len(set_directional) != 3:
            missing = set(directional_kwargs) - set_directional
            raise ConfigError(
                f"Directional analysis requires all of "
                f"{{direction_column, outcome_class_column, alternative}}; "
                f"missing: {sorted(missing)}."
            )
        self.directional = bool(set_directional)

        if self.directional:
            if self.mode == "free_form":
                raise ConfigError(
                    "Directional regression requires classification mode "
                    "(classes=...). Free-form mode has no class probabilities "
                    "to regress. Either drop direction_column/"
                    "outcome_class_column/alternative, or switch to "
                    "classification mode by passing classes=[...] instead of "
                    "extract_label."
                )
            if direction_column not in data.columns:
                raise ConfigError(f"direction_column={direction_column!r} not found in data.")
            if outcome_class_column not in data.columns:
                raise ConfigError(
                    f"outcome_class_column={outcome_class_column!r} not found in data."
                )
            if alternative not in ("greater", "less"):
                raise ConfigError(f"alternative must be 'greater' or 'less'; got {alternative!r}.")
            self._validate_direction_column(data, direction_column)
            self._validate_outcome_class_column(data, outcome_class_column, classes)

        if self.mode == "classification":
            self._openai_preflight(target_model, classes)

        self.data = data.copy()
        self.perturb_column = perturb_column
        self.target_perturbation = target_perturbation
        self.prompt_template = prompt_template
        self.target_model = target_model
        self.paraphrase_model = paraphrase_model
        self.classes = list(classes) if classes is not None else None
        self.extract_label = extract_label
        self.direction_column = direction_column
        self.outcome_class_column = outcome_class_column
        self.alternative = alternative
        self.tolerance = tolerance
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.n_bootstrap = n_bootstrap

        self._baselines_df: pd.DataFrame | None = None
        self._inference_df: pd.DataFrame | None = None
        self._drop_counts: dict[str, int] = _init_drop_counts()
        self._baseline_refused_count: int = 0
        self._baseline_refused_sample_ids: list = []
        self._n_input: int = len(data)
        self._clipped_log_prob_count: int = 0
        self._clipped_kl_count: int = 0
        self._loaded_target_cache_id: str | None = None
        self._loaded_paraphrase_cache_id: str | None = None
        self._loaded_from_path: str | None = None
        self._allow_cache_id_mismatch: bool = False

        self._paraphrase_cache: DiskCache | None = (
            DiskCache(self.cache_dir, namespace="paraphrase")
            if self.cache_dir is not None
            else None
        )
        self._inference_cache: DiskCache | None = (
            DiskCache(self.cache_dir, namespace="inference") if self.cache_dir is not None else None
        )

    def _check_cache_id_match(self, stage_name: str) -> None:
        loaded = self._loaded_target_cache_id
        if loaded is not None and loaded != self.target_model.cache_id:
            msg = (
                f"loaded target_model.cache_id={loaded!r} but invoking "
                f"{stage_name} with re-supplied target_model.cache_id="
                f"{self.target_model.cache_id!r}"
            )
            if self._allow_cache_id_mismatch:
                _logger.warning(msg)
            else:
                raise ConfigError(msg)
        loaded_p = self._loaded_paraphrase_cache_id
        if loaded_p is not None and loaded_p != self.paraphrase_model.cache_id:
            msg = (
                f"loaded paraphrase_model.cache_id={loaded_p!r} but "
                f"invoking {stage_name} with re-supplied "
                f"paraphrase_model.cache_id={self.paraphrase_model.cache_id!r}"
            )
            if self._allow_cache_id_mismatch:
                _logger.warning(msg)
            else:
                raise ConfigError(msg)

    @staticmethod
    def _validate_direction_column(data: pd.DataFrame, col: str) -> None:
        vals = pd.to_numeric(data[col], errors="coerce")
        zero_rows = data.index[vals == 0].tolist()
        nan_rows = data.index[vals.isna()].tolist()
        inf_rows = data.index[np.isinf(vals.fillna(0))].tolist()
        if zero_rows or nan_rows or inf_rows:
            invalid_count = len(zero_rows) + len(nan_rows) + len(inf_rows)
            parts = []
            if zero_rows:
                parts.append(f"zero at rows {zero_rows[:10]!r}")
                if len(zero_rows) > 10:
                    parts[-1] += f" (+{len(zero_rows) - 10} more)"
            if nan_rows:
                parts.append(f"NaN at rows {nan_rows[:10]!r}")
            if inf_rows:
                parts.append(f"Inf at rows {inf_rows[:10]!r}")
            raise ConfigError(
                f"direction_column={col!r} has {invalid_count} invalid values: "
                + "; ".join(parts)
                + ". Use ±1 (or any nonzero magnitude) per row; the package "
                "generates the internal 0-coded reference row itself for the "
                "level regression."
            )

    @staticmethod
    def _validate_outcome_class_column(data: pd.DataFrame, col: str, classes: list[str]) -> None:
        invalid_mask = ~data[col].isin(classes)
        if invalid_mask.any():
            offenders = data.loc[invalid_mask, col].value_counts().to_dict()
            invalid_count = int(invalid_mask.sum())
            offending_rows = data.index[invalid_mask].tolist()[:10]
            raise ConfigError(
                f"outcome_class_column={col!r} has {invalid_count} invalid "
                f"values: {offenders} not in classes={classes}; first 10 "
                f"offending row indices: {offending_rows}."
            )

    @staticmethod
    def _openai_preflight(target_model: Model, classes: list[str]) -> None:
        from .models.openai import OpenAIModel

        if not isinstance(target_model, OpenAIModel):
            return
        if len(classes) > 20:
            raise ClassificationModeError(
                f"OpenAIModel classification capped at 20 classes "
                f"(top_logprobs limit). Got {len(classes)}. Use HFModel or "
                f"pass extract_label=... to switch to free-form mode."
            )
        for cls in classes:
            prefixed = target_model.class_prefix + cls
            ids = target_model.tokenizer.encode(prefixed)
            if len(ids) != 1:
                raise ClassificationModeError(
                    f"OpenAIModel classification requires single-token "
                    f"classes. With class_prefix={target_model.class_prefix!r}, "
                    f"class {cls!r} tokenizes as {prefixed!r} → {len(ids)} "
                    f"tokens ({ids}). Use HFModel or pass extract_label=... "
                    f"to switch to free-form mode."
                )
