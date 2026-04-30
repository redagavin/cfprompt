"""Study orchestrator — owns state, stages, cache, save/load."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ._version import (
    __inference_algorithm_version__,
    __paraphrase_algorithm_version__,
    __version__,
)
from ._version import __version__ as _cfprompt_version
from .cache import (
    DiskCache,
    derive_seed,
    inference_cache_key,
    paraphrase_cache_key,
    safe_format,
)
from .exceptions import (
    CfpromptError,
    ClassificationModeError,
    ConfigError,
    StageNotRunError,
)
from .metrics.distributional import jsd as _jsd
from .metrics.distributional import kl as _kl
from .metrics.label import flip_rate as _flip_rate
from .metrics.label import mutual_information as _mi
from .metrics.label import phi_coefficient as _phi
from .metrics.regression import fit_difference, fit_level
from .models.base import Model, Tokenizer
from .paraphrase import generate_adjusted_paraphrase
from .report import Report
from .stats import bootstrap_diff, paired_t, regression_test
from .tokenization import token_edit_distance_pct

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
            self._hf_preflight(target_model, classes)

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
    def _hf_preflight(target_model: Model, classes: list[str]) -> None:
        from .models.hf import HFModel

        if not isinstance(target_model, HFModel):
            return
        target_model.validate_classes(classes)

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


def _generate_baselines(self) -> None:
    if self._baselines_df is not None:
        return
    self._check_cache_id_match("generate_baselines")
    self._drop_counts = _init_drop_counts()
    self._baseline_refused_count = 0
    self._baseline_refused_sample_ids = []
    rows = []
    n_input = len(self.data)
    self._n_input = n_input

    for sample_id, row in self.data.iterrows():
        original = row[self.perturb_column]
        try:
            target = self.target_perturbation(original)
        except Exception as e:
            _logger.warning("target_perturbation raised on row %r: %s", sample_id, e)
            self._drop_counts["tokenization_failed"] += 1
            continue

        if target == original:
            self._drop_counts["zero_edit"] += 1
            continue

        try:
            orig_ids = self.target_model.tokenizer.encode(original)
            target_ids = self.target_model.tokenizer.encode(target)
        except Exception as e:
            _logger.warning("tokenizer raised on row %r: %s", sample_id, e)
            self._drop_counts["tokenization_failed"] += 1
            continue

        if len(orig_ids) == 0:
            self._drop_counts["tokenization_failed"] += 1
            continue

        target_edit_pct = token_edit_distance_pct(orig_ids, target_ids)

        per_sample_seed = derive_seed(self.seed, str(sample_id))

        result = DiskCache.MISS
        if self._paraphrase_cache is not None:
            key = paraphrase_cache_key(
                stage_version=__paraphrase_algorithm_version__,
                cfprompt_version=__version__,
                original=original,
                target_perturbed=target,
                target_edit_pct=target_edit_pct,
                paraphrase_model_cache_id=self.paraphrase_model.cache_id,
                tokenizer_cache_id=self.target_model.tokenizer.cache_id,
                tolerance=self.tolerance,
                max_retries=self.max_retries,
                seed=per_sample_seed,
            )
            result = self._paraphrase_cache.get(key, default=DiskCache.MISS)

        if result is DiskCache.MISS:
            result = generate_adjusted_paraphrase(
                text=original,
                target_edit_pct=target_edit_pct,
                paraphrase_model=self.paraphrase_model,
                tokenizer=self.target_model.tokenizer,
                tolerance=self.tolerance,
                max_retries=self.max_retries,
                seed=per_sample_seed,
            )
            if self._paraphrase_cache is not None:
                self._paraphrase_cache.set(key, result)
        baseline_ids = self.target_model.tokenizer.encode(result.paraphrase)
        baseline_edit_pct = (
            token_edit_distance_pct(orig_ids, baseline_ids) if len(orig_ids) else 0.0
        )

        if result.refused:
            self._baseline_refused_count += 1
            self._baseline_refused_sample_ids.append(sample_id)
            _logger.warning(
                "sample_id=%r: all %d paraphrase attempts refused; using "
                "original as baseline (baseline edit %% = 0). This biases "
                "JSD/KL/aggregate metrics positively for this sample; "
                "regression difference biased same direction; level "
                "regression mild precision loss.",
                sample_id,
                self.max_retries + 1,
            )

        record = dict(row)
        record["sample_id"] = sample_id
        record["original"] = original
        record["target_perturbed"] = target
        record["baseline_perturbed"] = result.paraphrase
        record["target_edit_pct"] = target_edit_pct
        record["baseline_edit_pct"] = baseline_edit_pct
        record["baseline_refused"] = result.refused
        record["retries_used"] = result.retries_used
        rows.append(record)

    self._baselines_df = pd.DataFrame(rows)


Study.generate_baselines = _generate_baselines
Study.baselines_df = property(lambda self: self._baselines_df)


CONDITION_NAMES = ("original", "target", "baseline")
CONDITION_KEYS = ("original", "target_perturbed", "baseline_perturbed")


def _run_inference(self) -> None:
    if self._inference_df is not None:
        return
    self._check_cache_id_match("run_inference")
    if self._baselines_df is None:
        self.generate_baselines()
    if len(self._baselines_df) == 0:
        self._inference_df = pd.DataFrame()
        return

    rows_out = []
    n_extraction_raised = 0
    n_extraction_none = 0
    n_openai_missing = 0

    def _cache_key_for(prompt: str, sample_id, condition: str) -> str:
        return inference_cache_key(
            stage_version=__inference_algorithm_version__,
            prompt=prompt,
            target_model_cache_id=self.target_model.cache_id,
            mode=self.mode,
            classes=self.classes,
            seed=derive_seed(self.seed, str(sample_id), condition),
        )

    for _, row in self._baselines_df.iterrows():
        sample_id = row["sample_id"]
        prompts: list[str] = []
        for text_key in CONDITION_KEYS:
            mapping = dict(row)
            mapping[self.perturb_column] = row[text_key]
            prompts.append(safe_format(self.prompt_template, mapping))

        per_prompt_seeds = [
            derive_seed(self.seed, str(sample_id), cond) for cond in CONDITION_NAMES
        ]

        record = dict(row)
        if self.mode == "classification":
            cached_probs: list = [DiskCache.MISS] * 3
            need_compute_idx: list[int] = []
            if self._inference_cache is not None:
                keys: list[str] = [
                    _cache_key_for(prompts[i], sample_id, CONDITION_NAMES[i]) for i in range(3)
                ]
                for i in range(3):
                    cached_probs[i] = self._inference_cache.get(keys[i], default=DiskCache.MISS)
                    if cached_probs[i] is DiskCache.MISS:
                        need_compute_idx.append(i)
            else:
                keys = []
                need_compute_idx = [0, 1, 2]

            if need_compute_idx:
                fresh = self.target_model.score_classes(
                    [prompts[i] for i in need_compute_idx],
                    self.classes,
                    per_prompt_seeds=[per_prompt_seeds[i] for i in need_compute_idx],
                )
                for j, i in enumerate(need_compute_idx):
                    val = None if np.isnan(fresh[j]).any() else fresh[j]
                    cached_probs[i] = val
                    if self._inference_cache is not None:
                        self._inference_cache.set(keys[i], val)

            assert all(p is not DiskCache.MISS for p in cached_probs)
            if any(p is None for p in cached_probs):
                n_openai_missing += 1
                continue

            record["probs_orig"] = list(cached_probs[0])
            record["probs_target"] = list(cached_probs[1])
            record["probs_base"] = list(cached_probs[2])
            record["label_orig"] = self.classes[int(np.argmax(cached_probs[0]))]
            record["label_target"] = self.classes[int(np.argmax(cached_probs[1]))]
            record["label_base"] = self.classes[int(np.argmax(cached_probs[2]))]
        else:
            cached_gens: list = [DiskCache.MISS] * 3
            need_compute_idx = []
            if self._inference_cache is not None:
                keys = [_cache_key_for(prompts[i], sample_id, CONDITION_NAMES[i]) for i in range(3)]
                for i in range(3):
                    cached_gens[i] = self._inference_cache.get(keys[i], default=DiskCache.MISS)
                    if cached_gens[i] is DiskCache.MISS:
                        need_compute_idx.append(i)
            else:
                keys = []
                need_compute_idx = [0, 1, 2]

            if need_compute_idx:
                fresh = self.target_model.generate(
                    [prompts[i] for i in need_compute_idx],
                    per_prompt_seeds=[per_prompt_seeds[i] for i in need_compute_idx],
                )
                for j, i in enumerate(need_compute_idx):
                    cached_gens[i] = fresh[j]
                    if self._inference_cache is not None:
                        self._inference_cache.set(keys[i], fresh[j])
            assert all(g is not DiskCache.MISS for g in cached_gens)

            labels: list[str] = []
            ok = True
            for g in cached_gens:
                try:
                    label = self.extract_label(g)
                except Exception as e:
                    _logger.warning(
                        "sample_id=%r extract_label raised %s: %s",
                        sample_id,
                        type(e).__name__,
                        str(e)[:200],
                    )
                    n_extraction_raised += 1
                    ok = False
                    break
                if label is None:
                    n_extraction_none += 1
                    ok = False
                    break
                labels.append(label)
            if not ok:
                continue
            record["generation_orig"] = cached_gens[0]
            record["generation_target"] = cached_gens[1]
            record["generation_base"] = cached_gens[2]
            record["label_orig"] = labels[0]
            record["label_target"] = labels[1]
            record["label_base"] = labels[2]
        rows_out.append(record)

    self._drop_counts["openai_missing_class"] += n_openai_missing
    self._drop_counts["extraction_returned_none"] += n_extraction_none
    self._drop_counts["extraction_raised"] += n_extraction_raised
    self._inference_df = pd.DataFrame(rows_out)


Study.run_inference = _run_inference
Study.inference_df = property(lambda self: self._inference_df)


_PER_SAMPLE_METRICS = {"jsd", "kl"}
_AGGREGATE_METRICS = {"flip_rate", "mi", "phi"}
_REGRESSION_METRIC = "regression"
_FREE_FORM_OK_METRICS = _AGGREGATE_METRICS


def _check_metric_mode_compat(self, metrics: list[str]) -> None:
    for m in metrics:
        if m == _REGRESSION_METRIC and not self.directional:
            raise ConfigError(
                "Metric 'regression' requires directional study. Pass "
                "direction_column, outcome_class_column, alternative."
            )
        if self.mode == "free_form" and m not in _FREE_FORM_OK_METRICS:
            raise ConfigError(
                f"Metric {m!r} requires classification mode (free-form has no "
                f"class probabilities). Use {sorted(_FREE_FORM_OK_METRICS)}, "
                f"or switch to classification mode (classes=[...]) if your "
                f"task supports it."
            )


def _label_metric_fn(name: str) -> Callable:
    return {"flip_rate": _flip_rate, "mi": _mi, "phi": _phi}[name]


def _study_test(
    self,
    metrics: list[str],
    regression_model: str = "difference",
) -> Report:
    if self._inference_df is None:
        raise StageNotRunError(
            "Cannot test before inference. Call run_inference() first or use run_all()."
        )
    self._check_metric_mode_compat(metrics)
    self._clipped_log_prob_count = 0
    self._clipped_kl_count = 0

    df = self._inference_df
    n = len(df)
    if n < 2:
        raise CfpromptError(
            f"Cannot run paired tests with N={n}; check drop_counts: {self._drop_counts}."
        )

    results = []
    labels_orig = df["label_orig"].to_numpy()
    labels_target = df["label_target"].to_numpy()
    labels_base = df["label_base"].to_numpy()

    if self.mode == "classification":
        probs_orig = np.vstack(df["probs_orig"].apply(np.asarray).tolist())
        probs_target = np.vstack(df["probs_target"].apply(np.asarray).tolist())
        probs_base = np.vstack(df["probs_base"].apply(np.asarray).tolist())
        from .metrics.distributional import EPS

        kl_clip_mask = (
            (probs_orig.min(axis=1) < EPS)
            | (probs_target.min(axis=1) < EPS)
            | (probs_base.min(axis=1) < EPS)
        )
        self._clipped_kl_count = int(kl_clip_mask.sum())

    for m in metrics:
        if m in _PER_SAMPLE_METRICS:
            fn = _jsd if m == "jsd" else _kl
            target_arr = fn(probs_orig, probs_target)
            base_arr = fn(probs_orig, probs_base)
            results.append(paired_t(target_arr, base_arr, metric_name=m))
        elif m in _AGGREGATE_METRICS:
            metric_fn = _label_metric_fn(m)
            results.append(
                bootstrap_diff(
                    labels_orig=labels_orig,
                    labels_target=labels_target,
                    labels_baseline=labels_base,
                    metric_fn=metric_fn,
                    metric_name=m,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
            )
        elif m == _REGRESSION_METRIC:
            results.append(
                self._run_regression(regression_model, probs_orig, probs_target, probs_base)
            )
        else:
            raise ConfigError(f"Unknown metric: {m!r}")

    metadata = self._build_metadata(metrics, regression_model)
    return Report(results=results, metadata=metadata)


def _run_regression(self, which: str, probs_orig, probs_target, probs_base):
    df = self._inference_df
    direction = df[self.direction_column].astype(float).to_numpy()
    outcome_idx = np.array([self.classes.index(c) for c in df[self.outcome_class_column]])
    eps = 1e-12
    n = len(df)
    raw_orig = probs_orig[np.arange(n), outcome_idx]
    raw_target = probs_target[np.arange(n), outcome_idx]
    raw_base = probs_base[np.arange(n), outcome_idx]
    clipped_mask = (raw_orig < eps) | (raw_target < eps) | (raw_base < eps)
    self._clipped_log_prob_count = int(clipped_mask.sum())
    logp_orig = np.log(np.clip(raw_orig, eps, 1.0))
    logp_target = np.log(np.clip(raw_target, eps, 1.0))
    logp_base = np.log(np.clip(raw_base, eps, 1.0))

    if which == "difference":
        delta = logp_target - logp_base
        fit = fit_difference(direction, delta)
    elif which == "level":
        sample_id = np.concatenate([np.arange(n), np.arange(n)])
        y = np.concatenate([logp_base, logp_target])
        z_i = np.concatenate([logp_orig, logp_orig])
        direction_stack = np.concatenate([np.zeros(n), direction])
        fit = fit_level(direction_stack, y, z_i, sample_id)
    else:
        raise ConfigError(f"regression_model must be 'difference' or 'level'; got {which!r}")
    return regression_test(fit, alternative=self.alternative, n_distinct_samples=n)


def _build_metadata(self, metrics: list[str], regression_model: str) -> dict:
    bdf = self._baselines_df
    if bdf is not None and len(bdf) > 0 and "baseline_edit_pct" in bdf.columns:
        edits = bdf["baseline_edit_pct"].astype(float).to_numpy()
        edit_mean = float(edits.mean())
        edit_std = float(edits.std(ddof=1)) if len(edits) > 1 else 0.0
        within = float(
            np.mean(
                np.abs(edits - bdf["target_edit_pct"].astype(float).to_numpy()) <= self.tolerance
            )
        )
        retries_hit = int((bdf["retries_used"].astype(int) >= self.max_retries).sum())
    else:
        edit_mean = 0.0
        edit_std = 0.0
        within = 0.0
        retries_hit = 0

    return {
        "cfprompt_version": _cfprompt_version,
        "n_input": self._n_input,
        "n_after_baselines": len(bdf) if bdf is not None else 0,
        "n_after_inference": len(self._inference_df),
        "n_used_for_test": len(self._inference_df),
        "drop_counts": dict(self._drop_counts),
        "baseline_refused_count": self._baseline_refused_count,
        "baseline_refused_rate": (
            self._baseline_refused_count / self._n_input if self._n_input else 0.0
        ),
        "baseline_refused_sample_ids": list(self._baseline_refused_sample_ids),
        "paraphrase_actual_edit_pct_mean": edit_mean,
        "paraphrase_actual_edit_pct_std": edit_std,
        "paraphrase_max_retries_hit": retries_hit,
        "tolerance_target": self.tolerance,
        "tolerance_achieved_rate": within,
        "clipped_log_prob_count": getattr(self, "_clipped_log_prob_count", 0),
        "clipped_kl_count": getattr(self, "_clipped_kl_count", 0),
        "target_model": self.target_model.cache_id,
        "paraphrase_model": self.paraphrase_model.cache_id,
        "seed": self.seed,
        "n_bootstrap": self.n_bootstrap,
        "regression_model": (regression_model if _REGRESSION_METRIC in metrics else None),
        "loaded_target_cache_id": self._loaded_target_cache_id,
        "loaded_paraphrase_cache_id": self._loaded_paraphrase_cache_id,
        "loaded_from_path": self._loaded_from_path,
    }


def _study_run_all(
    self,
    metrics: list[str],
    regression_model: str = "difference",
) -> Report:
    self.generate_baselines()
    self.run_inference()
    return self.test(metrics=metrics, regression_model=regression_model)


Study._check_metric_mode_compat = _check_metric_mode_compat
Study._run_regression = _run_regression
Study._build_metadata = _build_metadata
Study.test = _study_test
Study.run_all = _study_run_all


def _study_save(self, path) -> None:
    payload = {
        "config": {
            "perturb_column": self.perturb_column,
            "prompt_template": self.prompt_template,
            "classes": self.classes,
            "mode": self.mode,
            "directional": self.directional,
            "direction_column": self.direction_column,
            "outcome_class_column": self.outcome_class_column,
            "alternative": self.alternative,
            "tolerance": self.tolerance,
            "max_retries": self.max_retries,
            "seed": self.seed,
            "n_bootstrap": self.n_bootstrap,
        },
        "data": self.data,
        "baselines_df": self._baselines_df,
        "inference_df": self._inference_df,
        "drop_counts": self._drop_counts,
        "baseline_refused_count": self._baseline_refused_count,
        "baseline_refused_sample_ids": list(self._baseline_refused_sample_ids),
        "n_input": self._n_input,
        "saved_target_cache_id": self.target_model.cache_id,
        "saved_paraphrase_cache_id": self.paraphrase_model.cache_id,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


@classmethod
def _study_load(
    cls,
    path,
    target_model=None,
    paraphrase_model=None,
    target_perturbation=None,
    extract_label=None,
    allow_cache_id_mismatch: bool = False,
):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    config = payload["config"]
    saved_t_cid = payload["saved_target_cache_id"]
    saved_p_cid = payload["saved_paraphrase_cache_id"]

    bound_target = target_model if target_model is not None else _StubModel(saved_t_cid)
    bound_para = paraphrase_model if paraphrase_model is not None else _StubModel(saved_p_cid)

    s = cls.__new__(cls)
    s.data = payload["data"]
    s.perturb_column = config["perturb_column"]
    s.target_perturbation = target_perturbation
    s.prompt_template = config["prompt_template"]
    s.target_model = bound_target
    s.paraphrase_model = bound_para
    s.classes = config["classes"]
    s.extract_label = extract_label
    s.direction_column = config["direction_column"]
    s.outcome_class_column = config["outcome_class_column"]
    s.alternative = config["alternative"]
    s.tolerance = config["tolerance"]
    s.max_retries = config["max_retries"]
    s.cache_dir = None
    s._paraphrase_cache = None
    s._inference_cache = None
    s.seed = config["seed"]
    s.n_bootstrap = config["n_bootstrap"]
    s.mode = config["mode"]
    s.directional = config["directional"]
    s._baselines_df = payload["baselines_df"]
    s._inference_df = payload["inference_df"]
    s._drop_counts = dict(payload.get("drop_counts") or _init_drop_counts())
    s._baseline_refused_count = int(payload.get("baseline_refused_count", 0))
    s._baseline_refused_sample_ids = list(payload.get("baseline_refused_sample_ids", []))
    s._n_input = int(payload.get("n_input", 0))
    s._loaded_target_cache_id = saved_t_cid
    s._loaded_paraphrase_cache_id = saved_p_cid
    s._loaded_from_path = str(path)
    s._allow_cache_id_mismatch = allow_cache_id_mismatch

    _logger.info(
        "loaded study from %s; saved target_model.cache_id=%r; saved paraphrase_model.cache_id=%r",
        path,
        saved_t_cid,
        saved_p_cid,
    )
    if target_model is not None and target_model.cache_id != saved_t_cid:
        _logger.info(
            "re-supplied target_model.cache_id=%r differs from saved %r; stage invocation will %s.",
            target_model.cache_id,
            saved_t_cid,
            "WARN only" if allow_cache_id_mismatch else "raise ConfigError",
        )
    if paraphrase_model is not None and paraphrase_model.cache_id != saved_p_cid:
        _logger.info(
            "re-supplied paraphrase_model.cache_id=%r differs from saved %r; "
            "stage invocation will %s.",
            paraphrase_model.cache_id,
            saved_p_cid,
            "WARN only" if allow_cache_id_mismatch else "raise ConfigError",
        )
    return s


def _study_reextract(self, extract_label) -> None:
    if self.mode != "free_form":
        raise ConfigError("reextract() is only valid in free-form mode.")
    if self._inference_df is None:
        raise StageNotRunError("reextract requires run_inference() output; load a study first.")
    self.extract_label = extract_label
    rows = []
    n_none = 0
    n_raised = 0
    for _, row in self._inference_df.iterrows():
        labels = []
        ok = True
        for cond in ("orig", "target", "base"):
            gen = row[f"generation_{cond}"]
            try:
                lab = extract_label(gen)
            except Exception as e:
                _logger.warning(
                    "reextract: extract_label raised %s on sample_id=%r/%s: %s",
                    type(e).__name__,
                    row.get("sample_id", "?"),
                    cond,
                    str(e)[:200],
                )
                n_raised += 1
                ok = False
                break
            if lab is None:
                n_none += 1
                ok = False
                break
            labels.append(lab)
        if not ok:
            continue
        new_row = dict(row)
        new_row["label_orig"], new_row["label_target"], new_row["label_base"] = labels
        rows.append(new_row)
    self._drop_counts["extraction_returned_none"] = n_none
    self._drop_counts["extraction_raised"] = n_raised
    self._inference_df = pd.DataFrame(rows)


Study.save = _study_save
Study.load = _study_load
Study.reextract = _study_reextract
