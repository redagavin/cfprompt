# Contributing to cfprompt

## Setup

```bash
conda create -n cfprompt python=3.11
conda activate cfprompt
pip install -e ".[dev]"
```

## Run tests

```bash
pytest                                # unit + integration
pytest -m unit                        # fast subset
pytest -m smoke                       # downloads small models
pytest --cov=cfprompt                 # with coverage
```

### Coverage exclusions

`cfprompt/cli/run.py` is excluded from coverage in `pyproject.toml` because exercising its full pipeline requires real models (HF or OpenAI). The integration tests cover individual pieces (`_resolve_callable`, `StudyConfig` validation, `init`, `validate`); end-to-end coverage of `run_command` itself is exercised manually via `cfprompt run` against tiny models. If you add new branches to `run.py`, add corresponding tests via the smoke marker (`pytest -m smoke`).

## Lint and format

```bash
ruff check cfprompt tests
ruff format cfprompt tests
```

## Versioning

Bump the appropriate version constant in `cfprompt/_version.py` whenever the corresponding behavior changes:

- `__version__` — package version (semver).
- `__paraphrase_algorithm_version__` — bump when paraphrase prompt, refusal phrases, edit-distance logic, or select-best-undershoot semantics change in a way that invalidates cached paraphrase outputs.
- `__inference_algorithm_version__` — bump when prompt formatting, `score_classes` normalization, or any code path that changes cached probabilities/generations changes.

These versions are part of the per-sample cache key, so bumping them invalidates the cache.

## Adding a new metric

1. Implement the metric function in `cfprompt/metrics/{distributional,label,regression}.py`.
2. Add it to the appropriate set in `cfprompt/study.py` (`_PER_SAMPLE_METRICS`, `_AGGREGATE_METRICS`, etc.).
3. Add unit tests in `tests/unit/test_metrics_*.py`.
4. Update README and `cfprompt/cli/templates/config_*.yaml`.

## Adding a new Model backend

1. Subclass `cfprompt.models.base.Model` (and `Tokenizer` Protocol).
2. Implement `generate`, `score_classes`, `tokenizer`, `cache_id`, `close`.
3. Register the type name in `cfprompt/cli/run.py:_build_model`.
4. Add integration tests under `tests/integration/`.

## Pull requests

- Keep commits focused; one logical change per commit.
- All tests must pass and ruff must be clean.
- Mention which version constant(s) you bumped, if any.
