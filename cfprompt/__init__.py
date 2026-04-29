"""cfprompt — counterfactual prompting with adjusted paraphrase baselines."""

import logging

__version__ = "0.0.1"

# Bumped whenever paraphrase prompt template, refusal phrases, edit-distance
# logic, or select-best-undershoot semantics change in a way that affects
# cached paraphrase outputs.
__paraphrase_algorithm_version__ = "1"

# Bumped whenever prompt formatting (safe_format), score_classes
# normalization, or any code path that changes cached probability/generation
# values changes.
__inference_algorithm_version__ = "1"

_logger = logging.getLogger("cfprompt")


def set_log_level(level: str | int) -> None:
    """Set the package-wide log level. Accepts strings ('DEBUG', 'INFO', ...)
    or integer levels."""
    _logger.setLevel(level)


# Re-exports placed after version-constant definitions because submodules
# (e.g. cfprompt.study) import those constants at module load time.
from .exceptions import (  # noqa: E402
    CfpromptError,
    ClassificationModeError,
    ConfigError,
    DegenerateMetricError,
    StageNotRunError,
)
from .models.base import Model, Tokenizer  # noqa: E402
from .models.hf import HFModel, HFTokenizer  # noqa: E402
from .models.openai import OpenAIModel, TiktokenWrapper  # noqa: E402
from .report import Report, TestResult  # noqa: E402
from .study import Study  # noqa: E402

__all__ = [
    "__version__",
    "__paraphrase_algorithm_version__",
    "__inference_algorithm_version__",
    "set_log_level",
    "Study",
    "Report",
    "TestResult",
    "Model",
    "Tokenizer",
    "HFModel",
    "HFTokenizer",
    "OpenAIModel",
    "TiktokenWrapper",
    "CfpromptError",
    "ConfigError",
    "ClassificationModeError",
    "DegenerateMetricError",
    "StageNotRunError",
]
