"""cfprompt — counterfactual prompting with adjusted paraphrase baselines."""

import logging

from ._version import (
    __inference_algorithm_version__,
    __paraphrase_algorithm_version__,
    __version__,
)
from .exceptions import (
    CfpromptError,
    ClassificationModeError,
    ConfigError,
    DegenerateMetricError,
    StageNotRunError,
)
from .models.base import Model, Tokenizer
from .models.hf import HFModel, HFTokenizer
from .models.openai import OpenAIModel, TiktokenWrapper
from .report import Report, TestResult
from .study import Study

_logger = logging.getLogger("cfprompt")


def set_log_level(level: str | int) -> None:
    """Set the package-wide log level. Accepts strings ('DEBUG', 'INFO', ...)
    or integer levels."""
    _logger.setLevel(level)


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
