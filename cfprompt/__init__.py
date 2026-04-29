"""cfprompt — counterfactual prompting with adjusted paraphrase baselines."""

__version__ = "0.0.1"

# Bumped whenever paraphrase prompt template, refusal phrases, edit-distance
# logic, or select-best-undershoot semantics change in a way that affects
# cached paraphrase outputs.
__paraphrase_algorithm_version__ = "1"

# Bumped whenever prompt formatting (safe_format), score_classes
# normalization, or any code path that changes cached probability/generation
# values changes.
__inference_algorithm_version__ = "1"

import logging

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
]
