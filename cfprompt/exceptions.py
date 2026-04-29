"""Exception hierarchy for cfprompt."""


class CfpromptError(Exception):
    """Base class for all cfprompt errors."""


class ConfigError(CfpromptError):
    """Mode mismatch, missing kwargs, bad columns, cache_id mismatch."""


class ClassificationModeError(CfpromptError):
    """OpenAI multi-token / >20 classes; HFModel first-token collisions."""


class StageNotRunError(CfpromptError):
    """Test() called before run_inference(); load() then stage call without
    re-supplied inputs."""


class DegenerateMetricError(CfpromptError):
    """Raised by metric functions (e.g., phi_coefficient) when input is
    degenerate (e.g., zero marginal). Caught and counted by bootstrap;
    propagates to user only when the observed (un-resampled) data is
    degenerate."""
