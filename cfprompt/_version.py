"""Package version constants.

Lives in its own submodule so cfprompt.study (and other submodules) can import
the constants without triggering the import cycle that arises when re-exports
in cfprompt/__init__.py run before the constants are bound.
"""

__version__ = "0.0.1"

# Bumped whenever paraphrase prompt template, refusal phrases, edit-distance
# logic, or select-best-undershoot semantics change in a way that affects
# cached paraphrase outputs.
__paraphrase_algorithm_version__ = "1"

# Bumped whenever prompt formatting (safe_format), score_classes
# normalization, or any code path that changes cached probability/generation
# values changes.
__inference_algorithm_version__ = "1"
