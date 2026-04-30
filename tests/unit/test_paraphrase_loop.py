import pytest

from cfprompt.paraphrase import (
    AdjustedParaphraseResult,
    generate_adjusted_paraphrase,
)


class _FakeTokenizer:
    """Whitespace tokenizer returning integer ids by hashing words."""

    def encode(self, text: str) -> list[int]:
        return [hash(w) & 0xFFFF for w in text.split()]

    @property
    def cache_id(self) -> str:
        return "fake:v1"


class _ScriptedModel:
    """Returns canned generations in sequence."""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)
        self.calls = 0

    def generate(self, prompts, per_prompt_seeds=None):
        out = []
        for _ in prompts:
            self.calls += 1
            if not self._scripted:
                out.append("")
            else:
                out.append(self._scripted.pop(0))
        return out


@pytest.mark.unit
class TestGenerateAdjustedParaphrase:
    def test_within_tolerance_first_attempt(self):
        tok = _FakeTokenizer()
        # Original = 10 tokens; produce a paraphrase with 1 token change = 10%.
        original = " ".join(f"w{i}" for i in range(10))
        modified = " ".join(["X"] + [f"w{i}" for i in range(1, 10)])
        model = _ScriptedModel([modified])
        result = generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=10.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=50,
        )
        assert isinstance(result, AdjustedParaphraseResult)
        assert result.refused is False
        assert result.paraphrase == modified
        assert abs(result.actual_edit_pct - 10.0) <= 0.5
        assert model.calls == 1
        # retries_used reports total attempts; first-attempt success → 1.
        assert result.retries_used == 1

    def test_retries_used_counts_all_attempts_on_full_refusal(self):
        """With max_retries=2 the loop runs 3 attempts; if all refuse,
        retries_used must be 3 (not 2)."""
        tok = _FakeTokenizer()
        original = " ".join(f"w{i}" for i in range(10))
        model = _ScriptedModel(["I can't help with that."] * 3)
        result = generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=10.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=2,
        )
        assert result.refused is True
        assert result.retries_used == 3

    def test_refusal_then_success(self):
        tok = _FakeTokenizer()
        original = " ".join(f"w{i}" for i in range(10))
        modified = " ".join(["X"] + [f"w{i}" for i in range(1, 10)])
        model = _ScriptedModel(["I can't help with that.", modified])
        result = generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=10.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=50,
        )
        assert result.refused is False
        assert result.paraphrase == modified

    def test_all_refusals_returns_original_with_refused_flag(self):
        tok = _FakeTokenizer()
        original = " ".join(f"w{i}" for i in range(10))
        # max_retries=2 means 3 attempts total
        model = _ScriptedModel(["I can't help with that."] * 3)
        result = generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=10.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=2,
        )
        assert result.refused is True
        assert result.paraphrase == original
        assert result.actual_edit_pct == 0.0

    def test_seed_propagates_to_paraphrase_model(self):
        """When seed= is passed, paraphrase_model.generate() must receive it
        in per_prompt_seeds. Same seed reused across retries (deterministic)."""
        tok = _FakeTokenizer()
        original = " ".join(f"w{i}" for i in range(10))
        modified = " ".join(["X"] + [f"w{i}" for i in range(1, 10)])

        class _RecordingModel:
            def __init__(self, scripted):
                self._scripted = list(scripted)
                self.seeds_seen: list = []

            def generate(self, prompts, per_prompt_seeds=None):
                self.seeds_seen.append(per_prompt_seeds)
                return [self._scripted.pop(0)]

        # Refusal then success → at least 2 attempts, seed must persist.
        model = _RecordingModel(["I can't help with that.", modified])
        generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=10.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=5,
            seed=12345,
        )
        assert model.seeds_seen == [[12345], [12345]]

    def test_no_seed_omits_per_prompt_seeds_kwarg(self):
        """When seed=None (default), generate() is called without per_prompt_seeds."""
        tok = _FakeTokenizer()
        original = " ".join(f"w{i}" for i in range(10))
        modified = " ".join(["X"] + [f"w{i}" for i in range(1, 10)])

        class _RecordingModel:
            def __init__(self, scripted):
                self._scripted = list(scripted)
                self.kwargs_seen: list = []

            def generate(self, prompts, **kwargs):
                self.kwargs_seen.append(kwargs)
                return [self._scripted.pop(0)]

        model = _RecordingModel([modified])
        generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=10.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=0,
        )
        # No seed passed → no per_prompt_seeds in the call kwargs.
        assert model.kwargs_seen == [{}]

    def test_overshoots_picks_closest_undershoot(self):
        tok = _FakeTokenizer()
        original = " ".join(f"w{i}" for i in range(10))
        # Two attempts, both outside tolerance:
        #   undershoot: 1 edit = 10% (target was 20%, dev=10)
        #   overshoot: 4 edits = 40% (dev=20)
        under = " ".join(["X"] + [f"w{i}" for i in range(1, 10)])  # 1 edit
        over = " ".join(["X1", "X2", "X3", "X4"] + [f"w{i}" for i in range(4, 10)])  # 4 edits
        model = _ScriptedModel([under, over, under, over])
        result = generate_adjusted_paraphrase(
            text=original,
            target_edit_pct=20.0,
            paraphrase_model=model,
            tokenizer=tok,
            tolerance=0.5,
            max_retries=1,
        )
        # Best undershoot is `under` at 10%.
        assert result.paraphrase == under
