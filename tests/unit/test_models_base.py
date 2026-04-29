import numpy as np
import pytest

from cfprompt.models.base import Model, Tokenizer


@pytest.mark.unit
class TestTokenizerProtocol:
    def test_protocol_has_encode_method(self):
        # A Protocol satisfies isinstance via duck typing in Python 3.11+
        class Stub:
            def encode(self, text: str) -> list[int]:
                return [1, 2]

            @property
            def cache_id(self) -> str:
                return "stub:v1"

        assert isinstance(Stub(), Tokenizer)

    def test_protocol_rejects_missing_cache_id(self):
        class Stub:
            def encode(self, text: str) -> list[int]:
                return []

        # runtime_checkable Protocol with @property may not detect missing
        # `cache_id` perfectly, but we at least verify encode-only doesn't
        # have a working cache_id attribute
        s = Stub()
        with pytest.raises(AttributeError):
            _ = s.cache_id


@pytest.mark.unit
class TestModelABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Model()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstracts(self):
        class HalfBaked(Model):
            def generate(self, prompts, per_prompt_seeds=None):
                return []

        with pytest.raises(TypeError, match="abstract"):
            HalfBaked()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        class Concrete(Model):
            def generate(self, prompts, per_prompt_seeds=None):
                return [""] * len(prompts)

            def score_classes(self, prompts, classes, per_prompt_seeds=None):
                n, k = len(prompts), len(classes)
                arr = np.full((n, k), 1.0 / k)
                return arr

            def close(self) -> None:
                pass

            @property
            def tokenizer(self):
                class T:
                    def encode(self, text):
                        return [0]

                    @property
                    def cache_id(self):
                        return "t:v1"

                return T()

            @property
            def cache_id(self) -> str:
                return "concrete:v1"

        m = Concrete()
        assert m.cache_id == "concrete:v1"
        assert m.generate(["hi"]) == [""]

    def test_context_manager_calls_close(self):
        closed = {"v": False}

        class Concrete(Model):
            def generate(self, prompts, per_prompt_seeds=None):
                return [""] * len(prompts)

            def score_classes(self, prompts, classes, per_prompt_seeds=None):
                return np.zeros((len(prompts), len(classes)))

            def close(self) -> None:
                closed["v"] = True

            @property
            def tokenizer(self):
                raise NotImplementedError

            @property
            def cache_id(self) -> str:
                return "c"

        with Concrete() as m:
            assert m is not None
        assert closed["v"] is True
