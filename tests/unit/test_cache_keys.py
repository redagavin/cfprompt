import pytest

from cfprompt.cache import inference_cache_key, paraphrase_cache_key


@pytest.mark.unit
class TestParaphraseCacheKey:
    def test_deterministic(self):
        kw = dict(
            stage_version="1",
            cfprompt_version="0.0.1",
            original="hello",
            target_perturbed="HELLO",
            target_edit_pct=10.0,
            paraphrase_model_cache_id="openai:gpt-4|fp_x",
            tokenizer_cache_id="hf:foo@abc",
            tolerance=0.5,
            max_retries=50,
            seed=12345,
        )
        a = paraphrase_cache_key(**kw)
        b = paraphrase_cache_key(**kw)
        assert a == b
        assert isinstance(a, str)
        assert len(a) == 64

    def test_target_perturbed_in_key_prevents_collision(self):
        kw = dict(
            stage_version="1",
            cfprompt_version="0.0.1",
            original="hello",
            target_edit_pct=10.0,
            paraphrase_model_cache_id="openai:gpt-4|fp_x",
            tokenizer_cache_id="hf:foo@abc",
            tolerance=0.5,
            max_retries=50,
            seed=12345,
        )
        a = paraphrase_cache_key(**kw, target_perturbed="HELLO")
        b = paraphrase_cache_key(**kw, target_perturbed="hElLo")
        assert a != b

    def test_seed_changes_key(self):
        kw = dict(
            stage_version="1",
            cfprompt_version="0.0.1",
            original="hello",
            target_perturbed="HELLO",
            target_edit_pct=10.0,
            paraphrase_model_cache_id="x",
            tokenizer_cache_id="y",
            tolerance=0.5,
            max_retries=50,
        )
        assert paraphrase_cache_key(**kw, seed=1) != paraphrase_cache_key(**kw, seed=2)


@pytest.mark.unit
class TestInferenceCacheKey:
    def test_classes_user_order_preserved(self):
        kw = dict(
            stage_version="1",
            prompt="Q?",
            target_model_cache_id="hf:x@a",
            mode="classification",
            seed=7,
        )
        a = inference_cache_key(**kw, classes=["A", "B"])
        b = inference_cache_key(**kw, classes=["B", "A"])
        assert a != b

    def test_free_form_no_classes_field(self):
        a = inference_cache_key(
            stage_version="1",
            prompt="Q?",
            target_model_cache_id="hf:x@a",
            mode="free_form",
            classes=None,
            seed=7,
        )
        assert isinstance(a, str)
        assert len(a) == 64
