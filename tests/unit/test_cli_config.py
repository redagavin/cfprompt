import pytest
import yaml
from pydantic import ValidationError

from cfprompt.cli.config import StudyConfig, load_yaml


@pytest.mark.unit
class TestStudyConfig:
    def _yaml(self, **overrides):
        cfg = {
            "data": "data/x.csv",
            "perturb_column": "q",
            "target_perturbation": "my_mod:swap_gender",
            "prompt_template": "Q: {q}\nA:",
            "target_model": {"type": "HFModel", "name_or_path": "x/y"},
            "paraphrase_model": {"type": "OpenAIModel", "name": "gpt-4.1"},
            "classes": ["A", "B"],
            "metrics": ["jsd"],
            "seed": 42,
            "n_bootstrap": 1000,
            "output": "out.xlsx",
        }
        cfg.update(overrides)
        return cfg

    def test_basic_classification_loads(self):
        cfg = StudyConfig.model_validate(self._yaml())
        assert cfg.classes == ["A", "B"]
        assert cfg.extract_label is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            StudyConfig.model_validate(self._yaml(typo_field=True))

    def test_classes_xor_extract_label_enforced_both(self):
        with pytest.raises(ValidationError):
            StudyConfig.model_validate(self._yaml(extract_label="my:fn"))

    def test_classes_xor_extract_label_enforced_neither(self):
        d = self._yaml()
        d.pop("classes")
        with pytest.raises(ValidationError):
            StudyConfig.model_validate(d)

    def test_directional_all_or_none(self):
        d = self._yaml(direction_column="dir")
        with pytest.raises(ValidationError):
            StudyConfig.model_validate(d)

    def test_safe_load_yaml_rejects_python_object(self, tmp_path):
        bad = tmp_path / "evil.yaml"
        bad.write_text('!!python/object/apply:os.system ["echo pwned"]')
        with pytest.raises(yaml.YAMLError):
            load_yaml(bad)

    def test_api_key_in_model_config_rejected(self):
        d = self._yaml(
            target_model={
                "type": "OpenAIModel",
                "name": "gpt-4.1",
                "api_key": "sk-leaked",
            }
        )
        with pytest.raises(ValidationError, match="secret kwargs"):
            StudyConfig.model_validate(d)

    def test_secret_kwarg_in_model_config_rejected(self):
        d = self._yaml(
            paraphrase_model={
                "type": "OpenAIModel",
                "name": "gpt-4.1",
                "secret": "hunter2",
            }
        )
        with pytest.raises(ValidationError, match="secret kwargs"):
            StudyConfig.model_validate(d)


@pytest.mark.unit
class TestResolveCallable:
    def test_resolves_simple_module_attr(self):
        from cfprompt.cli.imports import resolve_callable

        assert resolve_callable("os.path:join") is __import__("os").path.join

    def test_resolves_dotted_attribute_path(self):
        from collections import OrderedDict

        from cfprompt.cli.imports import resolve_callable

        resolved = resolve_callable("collections:OrderedDict.fromkeys")
        assert resolved == OrderedDict.fromkeys
        assert resolved(["a", "b"]) == OrderedDict.fromkeys(["a", "b"])

    def test_missing_colon_raises(self):
        from cfprompt.cli.imports import resolve_callable

        with pytest.raises(ValueError, match="module:function"):
            resolve_callable("os.path.join")

    def test_unknown_module_raises_import_error(self):
        from cfprompt.cli.imports import resolve_callable

        with pytest.raises(ImportError):
            resolve_callable("nonexistent_module_xyz:fn")

    def test_unknown_attribute_raises_attribute_error(self):
        from cfprompt.cli.imports import resolve_callable

        with pytest.raises(AttributeError):
            resolve_callable("os.path:nonexistent_function_xyz")
