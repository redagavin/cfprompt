import stat
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cfprompt.exceptions import ConfigError
from cfprompt.study import Study


def _stub_model(cache_id: str = "stub:1"):
    m = MagicMock()
    m.cache_id = cache_id

    class _T:
        def encode(self, t):
            return [1, 2]

        @property
        def cache_id(self):
            return "tok:1"

    m.tokenizer = _T()
    return m


@pytest.mark.unit
class TestStudyInit:
    def _df(self):
        return pd.DataFrame({"q": ["a", "b"], "outcome": ["A", "B"], "dir": [1, -1]})

    def test_classification_mode_constructs(self):
        s = Study(
            data=self._df(),
            perturb_column="q",
            target_perturbation=lambda x: x.upper(),
            prompt_template="Q: {q}\nA:",
            target_model=_stub_model(),
            paraphrase_model=_stub_model("para:1"),
            classes=["A", "B"],
        )
        assert s.mode == "classification"

    def test_freeform_mode_constructs(self):
        s = Study(
            data=self._df(),
            perturb_column="q",
            target_perturbation=lambda x: x.upper(),
            prompt_template="Q: {q}\nA:",
            target_model=_stub_model(),
            paraphrase_model=_stub_model("para:1"),
            extract_label=lambda r: r,
        )
        assert s.mode == "free_form"

    def test_both_modes_set_raises(self):
        with pytest.raises(ConfigError, match="mutually exclusive"):
            Study(
                data=self._df(),
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
                extract_label=lambda r: r,
            )

    def test_neither_mode_raises(self):
        with pytest.raises(ConfigError, match="Mode required"):
            Study(
                data=self._df(),
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
            )

    def test_perturb_column_missing_raises(self):
        with pytest.raises(ConfigError, match="not found in data"):
            Study(
                data=self._df(),
                perturb_column="nonexistent",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
            )

    def test_partial_directional_kwargs_raises(self):
        with pytest.raises(ConfigError, match="all of"):
            Study(
                data=self._df(),
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
                direction_column="dir",
            )

    def test_direction_column_with_zero_raises(self):
        df = pd.DataFrame({"q": ["a", "b"], "outcome": ["A", "B"], "dir": [1, 0]})
        with pytest.raises(ConfigError, match="invalid values"):
            Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
                direction_column="dir",
                outcome_class_column="outcome",
                alternative="greater",
            )

    def test_unwritable_cache_dir_raises_config_error(self, tmp_path: Path):
        """A read-only cache_dir parent must surface as ConfigError, not raw
        OSError/PermissionError, so users get an actionable message."""
        readonly_parent = tmp_path / "readonly"
        readonly_parent.mkdir()
        # Strip write permission.
        readonly_parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            target_dir = readonly_parent / "cache"
            with pytest.raises(ConfigError, match=r"not writable"):
                Study(
                    data=self._df(),
                    perturb_column="q",
                    target_perturbation=lambda x: x.upper(),
                    prompt_template="Q: {q}\nA:",
                    target_model=_stub_model(),
                    paraphrase_model=_stub_model("p"),
                    classes=["A", "B"],
                    cache_dir=target_dir,
                )
        finally:
            # Restore permissions so tmp_path cleanup works.
            readonly_parent.chmod(stat.S_IRWXU)

    def test_reserved_column_name_raises(self):
        """User-supplied data containing a column name cfprompt writes to
        (e.g. 'label_orig', 'sample_id') would silently overwrite the
        package's output."""
        df = pd.DataFrame(
            {
                "q": ["a", "b"],
                "outcome": ["A", "B"],
                "label_orig": ["X", "Y"],  # reserved
            }
        )
        with pytest.raises(ConfigError, match=r"reserved names"):
            Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
            )

    def test_duplicate_data_index_raises(self):
        """Duplicate index values would silently collide in the per-sample
        cache key, corrupting cached paraphrase/inference results."""
        df = pd.DataFrame(
            {"q": ["a", "b", "c"], "outcome": ["A", "B", "A"]},
            index=[0, 1, 0],
        )
        with pytest.raises(ConfigError, match=r"duplicate"):
            Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
            )

    def test_outcome_class_value_not_in_classes_raises(self):
        df = pd.DataFrame({"q": ["a"], "outcome": ["Z"], "dir": [1]})
        with pytest.raises(ConfigError, match="invalid values"):
            Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA:",
                target_model=_stub_model(),
                paraphrase_model=_stub_model("p"),
                classes=["A", "B"],
                direction_column="dir",
                outcome_class_column="outcome",
                alternative="greater",
            )
