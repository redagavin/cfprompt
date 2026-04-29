from pathlib import Path

import pytest
from typer.testing import CliRunner

from cfprompt.cli.main import app

runner = CliRunner()


@pytest.mark.integration
class TestCfpromptInit:
    def test_init_classification_creates_files(self, tmp_path: Path):
        target = tmp_path / "study"
        result = runner.invoke(app, ["init", str(target), "--mode", "classification"])
        assert result.exit_code == 0
        assert (target / "config.yaml").exists()
        assert (target / "perturbations.py").exists()
        assert (target / "README.md").exists()
        assert (target / "data").is_dir()

    def test_init_freeform_uses_freeform_template(self, tmp_path: Path):
        target = tmp_path / "study"
        result = runner.invoke(app, ["init", str(target), "--mode", "freeform"])
        assert result.exit_code == 0
        cfg = (target / "config.yaml").read_text()
        assert "extract_label" in cfg
