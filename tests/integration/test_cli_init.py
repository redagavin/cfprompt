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

    def test_init_refuses_to_overwrite_existing_files(self, tmp_path: Path):
        target = tmp_path / "study"
        target.mkdir()
        (target / "config.yaml").write_text("existing: true\n")
        result = runner.invoke(app, ["init", str(target), "--mode", "classification"])
        assert result.exit_code == 1
        assert "refusing to overwrite" in result.output
        assert (target / "config.yaml").read_text() == "existing: true\n"

    def test_init_directional_appends_fields(self, tmp_path: Path):
        target = tmp_path / "study"
        result = runner.invoke(
            app,
            ["init", str(target), "--mode", "classification", "--directional"],
        )
        assert result.exit_code == 0
        cfg = (target / "config.yaml").read_text()
        assert "direction_column:" in cfg
        assert "outcome_class_column:" in cfg
        assert "alternative:" in cfg
