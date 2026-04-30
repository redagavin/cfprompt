from pathlib import Path

import pytest
from typer.testing import CliRunner

from cfprompt.cli.main import app

runner = CliRunner()


def _classification_yaml_text(target_perturbation: str = "os.path:join") -> str:
    return f"""
data: data.csv
perturb_column: q
target_perturbation: {target_perturbation}
prompt_template: "{{q}}"
target_model:
  type: HFModel
  name_or_path: hf-internal-testing/tiny-random-LlamaForCausalLM
paraphrase_model:
  type: OpenAIModel
  name: gpt-4.1
classes: [A, B]
metrics: [flip_rate]
seed: 42
n_bootstrap: 100
output: out.xlsx
"""


@pytest.mark.integration
class TestCfpromptRun:
    def test_dry_run_succeeds(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text())
        result = runner.invoke(app, ["run", str(cfg), "--dry-run"])
        assert result.exit_code == 0

    def test_dry_run_unresolvable_callable_exits_1(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text("nonexistent_module_xyz:fn"))
        result = runner.invoke(app, ["run", str(cfg), "--dry-run"])
        assert result.exit_code == 1
        assert "callable resolution failed" in result.output

    def test_invalid_log_level_exits_1(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text())
        result = runner.invoke(
            app, ["run", str(cfg), "--dry-run", "--log-level", "FOO"]
        )
        assert result.exit_code == 1
        assert "unknown log level" in result.output
