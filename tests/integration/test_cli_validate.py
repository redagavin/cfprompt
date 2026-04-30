from pathlib import Path

import pytest
from typer.testing import CliRunner

from cfprompt.cli.main import app

runner = CliRunner()


def _classification_yaml_text(target_perturbation: str = "pert:fn") -> str:
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
class TestCfpromptValidate:
    def test_missing_file_exit_1(self, tmp_path: Path):
        result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code == 1

    def test_valid_config_returns_0(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text())
        result = runner.invoke(app, ["validate", str(cfg), "--no-import"])
        assert result.exit_code == 0

    def test_extra_field_exits_1(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("""
data: data.csv
perturb_column: q
target_perturbation: pert:fn
prompt_template: "{q}"
target_model: {type: HFModel, name_or_path: x}
paraphrase_model: {type: OpenAIModel, name: gpt-4.1}
classes: [A, B]
metrics: []
seed: 42
n_bootstrap: 1000
output: out.xlsx
typo_field: yes
""")
        result = runner.invoke(app, ["validate", str(cfg), "--no-import"])
        assert result.exit_code == 1

    def test_nonexistent_callable_without_no_import_exits_1(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text("nonexistent_module_xyz:fn"))
        result = runner.invoke(app, ["validate", str(cfg)])
        assert result.exit_code == 1
        assert "target_perturbation resolution failed" in result.output

    def test_nonexistent_callable_with_no_import_exits_0(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text("nonexistent_module_xyz:fn"))
        result = runner.invoke(app, ["validate", str(cfg), "--no-import"])
        assert result.exit_code == 0

    def test_real_callable_without_no_import_exits_0(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_classification_yaml_text("os.path:join"))
        result = runner.invoke(app, ["validate", str(cfg)])
        assert result.exit_code == 0
        assert "(resolved)" in result.output
