from pathlib import Path

import pytest
from typer.testing import CliRunner

from cfprompt.cli.main import app

runner = CliRunner()


@pytest.mark.integration
class TestCfpromptRun:
    def test_dry_run_succeeds(self, tmp_path: Path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("""
data: data.csv
perturb_column: q
target_perturbation: pert:fn
prompt_template: "{q}"
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
""")
        result = runner.invoke(app, ["run", str(cfg), "--dry-run"])
        assert result.exit_code == 0
