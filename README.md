# cfprompt

Counterfactual prompting with adjusted paraphrase baselines.

Companion package to *"Compared to What? Baselines and Metrics for Counterfactual Prompting."*

## Install

```bash
pip install cfprompt   # PyPI release pending; for now:
pip install -e .       # from source clone
```

## Quickstart

```python
import pandas as pd
import cfprompt

df = pd.DataFrame({"question": [...]})

study = cfprompt.Study(
    data=df,
    perturb_column="question",
    target_perturbation=my_swap_function,
    prompt_template="Q: {question}\nA:",
    target_model=cfprompt.HFModel("hf-internal-testing/tiny-random-LlamaForCausalLM"),
    paraphrase_model=cfprompt.OpenAIModel("gpt-4.1"),
    classes=["A", "B", "C", "D"],
    seed=42,
)
report = study.run_all(metrics=["jsd", "flip_rate"])
report.to_excel("results.xlsx")
```

See `examples/` for runnable scripts.

## CLI

```bash
cfprompt init my_study             # scaffold a starter directory
cfprompt validate config.yaml      # schema check
cfprompt run config.yaml           # execute the study
```

## License

MIT.
