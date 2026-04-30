"""Laptop-side analysis of a saved Study — no GPU, no model required.

Demonstrates: Study.load(path) with no kwargs → test() works on cached
inference_df → write Excel.

Run after a previous run that produced study.pkl.
"""

from pathlib import Path

from cfprompt.study import Study


def main(path: Path = Path("study.pkl")) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found; run an earlier example first")
    study = Study.load(path)
    report = study.test(metrics=["flip_rate", "mi"])
    print(report.summary_table().to_string(index=False))
    report.to_excel("results.xlsx")
    print("Wrote results.xlsx")


if __name__ == "__main__":
    main()
