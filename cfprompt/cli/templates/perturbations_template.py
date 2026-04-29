"""Define your target perturbation here.

Reference this from your YAML config as:
  target_perturbation: perturbations:my_perturbation
"""


def my_perturbation(text: str) -> str:
    """Replace this stub with the perturbation you want to study."""
    swaps = {"he": "she", "him": "her", "his": "hers", "she": "he", "her": "him"}
    out_words = []
    for w in text.split():
        out_words.append(swaps.get(w.lower(), w))
    return " ".join(out_words)


def extract_letter(response: str) -> str | None:
    """Free-form extractor stub. Return None if the response can't be parsed."""
    import re

    match = re.search(r"\b([A-D])\b", response)
    return match.group(1) if match else None
