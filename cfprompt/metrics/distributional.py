"""Per-sample distributional metrics: JSD, KL.

Both metrics clip probabilities at eps=1e-12 before taking logs to defend
against fp16 underflow (HFModel) and OpenAI's renormalization-after-
top_logprobs path.

Clip-count semantics (study.py reports ``clipped_kl_count``): the count
records ANY channel whose probability falls below EPS in EITHER condition
arm — including harmless clips that do not change the metric value (e.g.,
matched zeros across both arms, or zeros in p that are masked by the
0*log(0):=0 convention). The count is intentionally an over-estimate of
"rows whose KL value materially depended on the clip"; treat it as an
upper bound when diagnosing fp16 underflow.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def kl(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Per-sample KL(p1 || p2). Returns shape (n,).

    Clipped at EPS=1e-12 on both arrays before log to avoid +inf.
    Uses the 0*log(0):=0 convention via clipping.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p1c = np.clip(p1, EPS, 1.0)
    p2c = np.clip(p2, EPS, 1.0)
    log_ratio = np.log(p1c) - np.log(p2c)
    # 0 * log(0/x) := 0 — implement by zeroing the contribution where p1 == 0.
    contribution = np.where(p1 > 0, p1 * log_ratio, 0.0)
    return contribution.sum(axis=-1)


def jsd(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Per-sample Jensen-Shannon divergence. Returns shape (n,).

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q).
    Symmetric. JSD(p, p) = 0.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    m = 0.5 * (p1 + p2)
    return 0.5 * kl(p1, m) + 0.5 * kl(p2, m)
