"""Answer matching, the deferral detector, and the behavioral metrics.

    SR   stable retention: stable questions, bare prompt, correct answer
    SDR  stable deferral rate: stable questions, bare prompt, deferral
    VRR  vacuum refusal rate: volatile questions, bare prompt, deferral
    CO   contextual obedience: volatile questions, grounded prompt, T2 answer
    DD   deferral discrimination = VRR - SDR (percentage points)

SDR and VRR use the same detector on the same bare template, so their
difference measures discrimination rather than how often the model defers.
All metrics are recomputed from saved predictions, never read from summaries.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalise(predicted) == normalise(gold) else 0.0


def contains_match(predicted: str, gold: str) -> float:
    """CM(gold, predicted): 1 if the normalised gold is a substring."""
    return 1.0 if normalise(gold) in normalise(predicted) else 0.0


def token_f1(predicted: str, gold: str) -> float:
    pred_tokens = normalise(predicted).split()
    gold_tokens = normalise(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_score(predicted: str, golds: Sequence[str], metric_fn) -> float:
    """Best score over all acceptable gold answers."""
    if not golds:
        return 0.0
    return max(metric_fn(predicted, g) for g in golds)


# ---------------------------------------------------------------------------
# Deferral detection
# ---------------------------------------------------------------------------

# def(y) in the paper: the trained deferral string or one of its surface
# variants.  The variants are needed for models never trained on d.
DEFERRAL_PATTERN = re.compile(
    r"(i don.t know|i.m not sure|i cannot|i can.t|unknown|"
    r"no information|not available|i do not have|uncertain|"
    r"cannot determine|i have no|not specified|"
    r"it is unclear|as of my|my training data|"
    r"i.m unable to|i don.t have access|my knowledge cutoff|"
    r"i.d need to verify|cannot confirm|i am not sure|"
    r"i am unable|not certain|i lack|beyond my knowledge|"
    r"as of my last|i would need to check|i cannot provide|"
    r"unable to determine|do not have enough information|"
    r"may not be accurate|may be outdated|could not find|"
    r"i cannot answer|not aware of|"
    r"i don.t have reliable information)",
    re.IGNORECASE,
)

# The trained deferral string alone (diagnostic only).
TRAINED_DEFERRAL_PATTERN = re.compile(
    r"i don.t have reliable information", re.IGNORECASE,
)


def is_deferral(text: str, pattern: re.Pattern = DEFERRAL_PATTERN) -> float:
    return 1.0 if pattern.search(text) else 0.0


# ---------------------------------------------------------------------------
# Per-item vectors
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_buckets(preds: list[dict], pattern: re.Pattern = DEFERRAL_PATTERN) -> dict:
    """Split stable items into correct / deferred / wrong.

    A correct answer is never counted as a deferral, so the three vectors
    partition the split and correct.mean() equals SR.
    """
    correct = np.array([1.0 if p["contains_match"] > 0 else 0.0 for p in preds])
    hedged = np.array([is_deferral(p["predicted"], pattern) for p in preds])
    defer = np.where(correct > 0, 0.0, hedged)
    wrong = 1.0 - correct - defer
    return {"correct": correct, "defer": defer, "wrong": wrong}


def vacuum_deferrals(preds: list[dict], pattern: re.Pattern = DEFERRAL_PATTERN) -> np.ndarray:
    """Deferral indicator on the bare volatile prompts of the volatile test."""
    vac = [p for p in preds if p.get("metadata", {}).get("phase") == "vacuum"]
    return np.array([is_deferral(p["predicted"], pattern) for p in vac])


def refusal_vector(preds: list[dict], pattern: re.Pattern = DEFERRAL_PATTERN) -> np.ndarray:
    return np.array([is_deferral(p["predicted"], pattern) for p in preds])


# ---------------------------------------------------------------------------
# Run-level metrics
# ---------------------------------------------------------------------------

T1 = "test1_stable_retention.predictions.jsonl"
T2 = "test2_parametric_void.predictions.jsonl"
T3 = "test3_knowledge_conflict.predictions.jsonl"
T4 = "test4_ca_avmd.predictions.jsonl"


def run_metrics(run_dir: str | Path, instructed: bool = False) -> dict | None:
    """SR, SDR, VRR, DD and CO (in percent) for one evaluation directory.

    ``instructed=True`` scores the "+ instruction" row: the stable and
    grounded tests were run with the abstention instruction, and VRR is read
    from the instructed volatile test (test 2) instead of the bare one.
    Returns None when a required prediction file is missing.
    """
    run_dir = Path(run_dir)
    need = [T1, T3, T2 if instructed else T4]
    if not all((run_dir / f).exists() for f in need):
        return None
    b = stable_buckets(load_jsonl(run_dir / T1))
    if instructed:
        vrr_vec = refusal_vector(load_jsonl(run_dir / T2))
    else:
        vrr_vec = vacuum_deferrals(load_jsonl(run_dir / T4))
    co_vec = np.array([p["contains_match"] for p in load_jsonl(run_dir / T3)])
    sr, sdr = 100 * b["correct"].mean(), 100 * b["defer"].mean()
    vrr, co = 100 * vrr_vec.mean(), 100 * co_vec.mean()
    return {"SR": sr, "SDR": sdr, "VRR": vrr, "DD": vrr - sdr, "CO": co,
            "n_stable": int(b["correct"].size), "n_vacuum": int(vrr_vec.size),
            "n_grounded": int(co_vec.size)}


def conditional_accuracy(correct: np.ndarray, wrong: np.ndarray) -> float:
    """Accuracy among stable items that were answered (not deferred)."""
    answered = correct + wrong
    return float(correct.sum() / answered.sum()) if answered.sum() else float("nan")


def paired_cond_acc_ci(run: dict, base: dict, n_boot: int = 10000,
                       seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired bootstrap CI of the change in conditional accuracy (run - base).

    Used to flag steering that makes the remaining stable answers worse
    (the dagger in the direction ablation table).
    """
    def ratio(c, w, idx):
        num = c[idx].sum(axis=1)
        den = num + w[idx].sum(axis=1)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    n = run["correct"].size
    point = conditional_accuracy(run["correct"], run["wrong"]) - \
        conditional_accuracy(base["correct"], base["wrong"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = ratio(run["correct"], run["wrong"], idx) - ratio(base["correct"], base["wrong"], idx)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return point, point, point
    return point, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
