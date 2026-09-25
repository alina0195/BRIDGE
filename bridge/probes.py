"""Residual-stream activations and the linear probes read by BRIDGe.

Three directions are used in the paper:

    volatility probe   logistic probe, stable vs volatile, at several layers
                       (Section 3.2 diagnosis, probe-survival test)
    presence probe     logistic probe, no context vs grounded context; its
                       unit weight vector is c_hat, the steering direction
    v_hat              difference of class means, stable vs volatile, fitted
                       at the hook layer on the exact prompts the hook sees;
                       gates BRIDGe and sets the dose unit gap = ||mu_v - mu_s||
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bridge.model import chat_input_ids, get_layers, input_device


class ResidualStreamExtractor:
    """Captures block outputs at chosen layers for a single text input."""

    def __init__(self, model, tokenizer, target_layers: list[int] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        n = len(get_layers(model))
        layers = list(range(n)) if target_layers is None else target_layers
        self.target_layers = [l if l >= 0 else n + l for l in layers]

    @torch.inference_mode()
    def extract_at_positions(self, text: str, token_positions: list[int] | None = None
                             ) -> dict[int, torch.Tensor]:
        """Hidden states at token_positions (default: last token) per layer."""
        captured: dict[int, torch.Tensor] = {}
        handles = []
        layers = get_layers(self.model)
        for idx in self.target_layers:
            def hook(module, inputs, output, idx=idx):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden.squeeze(0).detach().cpu()
            handles.append(layers[idx].register_forward_hook(hook))
        try:
            ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=2048)["input_ids"]
            self.model(input_ids=ids.to(input_device(self.model)))
        finally:
            for h in handles:
                h.remove()
        seq_len = ids.shape[1]
        pos = [seq_len - 1] if token_positions is None else \
            [p if p >= 0 else seq_len + p for p in token_positions]
        return {l: s[pos] for l, s in captured.items()}


@torch.inference_mode()
def collect_hook_activations(model, tokenizer, prompts: list[str], layer: int) -> np.ndarray:
    """Last-prompt-token output of block ``layer`` under the chat template.

    Same module, token position and tokenisation as the steering hook at
    evaluation time, so projections, gap and thresholds are on its scale.
    """
    captured = {}

    def tap(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden[0, -1, :].detach().float().cpu()

    handle = get_layers(model)[layer].register_forward_hook(tap)
    out = []
    try:
        for i, prompt in enumerate(prompts):
            model(input_ids=chat_input_ids(tokenizer, prompt).to(input_device(model)))
            out.append(captured["h"].numpy())
            if (i + 1) % 100 == 0 or i + 1 == len(prompts):
                print(f"  [{i + 1}/{len(prompts)}]", flush=True)
    finally:
        handle.remove()
    return np.stack(out)


def fit_logistic_probe(acts: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Balanced logistic probe with 5-fold CV; returns the unit weight vector."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, acts, labels, cv=cv)
    clf.fit(acts, labels)
    direction = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    return {
        "cv_accuracy": float(balanced_accuracy_score(labels, preds)),
        "cv_f1_macro": float(f1_score(labels, preds, average="macro")),
        "direction": direction,
        "intercept": float(clf.intercept_[0]),
    }


def fit_mean_difference(acts: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """v_hat = (mu_1 - mu_0) / ||mu_1 - mu_0||, its gap and projection stats.

    labels: 0 = stable, 1 = volatile.
    """
    acts = acts.astype(np.float64)
    mu_s, mu_v = acts[labels == 0].mean(0), acts[labels == 1].mean(0)
    gap = float(np.linalg.norm(mu_v - mu_s))
    if gap < 1e-8:
        raise ValueError("class means coincide")
    unit = (mu_v - mu_s) / gap
    proj = acts @ unit
    p_s, p_v = proj[labels == 0], proj[labels == 1]
    return {
        "unit": unit,
        "gap": gap,
        "tau_v_midpoint": float((p_s.mean() + p_v.mean()) / 2),
        "mean_proj_stable": float(p_s.mean()),
        "mean_proj_volatile": float(p_v.mean()),
        "mean_activation_norm": float(np.linalg.norm(acts, axis=1).mean()),
    }


def tau_at_stable_pass(stable_proj: np.ndarray, rate: float) -> float:
    """Threshold that lets ``rate`` of stable prompts pass (gate fires on > tau)."""
    return float(np.quantile(stable_proj, 1.0 - rate))
