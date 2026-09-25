"""The BRIDGe hook: conditional activation steering with a dual gate.

At the last prompt token of the hook layer, with unit directions v_hat
(volatility) and c_hat (context presence):

    h' = h - alpha * c_hat * 1[ v_hat.h > tau_v  and  c_hat.h < tau_c ]

The hook fires during prefill only (autoregressive steps are left alone), so
the intervention is one edit of the prompt representation per query.

Gate modes, used by the ablations:

    full           volatility AND no-context   (BRIDGe)
    presence_only  no-context only             (the "+ presence" baseline)
    volatile_only  volatility only
    none           every prompt (ungated steering)

The injected direction defaults to -c_hat.  The direction ablation replaces it
with +v_hat, a shuffled-label v_hat or a random unit vector, at a magnitude
expressed as k * gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bridge.model import get_layers, resolve_layer

GATES = ("full", "presence_only", "volatile_only", "none")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("zero-norm direction")
    return v / n


class BridgeHook:
    def __init__(self, volatile_dir: np.ndarray, presence_dir: np.ndarray,
                 magnitude: float, tau_v: float = 0.0, tau_c: float = 0.0,
                 gate: str = "full", steer_dir: np.ndarray | None = None):
        if gate not in GATES:
            raise ValueError(f"unknown gate {gate!r}")
        self.v_hat = _unit(volatile_dir)
        self.c_hat = _unit(presence_dir)
        self.steer = _unit(steer_dir) if steer_dir is not None else -self.c_hat
        self.magnitude = float(magnitude)
        self.tau_v, self.tau_c, self.gate = tau_v, tau_c, gate
        self._handle: Any = None
        self.record_scores = False       # collect presence scores for fire-rate matching
        self.c_scores: list[float] = []
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {"calls": 0, "steered": 0, "not_steered": 0}
        self.v_log: list[float] = []
        self.c_log: list[float] = []

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.shape[1] <= 1:          # autoregressive step
            return None
        self.stats["calls"] += 1
        kw = {"dtype": hidden.dtype, "device": hidden.device}
        v_dir = torch.tensor(self.v_hat, **kw)
        c_dir = torch.tensor(self.c_hat, **kw)
        s_dir = torch.tensor(self.steer, **kw)
        hidden = hidden.clone()
        for b in range(hidden.shape[0]):
            h = hidden[b, -1, :]
            v_score, c_score = float(h @ v_dir), float(h @ c_dir)
            self.v_log.append(v_score)
            self.c_log.append(c_score)
            if self.record_scores:
                self.c_scores.append(c_score)
            volatile, no_context = v_score > self.tau_v, c_score < self.tau_c
            fire = {"full": volatile and no_context, "presence_only": no_context,
                    "volatile_only": volatile, "none": True}[self.gate]
            if fire:
                hidden[b, -1, :] = h + self.magnitude * s_dir
                self.stats["steered"] += 1
            else:
                self.stats["not_steered"] += 1
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    def register(self, model, layer: int) -> int:
        idx = resolve_layer(model, layer)
        self._handle = get_layers(model)[idx].register_forward_hook(self)
        print(f"[hook] BridgeHook on layer {idx}: magnitude={self.magnitude:.4f} "
              f"tau_v={self.tau_v:.4f} tau_c={self.tau_c:.4f} gate={self.gate}")
        return idx

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def fire_rate(self) -> float:
        n = self.stats["steered"] + self.stats["not_steered"]
        return self.stats["steered"] / n if n else 0.0

    def snapshot(self, with_scores: bool = False) -> dict:
        out = {**self.stats, "fire_rate": self.fire_rate()}
        if with_scores:
            out["v_scores"] = list(self.v_log)
            out["c_scores"] = list(self.c_log)
        return out
