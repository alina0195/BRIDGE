"""Transformer sub-module accessors and component-ablation hooks.

Shared by activation_patching.py, targeted_ablation.py and probe_survival.py.
All interventions act on the last prompt token of the prefill pass.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from bridge.model import get_layers, input_device
from bridge.prompts import BARE_QA


def get_attn(layer):
    for attr in ("self_attn", "attn", "attention"):
        sub = getattr(layer, attr, None)
        if sub is not None:
            return sub
    raise AttributeError("Cannot find the attention module.")


def get_mlp(layer):
    for attr in ("mlp", "feed_forward", "ffn"):
        sub = getattr(layer, attr, None)
        if sub is not None:
            return sub
    raise AttributeError("Cannot find the MLP module.")


def get_o_proj(attn):
    for attr in ("o_proj", "out_proj", "dense"):
        sub = getattr(attn, attr, None)
        if sub is not None:
            return sub
    raise AttributeError("Cannot find o_proj in the attention module.")


def head_dim_of(model) -> int:
    return int(model.config.hidden_size) // int(model.config.num_attention_heads)


def bare_input_ids(tokenizer, question: str, device) -> torch.Tensor:
    """Bare prompt without chat template, as used by the probes and patching."""
    prompt = BARE_QA.format(question=question)
    return tokenizer(prompt, return_tensors="pt", truncation=True,
                     max_length=2048)["input_ids"].to(device)


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """'L:H,L:H' -> [(L, H), ...]"""
    out = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if tok:
            l, h = tok.split(":")
            out.append((int(l), int(h)))
    return out


def parse_mlps(spec: str) -> list[int]:
    """'L,L,L' -> [L, ...]"""
    return [int(x.strip()) for x in (spec or "").split(",") if x.strip()]


@torch.inference_mode()
def collect_component_means(model, tokenizer, stable_records, heads, mlps,
                            n_samples: int, head_dim: int):
    """Mean o_proj input slice per target head and mean MLP output per target
    layer at the last token of the bare prompt, over n_samples stable questions.
    """
    device = input_device(model)
    layers = get_layers(model)
    head_layers = sorted({l for l, _ in heads})
    capture: dict[str, torch.Tensor] = {}
    handles = []

    def o_pre_hook(layer_idx):
        def hook(module, inp):
            x = inp[0]
            if x.shape[1] > 1:
                capture[f"opre_{layer_idx}"] = x[0, -1, :].detach().float().cpu()
        return hook

    def mlp_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                capture[f"mlp_{layer_idx}"] = h[0, -1, :].detach().float().cpu()
        return hook

    for l in head_layers:
        handles.append(get_o_proj(get_attn(layers[l])).register_forward_pre_hook(o_pre_hook(l)))
    for l in set(mlps):
        handles.append(get_mlp(layers[l]).register_forward_hook(mlp_hook(l)))

    head_sum: dict[tuple[int, int], torch.Tensor] = {}
    mlp_sum: dict[int, torch.Tensor] = {}
    n_seen = 0
    n_total = min(n_samples, len(stable_records))
    try:
        t0 = time.time()
        for i, rec in enumerate(stable_records[:n_samples]):
            capture.clear()
            model(input_ids=bare_input_ids(tokenizer, rec["question"], device))
            for (l, h) in heads:
                vec = capture.get(f"opre_{l}")
                if vec is not None:
                    s = vec[h * head_dim:(h + 1) * head_dim]
                    head_sum[(l, h)] = head_sum.get((l, h), torch.zeros_like(s)) + s
            for l in set(mlps):
                vec = capture.get(f"mlp_{l}")
                if vec is not None:
                    mlp_sum[l] = mlp_sum.get(l, torch.zeros_like(vec)) + vec
            n_seen += 1
            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                print(f"  [means] {i + 1}/{n_samples} ({time.time() - t0:.0f}s)")
    finally:
        for h in handles:
            h.remove()

    if n_seen == 0:
        raise RuntimeError("No samples processed for mean collection.")
    print(f"[means] averaged over {n_seen} stable samples")
    return ({k: v / n_seen for k, v in head_sum.items()},
            {k: v / n_seen for k, v in mlp_sum.items()})


class AblationHookSet:
    """Mean- or zero-ablation of target heads (o_proj input slice) and MLP outputs.

    Hooks act only during prefill (seq_len > 1) at the last prompt token, so
    cached keys/values stay consistent during generation.
    """

    def __init__(self, head_means, mlp_means, heads, mlps, head_dim: int,
                 mode: str = "mean"):
        self.head_means = head_means
        self.mlp_means = mlp_means
        self.heads = heads
        self.mlps = mlps
        self.head_dim = head_dim
        self.mode = mode
        self.handles: list[Any] = []
        self.n_interventions = 0

    def register(self, model) -> None:
        layers = get_layers(model)
        device = input_device(model)
        heads_by_layer: dict[int, list[int]] = {}
        for (l, h) in self.heads:
            heads_by_layer.setdefault(l, []).append(h)

        for l, head_list in heads_by_layer.items():
            repl = {}
            for h in head_list:
                if self.mode == "zero":
                    repl[h] = torch.zeros(self.head_dim, device=device)
                else:
                    if (l, h) not in self.head_means:
                        raise RuntimeError(f"Missing mean for head L{l}H{h}")
                    repl[h] = self.head_means[(l, h)].to(device)
            o_proj = get_o_proj(get_attn(layers[l]))
            self.handles.append(o_proj.register_forward_pre_hook(self._o_pre_hook(head_list, repl)))

        for l in self.mlps:
            if self.mode == "zero":
                repl = None
            else:
                if l not in self.mlp_means:
                    raise RuntimeError(f"Missing mean for L{l} MLP")
                repl = self.mlp_means[l].to(device)
            self.handles.append(get_mlp(layers[l]).register_forward_hook(self._mlp_hook(repl)))
        print(f"[hooks] {len(self.heads)} heads, {len(self.mlps)} MLPs, mode={self.mode}")

    def _o_pre_hook(self, head_list, repl):
        d = self.head_dim

        def hook(module, inp):
            x = inp[0]
            if x.shape[1] <= 1:
                return None
            self.n_interventions += 1
            x = x.clone()
            for h in head_list:
                x[0, -1, h * d:(h + 1) * d] = repl[h].to(x.dtype)
            return (x,) + tuple(inp[1:])
        return hook

    def _mlp_hook(self, repl):
        def hook(module, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            if h.shape[1] <= 1:
                return None
            self.n_interventions += 1
            h = h.clone()
            h[0, -1, :] = 0.0 if repl is None else repl.to(h.dtype)
            return (h,) + tuple(out[1:]) if is_tuple else h
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
