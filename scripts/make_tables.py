#!/usr/bin/env python3
"""Rebuild every results table from the saved predictions under outputs/.

Each table is written as Markdown and as LaTeX rows to outputs/tables/, and
the numbers behind the main table and figures to outputs/tables/results.json.
Missing runs are skipped, so this can be called after any driver.

    python scripts/make_tables.py [--root outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.metrics import (  # noqa: E402
    T1,
    load_jsonl,
    paired_cond_acc_ci,
    run_metrics,
    stable_buckets,
)

BACKBONES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral3-7B"), ("qwen", "Qwen2.5-7B")]
METRICS = ("SR", "SDR", "VRR", "DD", "CO")
OBJECTIVE_LABELS = [("sft", "SFT"), ("vmd", "VMD"), ("avmd", "A-VMD"),
                    ("ca_vmd", "CA-VMD"), ("ca_avmd", "CA-AVMD")]
DIRECTIONS = [("presence", "-c_hat presence"), ("volatility", "+v_hat volatility"),
              ("random", "random"), ("shuffled", "shuffled-label v_hat")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt(x, signed=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.1f}" if signed else f"{x:.1f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def fire_rates(run: Path) -> dict:
    """Share of prompts steered, per test (percent)."""
    per = read_json(run / "hook_stats.json").get("per_test", {})
    return {k: 100 * v["fire_rate"] for k, v in per.items()}


def write(out_dir: Path, name: str, md: list[str], tex: list[str]) -> None:
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n")
    (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"[tables] {name}")


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def tex_rows(rows: list[list[str]]) -> list[str]:
    return [" & ".join(r) + r" \\" for r in rows]


def per_backbone_row(label: str, cells: dict) -> list[str]:
    row = [label]
    for bb, _ in BACKBONES:
        m = cells.get(bb)
        row += [fmt(m[k]) if m else "--" for k in METRICS]
    return row


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def max_dd_cell(grid_root: Path) -> str | None:
    """Grid cell with the highest DD (ties to higher SR), for reference."""
    cells = [(run.name, m) for run in sorted(grid_root.glob("*")) if (m := run_metrics(run))]
    return max(cells, key=lambda c: (round(c[1]["DD"], 6), c[1]["SR"]))[0] if cells else None


def table_gap(root: Path, out: Path) -> None:
    rows = [per_backbone_row("Zero-shot", {bb: run_metrics(root / "eval" / bb / "zero_shot")
                                           for bb, _ in BACKBONES}),
            per_backbone_row("+ instruction", {
                bb: run_metrics(root / "eval" / bb / "zero_shot_instructed", instructed=True)
                for bb, _ in BACKBONES})]
    for key, label in OBJECTIVE_LABELS:
        rows.append(per_backbone_row(label, {bb: run_metrics(root / "eval" / bb / key)
                                             for bb, _ in BACKBONES}))
    header = ["Model"] + [f"{name} {m}" for _, name in BACKBONES for m in METRICS]
    write(out, "gap", md_table(header, rows), tex_rows(rows))


def table_main(root: Path, out: Path, results: dict) -> None:
    arms = {"zero_shot": {}, "sft": {}, "ca_avmd": {}, "presence_only": {}, "bridge": {}}
    for bb, _ in BACKBONES:
        arms["zero_shot"][bb] = run_metrics(root / "eval" / bb / "zero_shot")
        arms["sft"][bb] = run_metrics(root / "eval" / bb / "sft")
        arms["ca_avmd"][bb] = run_metrics(root / "eval" / bb / "ca_avmd")
        arms["presence_only"][bb] = run_metrics(root / "bridge" / bb / "presence_only")
        # outputs/bridge/<bb>/bridge links to the configured grid cell
        # (tau_v at 30% stable pass, BB_TAU_C); see run/04_bridge.sh.
        run = root / "bridge" / bb / "bridge"
        m = run_metrics(run)
        cfg = read_json(run / "config.json")
        arms["bridge"][bb] = m and {**m, "run": run.resolve().name, "tau_v": cfg.get("tau_v"),
                                    "tau_c": cfg.get("tau_c"),
                                    "max_dd_cell": max_dd_cell(root / "bridge" / bb / "grid")}
    labels = {"zero_shot": "Zero-shot", "sft": "SFT", "ca_avmd": "CA-AVMD",
              "presence_only": "+ presence", "bridge": "+ BRIDGe"}
    rows = [per_backbone_row(labels[a], arms[a]) for a in arms]
    header = ["Arm"] + [f"{name} {m}" for _, name in BACKBONES for m in METRICS]
    md = md_table(header, rows)
    for bb, _ in BACKBONES:
        b = arms["bridge"][bb]
        if b:
            md.append(f"\nBRIDGe cell for {bb}: {b['run']} (tau_v={b['tau_v']}, "
                      f"tau_c={b['tau_c']}); highest-DD grid cell: {b['max_dd_cell']}")
    write(out, "main", md, tex_rows(rows))
    results["main"] = arms


def table_frontier(root: Path, out: Path) -> None:
    md = []
    for bb, name in BACKBONES:
        rows = []
        for run in sorted((root / "bridge" / bb / "grid").glob("*")):
            m = run_metrics(run)
            if not m:
                continue
            cfg, fire = read_json(run / "config.json"), fire_rates(run)
            rows.append([run.name, f"{cfg.get('tau_v', 0):+.4g}", f"{cfg.get('tau_c', 0):+.3g}",
                         fmt(fire.get("test1_stable_retention")), fmt(fire.get("test4_ca_avmd")),
                         *[fmt(m[k]) for k in METRICS]])
        if rows:
            md += [f"\n## {name}\n"] + md_table(
                ["cell", "tau_v", "tau_c", "fire S", "fire V", *METRICS], rows)
    if md:
        write(out, "bridge_grid", md, [])


def table_gates(root: Path, out: Path) -> None:
    rows = []
    for bb, name in BACKBONES:
        base = run_metrics(root / "eval" / bb / "ca_avmd")
        if base:
            rows.append([name, "un-steered", fmt(base["SR"]), fmt(base["DD"]), fmt(base["CO"]),
                         "--", "--", "--"])
        for gate in ("none", "volatile_only", "presence_only", "full"):
            run = root / "ablations" / "gates" / bb / gate
            m = run_metrics(run)
            if not m:
                continue
            f = fire_rates(run)
            rows.append([name, gate, fmt(m["SR"]), fmt(m["DD"]), fmt(m["CO"]),
                         fmt(f.get("test1_stable_retention")), fmt(f.get("test4_ca_avmd")),
                         fmt(f.get("test3_knowledge_conflict"))])
    if rows:
        header = ["Backbone", "Gate", "SR", "DD", "CO", "fire S", "fire V", "fire grounded"]
        write(out, "gate_conditions", md_table(header, rows), tex_rows(rows))


def table_direction(root: Path, out: Path) -> None:
    """Change in DD against the un-steered checkpoint; dagger when the answers
    that remain on stable questions get worse (95% CI of the change in
    conditional accuracy below zero)."""
    rows, ks_all = [], set()
    cells = {}
    for bb, _ in BACKBONES:
        base_dir = root / "eval" / bb / "ca_avmd"
        base = run_metrics(base_dir)
        if not base:
            continue
        base_b = stable_buckets(load_jsonl(base_dir / T1))
        for run in sorted((root / "ablations" / "direction" / bb).glob("*_k*")):
            d, k = run.name.rsplit("_k", 1)
            m = run_metrics(run)
            if not m:
                continue
            run_b = stable_buckets(load_jsonl(run / T1))
            _, _, hi = paired_cond_acc_ci(run_b, base_b)
            cells[(bb, d, float(k))] = (m["DD"] - base["DD"], hi < 0)
            ks_all.add(float(k))
    if not cells:
        return
    ks = sorted(ks_all)
    md_rows, tex = [], []
    for d, label in DIRECTIONS:
        md_row, tex_row = [label], [label]
        for bb, _ in BACKBONES:
            for k in ks:
                c = cells.get((bb, d, k))
                if c is None:
                    md_row.append("--")
                    tex_row.append("--")
                    continue
                md_row.append(fmt(c[0], signed=True) + (" *" if c[1] else ""))
                tex_row.append(f"${fmt(c[0], signed=True)}" + (r"^{\dagger}$" if c[1] else "$"))
        md_rows.append(md_row)
        tex.append(" & ".join(tex_row) + r" \\")
    header = ["Direction"] + [f"{bb} k={k:g}" for bb, _ in BACKBONES for k in ks]
    md = md_table(header, md_rows) + ["", "* the change in conditional accuracy on stable "
                                      "questions has a 95% CI below zero."]
    write(out, "direction_ablation", md, tex)


def table_sensitivity(root: Path, out: Path) -> None:
    rows = []
    sens = root / "ablations" / "sensitivity" / "llama"
    runs = [(sens / f"alpha_{a}", f"alpha={a}") for a in (10, 20, 25, 50, 75, 100)]
    runs += [(sens / f"tau_c_{t}", f"tau_c={t}") for t in ("-0.7", "-0.6", "-0.4", "-0.3")]
    runs += [(sens / "layer_16_matched", "l=16, matched"), (sens / "layer_24_matched", "l=24, matched"),
             (sens / "layer_16_fixed", "l=16, fixed tau_c")]
    runs += [(root / "eval" / "llama" / "ca_avmd", "un-steered")]
    for rate in ("0.40", "0.30", "0.20"):
        for run in sorted((root / "bridge" / "llama" / "grid").glob(f"tv{rate}_tc*")):
            if read_json(run / "config.json").get("tau_c") == 0.5:
                runs.append((run, f"BRIDGe tau_v={int(float(rate) * 100)}%"))
    for run, label in runs:
        m = run_metrics(run)
        if not m:
            continue
        f = fire_rates(run)
        fire = f"{fmt(f.get('test1_stable_retention', 0.0))}/{fmt(f.get('test4_ca_avmd', 0.0))}"
        rows.append([label, fire, fmt(m["SR"]), fmt(m["SDR"]), fmt(m["VRR"]), fmt(m["DD"]),
                     fmt(m["CO"])])
    if rows:
        header = ["Setting", "Fire (S/V)", "SR", "SDR", "VRR", "DD", "CO"]
        write(out, "sensitivity", md_table(header, rows), tex_rows(rows))


def table_seeds(root: Path, out: Path) -> None:
    variants = OBJECTIVE_LABELS + [("bridge", "CA-AVMD + BRIDGe")]
    rows = []
    for bb, name in BACKBONES:
        per = {}
        for key, label in variants:
            ms = {s: run_metrics(p) for p in sorted((root / "seeds" / bb).glob(f"{key}_seed*"))
                  if (s := p.name.rsplit("_seed", 1)[1])}
            ms = {s: m for s, m in ms.items() if m}
            if not ms:
                continue
            per[key] = ms
            arr = {k: np.array([m[k] for m in ms.values()]) for k in METRICS}
            rows.append([name, label, str(len(ms))] +
                        [f"{arr[k].mean():.1f} +- {arr[k].std(ddof=1) if len(ms) > 1 else 0:.1f}"
                         for k in METRICS])
        if "bridge" in per and "ca_avmd" in per:
            seeds = sorted(set(per["bridge"]) & set(per["ca_avmd"]))
            deltas = {k: np.array([per["bridge"][s][k] - per["ca_avmd"][s][k] for s in seeds])
                      for k in METRICS}
            rows.append([name, "paired delta (BRIDGe - CA-AVMD)", str(len(seeds))] +
                        [f"{deltas[k].mean():+.1f} +- "
                         f"{deltas[k].std(ddof=1) if len(seeds) > 1 else 0:.1f}" for k in METRICS])
    if rows:
        write(out, "seed_variance", md_table(["Backbone", "Variant", "n", *METRICS], rows),
              tex_rows(rows))


def table_baselines(root: Path, out: Path) -> None:
    rows = []
    for bb, name in BACKBONES:
        for fw, label in (("flare", "FLARE"), ("selfrag", "Self-RAG")):
            m = run_metrics(root / "baselines" / bb / fw)
            if m:
                rows.append([f"{name} {label}", *[fmt(m[k]) for k in METRICS]])
    if rows:
        write(out, "baselines", md_table(["Framework", *METRICS], rows), tex_rows(rows))


def table_dose(root: Path, out: Path, results: dict) -> None:
    """alpha in class-gap units per backbone (k = alpha / gap)."""
    rows, dose = [], {}
    for bb, name in BACKBONES:
        metas = sorted((root / "directions" / bb / "hook").glob("layer_*_vhat.json"))
        cfg = read_json(root / "bridge" / bb / "presence_only" / "config.json")
        if not metas or not cfg:
            continue
        meta = read_json(metas[0])
        alpha = cfg["alpha"]
        dose[bb] = {"layer": meta["layer"], "gap": meta["gap"],
                    "mean_norm": meta["mean_activation_norm"], "alpha": alpha,
                    "k": alpha / meta["gap"]}
        rows.append([name, str(meta["layer"]), f"{meta['gap']:.2f}",
                     f"{meta['mean_activation_norm']:.1f}", f"{alpha:g}",
                     f"{alpha / meta['gap']:.2f}",
                     f"{100 * alpha / meta['mean_activation_norm']:.1f}"])
    if rows:
        header = ["Backbone", "Hook layer", "gap", "mean |h|", "alpha", "k", "|inj|/|h| (%)"]
        write(out, "dose_calibration", md_table(header, rows), tex_rows(rows))
    results["dose"] = dose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    out = args.root / "tables"
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    table_gap(args.root, out)
    table_main(args.root, out, results)
    table_frontier(args.root, out)
    table_gates(args.root, out)
    table_direction(args.root, out)
    table_sensitivity(args.root, out)
    table_seeds(args.root, out)
    table_baselines(args.root, out)
    table_dose(args.root, out, results)
    results["instructed"] = {bb: run_metrics(args.root / "eval" / bb / "zero_shot_instructed",
                                             instructed=True) for bb, _ in BACKBONES}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
