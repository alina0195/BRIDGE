#!/usr/bin/env python3
"""Behavioral evaluation, optionally with the BRIDGe hook attached.

Without steering arguments this evaluates a model as is (zero-shot or a
fine-tuned adapter).  With --presence-dir it registers the hook of
Equation (bridge) before running the tests; every model.generate() call then
passes through it.

Examples
--------
    # CA-AVMD checkpoint, no steering
    python scripts/evaluate.py --base-model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/models/llama_ca_avmd/adapter \
        --output-dir outputs/eval/llama/ca_avmd

    # "+ instruction" row: zero-shot model, abstention instruction everywhere
    python scripts/evaluate.py --base-model meta-llama/Llama-3.1-8B-Instruct \
        --instructed --tests 1,2,3 --output-dir outputs/eval/llama/zero_shot_instructed

    # BRIDGe (dual gate), alpha = 20
    python scripts/evaluate.py --base-model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path outputs/models/llama_ca_avmd/adapter \
        --volatile-dir outputs/directions/llama/hook/layer_31_vhat.npy \
        --presence-dir outputs/directions/llama/presence/layer_31_presence.npy \
        --layer 31 --alpha 20 --gate full --tau-v 4.269 --tau-c 0.5 \
        --output-dir outputs/bridge/llama/grid/tv0.30_tc0.5

    # Direction ablation: ungated +v_hat at k = 2 class gaps
    python scripts/evaluate.py ... --gate none --steer volatility \
        --steer-dir outputs/directions/llama/hook/layer_31_vhat.npy \
        --dir-meta outputs/directions/llama/hook/layer_31_vhat.json --k 2
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.evaluation import run_tests  # noqa: E402
from bridge.metrics import run_metrics  # noqa: E402
from bridge.model import load_model  # noqa: E402
from bridge.steering import GATES, BridgeHook  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/evowiki"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tests", default="1,2,3,4")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--instructed", action="store_true",
                    help="Add the abstention instruction to tests 1 and 3.")

    st = ap.add_argument_group("steering (BRIDGe hook)")
    st.add_argument("--presence-dir", type=Path, default=None,
                    help="Presence probe direction c_hat (.npy). Enables the hook.")
    st.add_argument("--volatile-dir", type=Path, default=None,
                    help="Volatility direction v_hat (.npy); required by gates that read it.")
    st.add_argument("--layer", type=int, default=-1)
    st.add_argument("--gate", choices=GATES, default="full")
    st.add_argument("--alpha", type=float, default=20.0, help="Injection magnitude.")
    st.add_argument("--tau-v", type=float, default=0.0)
    st.add_argument("--tau-c", type=float, default=0.0)
    st.add_argument("--steer", choices=("presence", "volatility", "random"), default="presence",
                    help="Injected direction: -c_hat (BRIDGe), +v from --steer-dir, "
                         "or a random unit vector.")
    st.add_argument("--steer-dir", type=Path, default=None)
    st.add_argument("--steer-seed", type=int, default=0)
    st.add_argument("--k", type=float, default=None,
                    help="Magnitude in class-gap units (k * gap); overrides --alpha.")
    st.add_argument("--dir-meta", type=Path, default=None,
                    help="JSON written next to v_hat; supplies gap for --k.")
    st.add_argument("--match-fire-rate", type=float, default=None,
                    help="Set tau_c to the quantile that gives this overall fire rate "
                         "(presence scores measured in a prefill-only pass).")
    args = ap.parse_args()

    tests = {int(t) for t in args.tests.split(",")}
    tokenizer, model = load_model(args.base_model, adapter_path=args.adapter_path)

    hook, abs_layer, magnitude = None, None, None
    if args.presence_dir is not None:
        presence = np.load(args.presence_dir)
        if args.volatile_dir is not None:
            volatile = np.load(args.volatile_dir)
        elif args.gate in ("presence_only", "none"):
            volatile = presence          # not read by these gates
        else:
            ap.error(f"--gate {args.gate} needs --volatile-dir")

        steer_dir = None
        if args.steer == "volatility":
            steer_dir = np.load(args.steer_dir).astype(np.float64)
        elif args.steer == "random":
            rng = np.random.default_rng(args.steer_seed)
            steer_dir = rng.standard_normal(presence.shape[0])

        magnitude = args.alpha
        if args.k is not None:
            gap = json.loads(args.dir_meta.read_text())["gap"]
            magnitude = args.k * gap
        hook = BridgeHook(volatile, presence, magnitude, args.tau_v, args.tau_c,
                          args.gate, steer_dir)
        abs_layer = hook.register(model, args.layer)

        if args.match_fire_rate is not None:
            hook.record_scores, hook.magnitude = True, 0.0
            with tempfile.TemporaryDirectory() as tmp:
                run_tests(model, tokenizer, args.eval_dir, Path(tmp), tests, max_new_tokens=1)
            scores = np.asarray(hook.c_scores)
            hook.tau_c = float(np.quantile(scores, args.match_fire_rate))
            hook.record_scores, hook.magnitude, hook.c_scores = False, magnitude, []
            print(f"[hook] tau_c={hook.tau_c:.4f} matches fire rate {args.match_fire_rate:.3f}")

    per_test = {}

    def before(stem):
        if hook:
            hook.reset_stats()

    def after(stem):
        if hook:
            per_test[stem] = hook.snapshot()

    summary = run_tests(model, tokenizer, args.eval_dir, args.output_dir, tests,
                        args.max_new_tokens, args.instructed, before, after)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    config = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "tests": sorted(tests),
        "instructed": args.instructed,
        "max_new_tokens": args.max_new_tokens,
    }
    if hook:
        steered = sum(s["steered"] for s in per_test.values())
        calls = sum(s["calls"] for s in per_test.values())
        config.update({
            "presence_dir": str(args.presence_dir),
            "volatile_dir": str(args.volatile_dir) if args.volatile_dir else None,
            "layer": abs_layer, "gate": args.gate, "steer": args.steer,
            "steer_dir": str(args.steer_dir) if args.steer_dir else None,
            "steer_seed": args.steer_seed if args.steer == "random" else None,
            "alpha": args.alpha, "k": args.k, "magnitude": magnitude,
            "tau_v": hook.tau_v, "tau_c": hook.tau_c,
        })
        (args.output_dir / "hook_stats.json").write_text(json.dumps({
            "overall_fire_rate": steered / calls if calls else None,
            "per_test": per_test,
        }, indent=2))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    m = run_metrics(args.output_dir, instructed=args.instructed)
    if m:
        print("\n" + "  ".join(f"{k}={m[k]:.1f}" for k in ("SR", "SDR", "VRR", "DD", "CO")))
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
