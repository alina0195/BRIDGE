"""Behavioral tests for the retrieval-framework baselines.

A framework is given as two callables:

    adaptive(instruction, paragraph) -> dict
        the framework's own adaptive-retrieval loop.  ``paragraph`` is the
        passage it may retrieve (None: nothing to retrieve).  Returns
        ``clean_answer`` and ``did_retrieve``; other fields are kept as
        per-item metadata.
    grounded(question, context, test) -> (answer, metadata)
        generation with the passage always provided (test 3, and the grounded
        phase of test 4).

Protocol.  Files, prompts and splits are those of bridge.evaluation, so
bridge.metrics.run_metrics scores these runs exactly like every other row.

    test 1   stable, BARE_QA, no passage                -> SR, SDR
    test 2   volatile, BARE_QA_WITH_REFUSAL, gold passage
             offered to the adaptive loop                -> Ref, Halluc
    test 3   volatile, passage always provided           -> CO
    test 4   vacuum: volatile, BARE_QA, no passage       -> VRR
             grounded: passage always provided           -> appendix CO
    oracle   optional: test 1 with the gold passage offered to the adaptive
             loop -> test1_stable_retention_oracle.* (not read by run_metrics)

SDR and VRR are both measured on the bare question with nothing to retrieve,
so DD = VRR - SDR compares like with like.  The retrieval trigger still runs
in these conditions and its decision is logged per item.  Each question is
decoded once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

from bridge.evaluation import TEST_FILES, TestResult, _case, _mean, _refusal, save_result
from bridge.metrics import (
    TRAINED_DEFERRAL_PATTERN,
    best_score,
    contains_match,
    load_jsonl,
    run_metrics,
)
from bridge.prompts import BARE_QA, BARE_QA_WITH_REFUSAL

Adaptive = Callable[[str, Optional[str]], dict]
Grounded = Callable[[str, str, int], Tuple[str, dict]]

ORACLE_T1 = "test1_stable_retention_oracle"


def _extras(out: dict) -> dict:
    """Framework outputs other than the answer, as JSON-friendly metadata."""
    return {k: float(v) if isinstance(v, bool) else v
            for k, v in out.items() if k != "clean_answer"}


def _progress(tag: str, i: int, n: int) -> None:
    if (i + 1) % 50 == 0 or i + 1 == n:
        print(f"  [{tag}] {i + 1}/{n}")


def _retrieval_rate(cases) -> float:
    return _mean(c.metadata["did_retrieve"] for c in cases)


def stable_retention(adaptive: Adaptive, eval_path: Path, oracle: bool = False) -> TestResult:
    """Test 1: stable questions, bare prompt; ``oracle`` offers the gold passage."""
    records = load_jsonl(eval_path)
    cases = []
    for i, rec in enumerate(records):
        ctx = rec.get("golden_context", "") if oracle else ""
        out = adaptive(BARE_QA.format(question=rec["question"]),
                       ctx if ctx.strip() else None)
        pred = out["clean_answer"]
        golds = [rec["latest_answer"]] + rec.get("all_answers", [])
        cases.append(_case(rec["question"], pred, golds,
                           is_refusal=_refusal(pred), **_extras(out)))
        _progress("test 1" + (" oracle" if oracle else ""), i, len(records))
    return TestResult("stable_retention", {
        "exact_match": _mean(c.exact_match for c in cases),
        "contains_match": _mean(c.contains_match for c in cases),
        "token_f1": _mean(c.token_f1 for c in cases),
        "refusal_rate": _mean(c.metadata["is_refusal"] for c in cases),
        "retrieval_frequency": _retrieval_rate(cases),
    }, cases)


def parametric_void(adaptive: Adaptive, eval_path: Path) -> TestResult:
    """Test 2: volatile questions, instructed prompt, gold passage retrievable."""
    records = load_jsonl(eval_path)
    cases = []
    for i, rec in enumerate(records):
        ctx = rec.get("golden_context", "")
        out = adaptive(BARE_QA_WITH_REFUSAL.format(question=rec["question"]),
                       ctx if ctx.strip() else None)
        pred = out["clean_answer"]
        outdated, latest = rec["answer"], rec["latest_answer"]
        cases.append(_case(
            rec["question"], pred, [outdated],
            outdated_answer=outdated, latest_answer=latest,
            is_refusal=_refusal(pred),
            is_deferral=1.0 if TRAINED_DEFERRAL_PATTERN.search(pred) else 0.0,
            freshness_match=best_score(pred, [latest], contains_match),
            **_extras(out),
        ))
        _progress("test 2", i, len(records))
    return TestResult("parametric_void", {
        "hallucination_rate": _mean(c.contains_match for c in cases),
        "refusal_rate": _mean(c.metadata["is_refusal"] for c in cases),
        "deferral_rate": _mean(c.metadata["is_deferral"] for c in cases),
        "freshness_rate": _mean(c.metadata["freshness_match"] for c in cases),
        "retrieval_frequency": _retrieval_rate(cases),
    }, cases)


def _grounded_cases(grounded: Grounded, records: list[dict], test: int, tag: str,
                    **metadata) -> list:
    cases = []
    for i, rec in enumerate(records):
        pred, meta = grounded(rec["question"], rec["golden_context"], test)
        latest, outdated = rec["latest_answer"], rec["answer"]
        cases.append(_case(
            rec["question"], pred, [latest], **metadata,
            outdated_answer=outdated, latest_answer=latest,
            conflict_rate=best_score(pred, [outdated], contains_match),
            is_refusal=_refusal(pred), **meta,
        ))
        _progress(tag, i, len(records))
    return cases


def knowledge_conflict(grounded: Grounded, eval_path: Path) -> TestResult:
    """Test 3: volatile questions with a passage holding the current value (CO)."""
    records = [r for r in load_jsonl(eval_path) if r.get("golden_context", "").strip()]
    cases = _grounded_cases(grounded, records, 3, "test 3")
    return TestResult("knowledge_conflict", {
        "contextual_obedience": _mean(c.contains_match for c in cases),
        "conflict_rate": _mean(c.metadata["conflict_rate"] for c in cases),
        "refusal_rate": _mean(c.metadata["is_refusal"] for c in cases),
        "exact_match": _mean(c.exact_match for c in cases),
        "token_f1": _mean(c.token_f1 for c in cases),
    }, cases)


def vacuum_and_grounded(adaptive: Adaptive, grounded: Grounded, evolved_path: Path,
                        conflict_path: Path) -> TestResult:
    """Test 4: volatile questions with nothing to retrieve (VRR), then with the passage."""
    evolved = load_jsonl(evolved_path)
    cases = []
    for i, rec in enumerate(evolved):
        out = adaptive(BARE_QA.format(question=rec["question"]), None)
        pred = out["clean_answer"]
        cases.append(_case(
            rec["question"], pred, [rec["answer"]], phase="vacuum",
            outdated_answer=rec["answer"], latest_answer=rec["latest_answer"],
            is_refusal=_refusal(pred), **_extras(out),
        ))
        _progress("test 4 vacuum", i, len(evolved))
    conflict = [r for r in load_jsonl(conflict_path) if r.get("golden_context", "").strip()]
    cases += _grounded_cases(grounded, conflict, 4, "test 4 grounded", phase="grounded")
    vac = [c for c in cases if c.metadata["phase"] == "vacuum"]
    grd = [c for c in cases if c.metadata["phase"] == "grounded"]
    return TestResult("ca_avmd", {
        "vrr": _mean(c.metadata["is_refusal"] for c in vac),
        "co": _mean(c.contains_match for c in grd),
        "kcr": _mean(c.metadata["conflict_rate"] for c in grd),
        "n_vacuum": len(vac),
        "n_grounded": len(grd),
    }, cases)


def run_baseline(adaptive: Adaptive, grounded: Grounded, eval_dir: Path, output_dir: Path,
                 tests: set[int], oracle_t1: bool = False, config: dict | None = None) -> dict:
    """Run the selected tests, save predictions, summary.json and config.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stable = eval_dir / "eval_stable.jsonl"
    evolved = eval_dir / "eval_evolved.jsonl"
    conflict = eval_dir / "eval_conflict.jsonl"
    runners = {
        1: (TEST_FILES[1], lambda: stable_retention(adaptive, stable)),
        2: (TEST_FILES[2], lambda: parametric_void(adaptive, evolved)),
        3: (TEST_FILES[3], lambda: knowledge_conflict(grounded, conflict)),
        4: (TEST_FILES[4], lambda: vacuum_and_grounded(adaptive, grounded, evolved, conflict)),
    }
    jobs = [runners[t] for t in sorted(tests)]
    if oracle_t1:
        jobs.append((ORACLE_T1, lambda: stable_retention(adaptive, stable, oracle=True)))

    summary = {}
    for stem, run in jobs:
        print(f"\n=== {stem} ===")
        result = run()
        save_result(result, output_dir / f"{stem}.json")
        summary[stem] = result.scores_detail
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in result.scores_detail.items()))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(
        {**(config or {}), "tests": sorted(tests), "oracle_t1": oracle_t1}, indent=2))

    m = run_metrics(output_dir)
    if m:
        print("\n" + "  ".join(f"{k}={m[k]:.1f}" for k in ("SR", "SDR", "VRR", "DD", "CO")))
    print(f"[done] {output_dir}")
    return summary
