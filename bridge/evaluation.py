"""Behavioral evaluation: one test per scored cell of the rule.

    test 1  stable, bare prompt           -> SR, SDR
    test 2  volatile, instructed prompt   -> VRR of the "+ instruction" row
    test 3  volatile, grounded prompt     -> CO
    test 4  volatile, bare prompt         -> VRR  (vacuum phase)
            volatile, grounded prompt     -> appendix CO (grounded phase)

Each test writes <name>.json (aggregate scores) and <name>.predictions.jsonl
(one line per item); tables are rebuilt from the prediction files only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from bridge.metrics import (
    DEFERRAL_PATTERN,
    TRAINED_DEFERRAL_PATTERN,
    best_score,
    contains_match,
    exact_match,
    load_jsonl,
    token_f1,
)
from bridge.model import generate
from bridge.prompts import (
    BARE_QA,
    BARE_QA_WITH_REFUSAL,
    RAG_QA,
    RAG_QA_GROUNDED,
    RAG_QA_WITH_REFUSAL,
)


@dataclass
class Case:
    question: str
    predicted: str
    gold: str
    exact_match: float
    contains_match: float
    token_f1: float
    metadata: dict = field(default_factory=dict)


@dataclass
class TestResult:
    test_name: str
    scores_detail: dict
    per_case: list[Case]


def _case(question: str, predicted: str, golds: Sequence[str], **metadata) -> Case:
    return Case(
        question=question, predicted=predicted, gold=golds[0],
        exact_match=best_score(predicted, golds, exact_match),
        contains_match=best_score(predicted, golds, contains_match),
        token_f1=best_score(predicted, golds, token_f1),
        metadata=metadata,
    )


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _refusal(text: str) -> float:
    return 1.0 if DEFERRAL_PATTERN.search(text) else 0.0


def _progress(tag: str, i: int, n: int) -> None:
    if (i + 1) % 50 == 0:
        print(f"  [{tag}] {i + 1}/{n}")


def test_stable_retention(model, tokenizer, eval_path, max_new_tokens=64,
                          prompt_template=BARE_QA) -> TestResult:
    """Test 1: stable questions, no context, scored against the correct value."""
    records = load_jsonl(eval_path)
    cases = []
    for i, rec in enumerate(records):
        pred = generate(prompt_template.format(question=rec["question"]),
                        tokenizer, model, max_new_tokens)
        golds = [rec["latest_answer"]] + rec.get("all_answers", [])
        cases.append(_case(rec["question"], pred, golds, is_refusal=_refusal(pred)))
        _progress("test 1", i, len(records))
    return TestResult("stable_retention", {
        "exact_match": _mean(c.exact_match for c in cases),
        "contains_match": _mean(c.contains_match for c in cases),
        "token_f1": _mean(c.token_f1 for c in cases),
        "refusal_rate": _mean(c.metadata["is_refusal"] for c in cases),
    }, cases)


def test_parametric_void(model, tokenizer, eval_path, max_new_tokens=64) -> TestResult:
    """Test 2: volatile questions, instructed bare prompt.

    Also records how often the output repeats the outdated T1 value
    (hallucination) or states the current T2 value (freshness).
    """
    records = load_jsonl(eval_path)
    cases = []
    for i, rec in enumerate(records):
        pred = generate(BARE_QA_WITH_REFUSAL.format(question=rec["question"]),
                        tokenizer, model, max_new_tokens)
        outdated, latest = rec["answer"], rec["latest_answer"]
        cases.append(_case(
            rec["question"], pred, [outdated],
            outdated_answer=outdated, latest_answer=latest,
            is_refusal=_refusal(pred),
            is_deferral=1.0 if TRAINED_DEFERRAL_PATTERN.search(pred) else 0.0,
            freshness_match=best_score(pred, [latest], contains_match),
        ))
        _progress("test 2", i, len(records))
    return TestResult("parametric_void", {
        "hallucination_rate": _mean(c.contains_match for c in cases),
        "refusal_rate": _mean(c.metadata["is_refusal"] for c in cases),
        "deferral_rate": _mean(c.metadata["is_deferral"] for c in cases),
        "freshness_rate": _mean(c.metadata["freshness_match"] for c in cases),
    }, cases)


def test_knowledge_conflict(model, tokenizer, eval_path, max_new_tokens=64,
                            prompt_template=RAG_QA) -> TestResult:
    """Test 3: volatile questions with a context that holds the T2 value (CO)."""
    records = [r for r in load_jsonl(eval_path) if r.get("golden_context", "").strip()]
    cases = []
    for i, rec in enumerate(records):
        pred = generate(prompt_template.format(context=rec["golden_context"],
                                               question=rec["question"]),
                        tokenizer, model, max_new_tokens)
        latest, outdated = rec["latest_answer"], rec["answer"]
        cases.append(_case(
            rec["question"], pred, [latest],
            outdated_answer=outdated, latest_answer=latest,
            conflict_rate=best_score(pred, [outdated], contains_match),
            is_refusal=_refusal(pred),
        ))
        _progress("test 3", i, len(records))
    return TestResult("knowledge_conflict", {
        "contextual_obedience": _mean(c.contains_match for c in cases),
        "conflict_rate": _mean(c.metadata["conflict_rate"] for c in cases),
        "refusal_rate": _mean(c.metadata["is_refusal"] for c in cases),
        "context_deferral_rate": _mean(
            1.0 if c.metadata["is_refusal"] > 0 and c.contains_match <= 0 else 0.0
            for c in cases),
    }, cases)


def test_vacuum_and_grounded(model, tokenizer, evolved_path, conflict_path,
                             max_new_tokens=64) -> TestResult:
    """Test 4: volatile questions without context (VRR), then with context."""
    cases = []
    evolved = load_jsonl(evolved_path)
    for i, rec in enumerate(evolved):
        pred = generate(BARE_QA.format(question=rec["question"]), tokenizer, model,
                        max_new_tokens)
        cases.append(_case(
            rec["question"], pred, [rec["answer"]], phase="vacuum",
            outdated_answer=rec["answer"], latest_answer=rec["latest_answer"],
            is_refusal=_refusal(pred),
        ))
        _progress("test 4 vacuum", i, len(evolved))
    grounded = [r for r in load_jsonl(conflict_path) if r.get("golden_context", "").strip()]
    for i, rec in enumerate(grounded):
        pred = generate(RAG_QA_GROUNDED.format(context=rec["golden_context"],
                                               question=rec["question"]),
                        tokenizer, model, max_new_tokens)
        cases.append(_case(
            rec["question"], pred, [rec["latest_answer"]], phase="grounded",
            outdated_answer=rec["answer"], latest_answer=rec["latest_answer"],
            conflict_rate=best_score(pred, [rec["answer"]], contains_match),
        ))
        _progress("test 4 grounded", i, len(grounded))
    vac = [c for c in cases if c.metadata["phase"] == "vacuum"]
    grd = [c for c in cases if c.metadata["phase"] == "grounded"]
    return TestResult("ca_avmd", {
        "vrr": _mean(c.metadata["is_refusal"] for c in vac),
        "co": _mean(c.contains_match for c in grd),
        "kcr": _mean(c.metadata["conflict_rate"] for c in grd),
        "n_vacuum": len(vac),
        "n_grounded": len(grd),
    }, cases)


TEST_FILES = {
    1: "test1_stable_retention",
    2: "test2_parametric_void",
    3: "test3_knowledge_conflict",
    4: "test4_ca_avmd",
}


def save_result(result: TestResult, path: Path) -> None:
    path.write_text(json.dumps({
        "test_name": result.test_name,
        "n_examples": len(result.per_case),
        "scores_detail": result.scores_detail,
    }, indent=2), encoding="utf-8")
    with path.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as f:
        for c in result.per_case:
            f.write(json.dumps({
                "question": c.question, "predicted": c.predicted, "gold": c.gold,
                "exact_match": c.exact_match, "contains_match": c.contains_match,
                "token_f1": c.token_f1, "metadata": c.metadata,
            }, ensure_ascii=False) + "\n")


def run_tests(model, tokenizer, eval_dir: Path, output_dir: Path, tests: set[int],
              max_new_tokens: int = 64, instructed: bool = False,
              before_test=None, after_test=None) -> dict:
    """Run the selected tests and save their outputs.

    ``instructed`` switches tests 1 and 3 to the abstention-instruction
    prompts (the "+ instruction" row).  ``before_test`` / ``after_test`` are
    optional callbacks taking the test file stem, used by the steering hook to
    keep per-test gate statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stable = eval_dir / "eval_stable.jsonl"
    evolved = eval_dir / "eval_evolved.jsonl"
    conflict = eval_dir / "eval_conflict.jsonl"
    runners = {
        1: lambda: test_stable_retention(
            model, tokenizer, stable, max_new_tokens,
            BARE_QA_WITH_REFUSAL if instructed else BARE_QA),
        2: lambda: test_parametric_void(model, tokenizer, evolved, max_new_tokens),
        3: lambda: test_knowledge_conflict(
            model, tokenizer, conflict, max_new_tokens,
            RAG_QA_WITH_REFUSAL if instructed else RAG_QA),
        4: lambda: test_vacuum_and_grounded(model, tokenizer, evolved, conflict,
                                            max_new_tokens),
    }
    summary = {}
    for t in sorted(tests):
        stem = TEST_FILES[t]
        print(f"\n=== {stem} ===")
        if before_test:
            before_test(stem)
        result = runners[t]()
        save_result(result, output_dir / f"{stem}.json")
        if after_test:
            after_test(stem)
        summary[result.test_name] = result.scores_detail
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in result.scores_detail.items()))
    return summary
