"""EvoWiki download, parsing and train / eval splits.

EvoWiki (Tang et al., ACL 2025) partitions Wikidata facts by how they moved
between knowledge snapshots: stable, evolved and uncharted.  We use the stable
and evolved facts.  For an evolved fact the training target is the outdated
(T1) value and the evaluation target is the current (T2) value.

Splits are made at the entity level so that no entity appears in both train
and eval.  Outputs:

    train.jsonl           stable + evolved training records
    eval_stable.jsonl     stable eval records (SR, SDR)
    eval_evolved.jsonl    evolved eval records (VRR)
    eval_conflict.jsonl   evolved eval records whose golden context contains
                          the T2 answer (CO)

Source: https://github.com/wtangdev/EvoWiki
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

MEGA_FOLDER_URL = "https://mega.nz/folder/lmEVgZqC#4OpyAY57gBiJFE1I18_fdA"
EVOLUTION_LEVELS = ("stable", "evolved", "uncharted")


@dataclass
class EvoWikiRecord:
    question: str
    latest_answer: str
    all_answers: list[str]
    evolution_level: str
    contriever_chunks: list[str] = field(default_factory=list)
    bm25_chunks: list[str] = field(default_factory=list)


@dataclass
class SplitRecord:
    question: str
    answer: str            # T1 (outdated) value for evolved, correct value for stable
    fact_type: str         # "stable" or "evolved"
    latest_answer: str     # T2 (current) value
    golden_context: str    # retrieved chunk that contains the T2 value
    all_answers: list[str]
    entity_key: str


# ---------------------------------------------------------------------------
# Download and parsing
# ---------------------------------------------------------------------------


def download_evowiki(output_dir: str | Path) -> Path:
    """Download the raw EvoWiki release into <output_dir>/raw (needs megatools)."""
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.rglob("*.json")):
        print(f"[data] raw EvoWiki already present in {raw_dir}")
        return raw_dir
    if shutil.which("megadl"):
        subprocess.run(["megadl", "--path", str(raw_dir), MEGA_FOLDER_URL], check=True)
    elif shutil.which("mega-get"):
        subprocess.run(["mega-get", MEGA_FOLDER_URL, str(raw_dir)], check=True)
    else:
        raise RuntimeError(
            "No MEGA download tool found. Install megatools, or download the "
            f"release manually from {MEGA_FOLDER_URL} into {raw_dir}."
        )
    if not any(raw_dir.rglob("*.json")):
        raise FileNotFoundError(f"No JSON files found in {raw_dir} after download.")
    return raw_dir


def load_evowiki(raw_dir: str | Path) -> list[EvoWikiRecord]:
    """Parse every EvoWiki JSON file under raw_dir, merging duplicate questions."""
    raw_dir = Path(raw_dir)
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {raw_dir}.")
    records: list[EvoWikiRecord] = []
    for jf in json_files:
        level = next((lv for lv in EVOLUTION_LEVELS if lv in jf.stem.lower()), "unknown")
        data = _load_json(jf)
        if isinstance(data, dict):
            items = [data] if "question" in data else list(data.values())
        elif isinstance(data, list):
            items = data
        else:
            continue
        for item in items:
            rec = _parse_record(item, level)
            if rec is not None:
                records.append(rec)

    # The release ships two retriever variants per file with the same
    # questions; merge them into one record per question.
    merged: dict[str, EvoWikiRecord] = {}
    for r in records:
        key = r.question.strip()
        if key in merged:
            merged[key].contriever_chunks.extend(r.contriever_chunks)
            merged[key].bm25_chunks.extend(r.bm25_chunks)
        else:
            merged[key] = r
    out = list(merged.values())
    counts = Counter(r.evolution_level for r in out)
    print(f"[data] loaded {len(out)} records ({dict(sorted(counts.items()))})")
    return out


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            pass
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunks(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _parse_record(item: dict, default_level: str) -> EvoWikiRecord | None:
    question = item.get("question", "").strip()
    if not question:
        return None
    latest = item.get("answer", "")
    if isinstance(latest, list):
        latest = latest[0] if latest else ""
    latest = str(latest).strip()
    all_raw = item.get("all_answer", item.get("all_answers", []))
    if isinstance(all_raw, str):
        all_raw = [all_raw]
    all_answers = [str(a).strip() for a in all_raw if str(a).strip()]
    if not latest and all_answers:
        latest = all_answers[0]
    if not latest:
        return None
    return EvoWikiRecord(
        question=question,
        latest_answer=latest,
        all_answers=all_answers,
        evolution_level=item.get("evolution_level", item.get("type", default_level)),
        contriever_chunks=_chunks(item.get("contriever_top30")),
        bm25_chunks=_chunks(item.get("bm25_top30")),
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def build_splits(records: Sequence[EvoWikiRecord], output_dir: str | Path,
                 train_ratio: float = 0.8, seed: int = 42) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stable = _group_by_entity([r for r in records if r.evolution_level == "stable"])
    evolved = _group_by_entity([r for r in records if r.evolution_level == "evolved"])
    s_train, s_eval = _split_keys(list(stable), train_ratio, seed)
    e_train, e_eval = _split_keys(list(evolved), train_ratio, seed)

    train = [_to_split_record(r, "stable") for k in s_train for r in stable[k]]
    train += [x for k in e_train for r in evolved[k]
              if (x := _to_split_record(r, "evolved")).answer]
    eval_stable = [_to_split_record(r, "stable") for k in s_eval for r in stable[k]]
    eval_evolved = [x for k in e_eval for r in evolved[k]
                    if (x := _to_split_record(r, "evolved")).answer]
    eval_conflict = [
        r for r in eval_evolved
        if r.golden_context.strip()
        and r.latest_answer.strip().lower() in r.golden_context.lower()
    ]

    counts = {
        "train": _write_jsonl(output_dir / "train.jsonl", train),
        "eval_stable": _write_jsonl(output_dir / "eval_stable.jsonl", eval_stable),
        "eval_evolved": _write_jsonl(output_dir / "eval_evolved.jsonl", eval_evolved),
        "eval_conflict": _write_jsonl(output_dir / "eval_conflict.jsonl", eval_conflict),
    }
    (output_dir / "stats.json").write_text(json.dumps({
        "train_ratio": train_ratio, "seed": seed, "splits": counts,
        "stable_entities_train": len(s_train), "stable_entities_eval": len(s_eval),
        "evolved_entities_train": len(e_train), "evolved_entities_eval": len(e_eval),
    }, indent=2), encoding="utf-8")
    return counts


def _to_split_record(rec: EvoWikiRecord, fact_type: str) -> SplitRecord:
    answer = _outdated_answer(rec) if fact_type == "evolved" else rec.latest_answer
    return SplitRecord(
        question=rec.question,
        answer=answer,
        fact_type=fact_type,
        latest_answer=rec.latest_answer,
        golden_context=_golden_context(rec),
        all_answers=rec.all_answers,
        entity_key=_entity_key(rec.question),
    )


def _outdated_answer(rec: EvoWikiRecord) -> str:
    """The first listed answer that differs from the current one."""
    latest = rec.latest_answer.strip().lower()
    for ans in rec.all_answers:
        if ans.strip() and ans.strip().lower() != latest:
            return ans.strip()
    return rec.all_answers[0].strip() if rec.all_answers else rec.latest_answer


def _golden_context(rec: EvoWikiRecord) -> str:
    """The first retrieved chunk that mentions the current answer."""
    latest = rec.latest_answer.strip().lower()
    for chunk in rec.contriever_chunks + rec.bm25_chunks:
        if latest in chunk.lower():
            return chunk
    if rec.contriever_chunks:
        return rec.contriever_chunks[0]
    return rec.bm25_chunks[0] if rec.bm25_chunks else ""


def _entity_key(question: str) -> str:
    """Deterministic key so questions about one entity land in one split."""
    q = question.strip().lower()
    q = re.sub(r"\b(what|who|where|when|which|how|is|are|was|were|the|of|in|a|an)\b", "", q)
    q = re.sub(r"[^a-z0-9 ]", "", q)
    key = " ".join(sorted(set(q.split())))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _group_by_entity(records: Sequence[EvoWikiRecord]) -> dict[str, list[EvoWikiRecord]]:
    groups: dict[str, list[EvoWikiRecord]] = defaultdict(list)
    for r in records:
        groups[_entity_key(r.question)].append(r)
    return dict(groups)


def _split_keys(keys: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    n_train = int(len(keys) * train_ratio)
    return keys[:n_train], keys[n_train:]


def _write_jsonl(path: Path, records: list[SplitRecord]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return len(records)
