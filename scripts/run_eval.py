#!/usr/bin/env python3
"""Unified evaluation harness for T-ChatKBQA.

Covers all evaluation dimensions in a single command:
  validity      — S-expression parse rate + operator distribution (offline)
  grounding     — Entity & relation grounding rates (requires Zenodo)
  answer        — Answer-level F1 / Hits@1 / Accuracy (requires Zenodo)
  temporal      — Temporal-subset metric breakdown
  all           — Everything possible given available data

Outputs: JSON report + Markdown summary.

Usage:
    # Offline — no external services needed
    python scripts/run_eval.py --pred_file preds.jsonl --mode validity

    # With Zenodo for grounding + answer eval
    python scripts/run_eval.py --pred_file preds.jsonl --mode all \
        --zenodo_dir /workspace/data/zenodo/idirlab-freebases/FB+CVT+REV \
        --label_file /workspace/data/zenodo/idirlab-freebases/Metadata/entities_id_label.csv \
        --gold_file data/TempQuestions/generation/merged/TempQuestions_test.json

    # Compare two runs
    python scripts/run_eval.py --pred_file run1.jsonl --mode all --output report_run1.md
    python scripts/run_eval.py --pred_file run2.jsonl --mode all --output report_run2.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Shared utilities ─────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return str(s).lower().strip().strip('"').strip("'")


def compute_f1(pred: List[str], gold: List[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    ps = {normalize(p) for p in pred}
    gs = {normalize(g) for g in gold}
    tp = len(ps & gs)
    if tp == 0:
        return 0.0
    return 2 * tp / (len(ps) + len(gs))


def compute_hits_at_1(pred: List[str], gold: List[str]) -> int:
    if not pred:
        return 0
    return int(normalize(pred[0]) in {normalize(g) for g in gold})


def compute_accuracy(pred: List[str], gold: List[str]) -> int:
    return int({normalize(p) for p in pred} == {normalize(g) for g in gold})


TEMPORAL_SIGNAL_RE = re.compile(
    r'\b(before|after|during|when|first|last|latest|earliest|most recent|'
    r'at the time|\d{4})\b',
    re.IGNORECASE,
)


def is_temporal_question(question: str) -> bool:
    return bool(TEMPORAL_SIGNAL_RE.search(question))


# ── S-expression analysis ────────────────────────────────────────────────

OPERATOR_RE = re.compile(r'\((JOIN|ARGMAX|ARGMIN|TC|AND|R|gt|ge|lt|le|COUNT|MAX|MIN)\b')


def detect_operator(sexpr: str) -> str:
    """Detect the primary operator of an S-expression."""
    if not sexpr or sexpr.lower() == "null":
        return "INVALID"
    # Find the outermost operator
    matches = OPERATOR_RE.findall(sexpr)
    if not matches:
        return "OTHER"
    # ARGMAX/ARGMIN take precedence over JOIN for temporal questions
    if "ARGMAX" in matches:
        return "ARGMAX"
    if "ARGMIN" in matches:
        return "ARGMIN"
    if "TC" in matches:
        return "TC"
    if "JOIN" in matches:
        return "JOIN"
    if matches[0] in ("gt", "ge", "lt", "le"):
        return "COMPARISON"
    return matches[0]


def parse_s_expr_safe(sexpr: str) -> bool:
    """Check if S-expression is parseable by ChatKBQA parser."""
    try:
        from components.expr_parser import parse_s_expr
        parse_s_expr(sexpr)
        return True
    except Exception:
        return False


def lisp_to_sparql_safe(sexpr: str) -> Optional[str]:
    """Try to convert S-expression to SPARQL. Returns SPARQL string or None."""
    try:
        from executor.logic_form_util import lisp_to_sparql
        sparql = lisp_to_sparql(sexpr)
        if sparql and "SELECT" in sparql.upper():
            return sparql
        return None
    except Exception:
        return None


def extract_sexpr_components(sexpr: str) -> Optional[Tuple[str, str]]:
    """Extract (relation_path, entity_mid) from an S-expression."""
    try:
        from components.expr_parser import parse_s_expr, extract_entities, extract_relations
        parse_s_expr(sexpr)
        entities = extract_entities(sexpr)
        relations = extract_relations(sexpr)
        if entities and relations:
            return (relations[0], entities[0])
        return None
    except Exception:
        return None


# ── Evaluation modes ─────────────────────────────────────────────────────

@dataclass
class ValidityResult:
    total_beams: int = 0
    valid_chatkbqa_parser: int = 0
    valid_lisp_to_sparql: int = 0
    invalid: int = 0
    operator_distribution: Counter = field(default_factory=Counter)
    per_question_stats: List[dict] = field(default_factory=list)


def eval_validity(preds: List[dict], top_k: int = 5) -> ValidityResult:
    """Evaluate S-expression validity and operator distribution (offline)."""
    result = ValidityResult()

    for entry in preds:
        question = entry.get("question", "")
        predictions = entry.get("predictions", [])[:top_k]
        q_valid_parser = 0
        q_valid_sparql = 0
        q_ops = []

        for sexpr in predictions:
            result.total_beams += 1
            if parse_s_expr_safe(sexpr):
                result.valid_chatkbqa_parser += 1
                q_valid_parser += 1
            if lisp_to_sparql_safe(sexpr):
                result.valid_lisp_to_sparql += 1
                q_valid_sparql += 1
            op = detect_operator(sexpr)
            result.operator_distribution[op] += 1
            q_ops.append(op)

        result.per_question_stats.append({
            "question": question,
            "num_beams": len(predictions),
            "valid_parser": q_valid_parser,
            "valid_sparql": q_valid_sparql,
            "operators": q_ops,
        })

    return result


@dataclass
class GroundingResult:
    total_valid: int = 0
    relations_exact: int = 0
    relations_fuzzy: int = 0
    entities_exact: int = 0
    entities_fuzzy: int = 0
    both_grounded: int = 0
    top_failures: List[Tuple[str, int]] = field(default_factory=list)
    per_question_stats: List[dict] = field(default_factory=list)


def _load_zenodo_if_needed(zenodo_dir: str, label_file: str):
    """Lazy-load Zenodo mapping tables. Returns (path_to_num, num_to_path,
    mid_to_num, num_to_mid, mid_to_label)."""
    import csv

    path_to_num: Dict[str, int] = {}
    num_to_path: Dict[int, str] = {}
    mid_to_num: Dict[str, int] = {}
    num_to_mid: Dict[int, str] = {}
    mid_to_label: Dict[str, str] = {}

    # Relations
    rel_file = os.path.join(zenodo_dir, "relation2id.txt")
    if os.path.isfile(rel_file):
        print(f"  Loading relation2id.txt...")
        with open(rel_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        path_to_num[parts[0]] = int(parts[1])
                        num_to_path[int(parts[1])] = parts[0]
                    except ValueError:
                        continue
        print(f"    {len(path_to_num):,} relations")

    # Entities
    ent_file = os.path.join(zenodo_dir, "entity2id.txt")
    if os.path.isfile(ent_file):
        print(f"  Loading entity2id.txt...")
        with open(ent_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        mid_to_num[parts[0]] = int(parts[1])
                        num_to_mid[int(parts[1])] = parts[0]
                    except ValueError:
                        continue
        print(f"    {len(mid_to_num):,} entities")

    # Labels
    if label_file and os.path.isfile(label_file):
        print(f"  Loading entity labels...")
        csv.field_size_limit(sys.maxsize)
        with open(label_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    mid_to_label[row[0]] = row[1]
        print(f"    {len(mid_to_label):,} labels")

    return path_to_num, num_to_path, mid_to_num, num_to_mid, mid_to_label


def dot_to_zenodo(dot_path: str) -> str:
    return "/" + dot_path.replace(".", "/")


def mid_to_zenodo(mid: str) -> str:
    if mid.startswith("m."):
        return "/m/" + mid[2:]
    if mid.startswith("g."):
        return "/g/" + mid[2:]
    return mid


def _ground_relation(predicted_rel: str, zenodo_paths: List[str], cutoff: float = 0.6) -> Tuple[Optional[str], str]:
    """Returns (matched_zenodo_path, method) where method is 'exact'/'fuzzy'/None."""
    slash_rel = dot_to_zenodo(predicted_rel)
    if slash_rel in zenodo_paths:
        return (slash_rel, "exact")
    if predicted_rel in zenodo_paths:
        return (predicted_rel, "exact")

    matches = get_close_matches(slash_rel, zenodo_paths, n=1, cutoff=cutoff)
    if matches:
        return (matches[0], "fuzzy")

    # Try suffix match on last segment
    pred_parts = predicted_rel.split(".")
    if len(pred_parts) >= 2:
        candidates = [p for p in zenodo_paths if p.endswith("/" + pred_parts[-1])]
        if candidates:
            matches = get_close_matches(
                "/" + "/".join(pred_parts[-3:]), candidates, n=1, cutoff=0.5
            )
            if matches:
                return (matches[0], "fuzzy")

    return (None, "none")


def _ground_entity(predicted_mid: str, zenodo_mids: Set[str]) -> Tuple[Optional[str], str]:
    """Returns (matched_mid, method)."""
    if predicted_mid in zenodo_mids:
        return (predicted_mid, "exact")
    z_mid = mid_to_zenodo(predicted_mid)
    if z_mid in zenodo_mids:
        return (z_mid, "exact")
    return (None, "none")


def eval_grounding(
    preds: List[dict],
    zenodo_dir: str,
    label_file: str,
    top_k: int = 5,
    fuzzy_cutoff: float = 0.6,
) -> GroundingResult:
    """Evaluate entity & relation grounding rates using Zenodo mappings."""
    (path_to_num, _, mid_to_num, _, _) = _load_zenodo_if_needed(zenodo_dir, label_file)
    all_zenodo_rels = list(path_to_num.keys())
    zenodo_mids: Set[str] = set(mid_to_num.keys())

    result = GroundingResult()
    failure_counter: Counter = Counter()

    for entry in preds:
        question = entry.get("question", "")
        predictions = entry.get("predictions", [])[:top_k]
        q_stats = {"question": question, "candidates": []}

        for sexpr in predictions:
            components = extract_sexpr_components(sexpr)
            if components is None:
                q_stats["candidates"].append({"sexpr": sexpr[:120], "error": "parse_failed"})
                continue

            result.total_valid += 1
            pred_rel, pred_mid = components

            real_rel, rel_method = _ground_relation(pred_rel, all_zenodo_rels, fuzzy_cutoff)
            real_mid, ent_method = _ground_entity(pred_mid, zenodo_mids)

            if real_rel:
                if rel_method == "exact":
                    result.relations_exact += 1
                else:
                    result.relations_fuzzy += 1
            else:
                failure_counter[f"rel:{pred_rel.split('.')[-1][:40]}"] += 1

            if real_mid:
                if ent_method == "exact":
                    result.entities_exact += 1
            else:
                failure_counter[f"ent:{pred_mid[:30]}"] += 1

            if real_rel and real_mid:
                result.both_grounded += 1

            q_stats["candidates"].append({
                "sexpr": sexpr[:150],
                "pred_rel": pred_rel,
                "pred_mid": pred_mid,
                "grounded_rel": real_rel,
                "grounded_mid": real_mid,
                "rel_method": rel_method,
                "ent_method": ent_method,
            })

        result.per_question_stats.append(q_stats)

    result.top_failures = failure_counter.most_common(15)
    return result


@dataclass
class AnswerResult:
    total: int = 0
    f1_sum: float = 0.0
    hits1_sum: float = 0.0
    acc_sum: float = 0.0
    answered: int = 0
    temporal_total: int = 0
    temporal_f1: float = 0.0
    temporal_hits1: float = 0.0
    temporal_answered: int = 0
    per_question_results: List[dict] = field(default_factory=list)


def eval_answers(
    preds: List[dict],
    zenodo_dir: str,
    label_file: str,
    gold_file: str,
    top_k: int = 5,
    fuzzy_cutoff: float = 0.6,
) -> AnswerResult:
    """Answer-level F1/Hits@1/Accuracy with Zenodo triple streaming + fuzzy grounding."""
    (path_to_num, num_to_path, mid_to_num, num_to_mid, mid_to_label) = \
        _load_zenodo_if_needed(zenodo_dir, label_file)
    all_zenodo_rels = list(path_to_num.keys())
    zenodo_mids: Set[str] = set(mid_to_num.keys())

    # Load gold
    with open(gold_file) as f:
        gold_items = json.load(f)
    print(f"  Gold entries: {len(gold_items)}")

    # Phase 1: Parse + Ground + Collect queries
    print(f"  Parsing & grounding predictions...")
    query_pairs: List[Tuple[int, int]] = []
    parsed_preds: List[dict] = []

    for entry in preds:
        question = entry.get("question", "")
        predictions = entry.get("predictions", [])[:top_k]
        parsed = {"question": question, "candidates": []}

        for sexpr in predictions:
            components = extract_sexpr_components(sexpr)
            if components is None:
                parsed["candidates"].append({"sexpr": sexpr[:120], "error": "parse_failed"})
                continue

            pred_rel, pred_mid = components
            real_rel, _ = _ground_relation(pred_rel, all_zenodo_rels, fuzzy_cutoff)
            real_mid, _ = _ground_entity(pred_mid, zenodo_mids)

            if real_rel is None or real_mid is None:
                parsed["candidates"].append({
                    "sexpr": sexpr[:120],
                    "error": f"not_grounded (rel={real_rel is None}, ent={real_mid is None})"
                })
                continue

            rel_num = path_to_num[real_rel]
            ent_num = mid_to_num[real_mid]
            query_pairs.append((ent_num, rel_num))

            parsed["candidates"].append({
                "sexpr": sexpr,
                "grounded_rel": real_rel,
                "grounded_mid": real_mid,
                "rel_num": rel_num,
                "ent_num": ent_num,
            })

        parsed_preds.append(parsed)

    unique_q = len(set(query_pairs))
    print(f"  Unique (entity, relation) queries: {unique_q}")

    # Phase 2: Stream triples to build answer index
    print(f"  Streaming train.txt for {unique_q} patterns...")
    query_set: Set[Tuple[int, int]] = set(query_pairs)
    answers: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

    train_file = os.path.join(zenodo_dir, "train.txt")
    if os.path.isfile(train_file):
        with open(train_file) as f:
            for i, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                try:
                    head, rel, tail = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                key = (head, rel)
                if key in query_set:
                    answers[key].add(num_to_mid.get(tail, f"m.{tail}"))
                if (i + 1) % 50_000_000 == 0:
                    print(f"    {i + 1:,} triples, "
                          f"{sum(len(v) for v in answers.values()):,} answers")
        answer_index = {k: sorted(v) for k, v in answers.items()}
    else:
        print(f"  WARNING: train.txt not found — answer index empty")
        answer_index = {}

    # Phase 3: Compute metrics vs gold
    print(f"  Computing answer-level metrics...")
    result = AnswerResult()
    temporal_total = 0
    temporal_f1 = 0.0
    temporal_hits1 = 0.0
    temporal_answered = 0

    for entry in parsed_preds:
        question = entry["question"]
        q_clean = question.replace("Question: { ", "").replace(" }", "").strip()

        gold_entry = None
        for gitem in gold_items:
            gq = gitem.get("question", "").strip()
            if q_clean.lower() == gq.lower():
                gold_entry = gitem
                break
        if gold_entry is None:
            continue

        result.total += 1
        gold_answers = gold_entry.get("answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        pred_labels: List[str] = []
        best_sexpr = None
        for cand in entry["candidates"]:
            if "error" in cand:
                continue
            key = (cand["ent_num"], cand["rel_num"])
            ans_mids = answer_index.get(key, [])
            if ans_mids:
                pred_labels = [mid_to_label.get(mid, mid) for mid in ans_mids]
                best_sexpr = cand["sexpr"]
                break

        if pred_labels:
            result.answered += 1

        f1 = compute_f1(pred_labels, gold_answers)
        hits1 = compute_hits_at_1(pred_labels, gold_answers)
        acc = compute_accuracy(pred_labels, gold_answers)

        result.f1_sum += f1
        result.hits1_sum += hits1
        result.acc_sum += acc

        temporal = is_temporal_question(q_clean)
        if temporal:
            temporal_total += 1
            temporal_f1 += f1
            temporal_hits1 += hits1
            if pred_labels:
                temporal_answered += 1

        result.per_question_results.append({
            "question": q_clean,
            "is_temporal": temporal,
            "gold": gold_answers,
            "pred": pred_labels[:10],
            "best_sexpr": best_sexpr,
            "f1": f1,
            "hits1": hits1,
            "accuracy": acc,
        })

    result.temporal_total = temporal_total
    result.temporal_f1 = temporal_f1
    result.temporal_hits1 = temporal_hits1
    result.temporal_answered = temporal_answered
    return result


# ── Report formatting ────────────────────────────────────────────────────

def _pct(numerator: float, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{100 * numerator / denominator:.1f}%"


def format_report(
    validity: Optional[ValidityResult] = None,
    grounding: Optional[GroundingResult] = None,
    answers: Optional[AnswerResult] = None,
    pred_file: str = "",
    mode: str = "all",
) -> str:
    """Generate a Markdown evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# T-ChatKBQA Evaluation Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Prediction file:** `{pred_file}`  ",
        f"**Mode:** `{mode}`  ",
        f"",
        "---",
        "",
    ]

    if validity:
        t = validity.total_beams
        lines += [
            "## 1. S-expression Validity",
            "",
            f"| Metric | Count | Rate |",
            f"|--------|-------|------|",
            f"| Total beams | {t} | — |",
            f"| Valid (ChatKBQA parser) | {validity.valid_chatkbqa_parser} | {_pct(validity.valid_chatkbqa_parser, t)} |",
            f"| Valid (lisp_to_sparql) | {validity.valid_lisp_to_sparql} | {_pct(validity.valid_lisp_to_sparql, t)} |",
            f"| Invalid | {validity.invalid if validity.invalid else t - validity.valid_chatkbqa_parser} | {_pct(t - validity.valid_chatkbqa_parser, t)} |",
            f"",
            "### Operator Distribution",
            "",
            "| Operator | Count | Share |",
            "|----------|-------|-------|",
        ]
        for op in ["ARGMAX", "ARGMIN", "JOIN", "TC", "COMPARISON", "AND", "OTHER", "INVALID"]:
            count = validity.operator_distribution.get(op, 0)
            if count > 0:
                lines.append(f"| {op} | {count} | {_pct(count, t)} |")
        lines.append("")
        # Validity at question level
        questions_with_any_valid = sum(
            1 for s in validity.per_question_stats if s["valid_parser"] > 0
        )
        n_questions = len(validity.per_question_stats)
        lines += [
            "### Question-level Validity",
            "",
            f"- Questions with ≥1 valid beam: {questions_with_any_valid}/{n_questions} ({_pct(questions_with_any_valid, n_questions)})",
            f"- Questions with 0 valid beams: {n_questions - questions_with_any_valid}/{n_questions}",
            "",
        ]

    if grounding:
        t = grounding.total_valid
        lines += [
            "## 2. Entity & Relation Grounding",
            "",
            f"| Metric | Count | Rate |",
            f"|--------|-------|------|",
            f"| Parseable S-expressions | {t} | — |",
            f"| Relations grounded (exact) | {grounding.relations_exact} | {_pct(grounding.relations_exact, t)} |",
            f"| Relations grounded (fuzzy) | {grounding.relations_fuzzy} | {_pct(grounding.relations_fuzzy, t)} |",
            f"| Relations grounded (total) | {grounding.relations_exact + grounding.relations_fuzzy} | {_pct(grounding.relations_exact + grounding.relations_fuzzy, t)} |",
            f"| Entities grounded (exact) | {grounding.entities_exact} | {_pct(grounding.entities_exact, t)} |",
            f"| Both grounded | {grounding.both_grounded} | {_pct(grounding.both_grounded, t)} |",
            f"",
        ]
        if grounding.top_failures:
            lines += [
                "### Top Grounding Failures",
                "",
                "| Pattern | Count |",
                "|---------|-------|",
            ]
            for reason, count in grounding.top_failures[:10]:
                lines.append(f"| `{reason}` | {count} |")
            lines.append("")

    if answers:
        t = answers.total
        lines += [
            "## 3. Answer-level Metrics (vs Gold)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total questions evaluated | {t} |",
            f"| Questions with answers found in KB | {answers.answered} ({_pct(answers.answered, t)}) |",
            f"| **F1 score** | **{answers.f1_sum / t:.4f}**" if t else "| F1 | N/A |",
            f"| **Hits@1** | **{answers.hits1_sum / t:.4f}**" if t else "",
            f"| **Accuracy** | **{answers.acc_sum / t:.4f}**" if t else "",
            f"",
        ]
        if answers.answered > 0:
            answered = answers.answered
            ans_f1 = sum(
                r["f1"] for r in answers.per_question_results if r["pred"]
            ) / max(1, answered)
            ans_h1 = sum(
                r["hits1"] for r in answers.per_question_results if r["pred"]
            ) / max(1, answered)
            lines += [
                f"### Answered-only Metrics",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| F1 (answered subset) | {ans_f1:.4f} |",
                f"| Hits@1 (answered subset) | {ans_h1:.4f} |",
                f"",
            ]

        if answers.temporal_total:
            tt = answers.temporal_total
            tf1 = answers.temporal_f1 / tt if tt else 0
            th1 = answers.temporal_hits1 / tt if tt else 0
            lines += [
                "## 4. Temporal Subset",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Temporal questions | {tt} ({_pct(tt, t)}) |",
                f"| Temporal F1 | {tf1:.4f} |",
                f"| Temporal Hits@1 | {th1:.4f} |",
                f"| Temporal answered | {answers.temporal_answered} ({_pct(answers.temporal_answered, tt)}) |",
                f"",
            ]

    lines += [
        "---",
        f"*Report generated by `scripts/run_eval.py` — {now}*",
    ]
    return "\n".join(lines)


def to_json_report(
    validity: Optional[ValidityResult] = None,
    grounding: Optional[GroundingResult] = None,
    answers: Optional[AnswerResult] = None,
    pred_file: str = "",
    mode: str = "all",
) -> dict:
    """Generate a structured JSON report."""
    report = {
        "meta": {
            "generated": datetime.now().isoformat(),
            "pred_file": pred_file,
            "mode": mode,
        },
    }

    if validity:
        t = validity.total_beams
        questions_with_any = sum(
            1 for s in validity.per_question_stats if s["valid_parser"] > 0
        )
        report["validity"] = {
            "total_beams": t,
            "valid_chatkbqa_parser": validity.valid_chatkbqa_parser,
            "valid_rate_parser": round(validity.valid_chatkbqa_parser / t, 4) if t else 0,
            "valid_lisp_to_sparql": validity.valid_lisp_to_sparql,
            "valid_rate_sparql": round(validity.valid_lisp_to_sparql / t, 4) if t else 0,
            "operator_distribution": dict(validity.operator_distribution),
            "questions_any_valid": questions_with_any,
            "questions_total": len(validity.per_question_stats),
        }

    if grounding:
        t = grounding.total_valid
        report["grounding"] = {
            "total_valid": t,
            "relations_exact": grounding.relations_exact,
            "relations_fuzzy": grounding.relations_fuzzy,
            "relations_total": grounding.relations_exact + grounding.relations_fuzzy,
            "relations_rate": round((grounding.relations_exact + grounding.relations_fuzzy) / t, 4) if t else 0,
            "entities_exact": grounding.entities_exact,
            "entities_rate": round(grounding.entities_exact / t, 4) if t else 0,
            "both_grounded": grounding.both_grounded,
            "both_rate": round(grounding.both_grounded / t, 4) if t else 0,
            "top_failures": [{"pattern": p, "count": c} for p, c in grounding.top_failures],
        }

    if answers:
        t = answers.total
        report["answers"] = {
            "total": t,
            "answered": answers.answered,
            "answered_rate": round(answers.answered / t, 4) if t else 0,
            "f1": round(answers.f1_sum / t, 4) if t else 0,
            "hits1": round(answers.hits1_sum / t, 4) if t else 0,
            "accuracy": round(answers.acc_sum / t, 4) if t else 0,
        }
        if answers.temporal_total:
            tt = answers.temporal_total
            report["temporal"] = {
                "total": tt,
                "f1": round(answers.temporal_f1 / tt, 4) if tt else 0,
                "hits1": round(answers.temporal_hits1 / tt, 4) if tt else 0,
                "answered": answers.temporal_answered,
            }

    return report


# ── Main ─────────────────────────────────────────────────────────────────

def load_predictions(pred_file: str) -> List[dict]:
    """Load predictions from JSONL or JSON file."""
    preds = []
    with open(pred_file) as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.strip().startswith("{"):
            # JSONL — one JSON object per line
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))
        else:
            # JSON — single object or list
            data = json.load(f)
            if isinstance(data, dict):
                # beam_test_top_k_predictions.json format: {qid: {predictions: [...]}}
                for qid, entry in data.items():
                    if isinstance(entry, dict) and "predictions" in entry:
                        preds.append(entry)
                    elif isinstance(entry, list):
                        preds.append({"question": qid, "predictions": entry})
            elif isinstance(data, list):
                preds = data
    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Unified T-ChatKBQA Evaluation Harness"
    )
    parser.add_argument("--pred_file", required=True, help="Path to predictions JSONL/JSON")
    parser.add_argument("--mode", default="all",
                        choices=["validity", "grounding", "answer", "all"],
                        help="Evaluation mode (default: all)")
    parser.add_argument("--top_k", type=int, default=5, help="Beams to consider")
    parser.add_argument("--fuzzy_cutoff", type=float, default=0.6,
                        help="Minimum similarity for fuzzy relation matching")
    parser.add_argument("--zenodo_dir", default=None, help="Path to Zenodo dataset")
    parser.add_argument("--label_file", default=None, help="Path to entities_id_label.csv")
    parser.add_argument("--gold_file", default=None, help="Path to gold answers JSON")
    parser.add_argument("--output", default=None, help="Output JSON report path")
    parser.add_argument("--report", default=None, help="Output Markdown report path")
    args = parser.parse_args()

    # Check pred_file exists
    if not os.path.isfile(args.pred_file):
        print(f"ERROR: Prediction file not found: {args.pred_file}")
        sys.exit(1)

    print("=" * 60)
    print("T-ChatKBQA Evaluation Harness")
    print("=" * 60)
    print(f"Prediction file: {args.pred_file}")
    print(f"Mode: {args.mode}")
    print()

    # Load predictions
    preds = load_predictions(args.pred_file)
    print(f"Loaded {len(preds)} prediction entries")

    validity_result = None
    grounding_result = None
    answer_result = None

    # ── Run modes ──

    if args.mode in ("validity", "all"):
        print("\n" + "-" * 40)
        print("MODE: Validity + Operator Distribution")
        print("-" * 40)
        validity_result = eval_validity(preds, top_k=args.top_k)
        t = validity_result.total_beams
        print(f"  Valid (parser):    {validity_result.valid_chatkbqa_parser}/{t} "
              f"({_pct(validity_result.valid_chatkbqa_parser, t)})")
        print(f"  Valid (sparql):    {validity_result.valid_lisp_to_sparql}/{t} "
              f"({_pct(validity_result.valid_lisp_to_sparql, t)})")
        print(f"  Operator distribution:")
        for op, count in validity_result.operator_distribution.most_common():
            print(f"    {op:12s}: {count:5d}  ({_pct(count, t)})")
        questions_any = sum(
            1 for s in validity_result.per_question_stats if s["valid_parser"] > 0
        )
        print(f"  Questions with ≥1 valid: {questions_any}/{len(validity_result.per_question_stats)}")

    if args.mode in ("grounding", "all"):
        if not args.zenodo_dir:
            print("\n[MODE: grounding] SKIP — --zenodo_dir not provided")
        else:
            print("\n" + "-" * 40)
            print("MODE: Entity & Relation Grounding")
            print("-" * 40)
            grounding_result = eval_grounding(
                preds, args.zenodo_dir, args.label_file or "",
                top_k=args.top_k, fuzzy_cutoff=args.fuzzy_cutoff,
            )
            t = grounding_result.total_valid
            r_total = grounding_result.relations_exact + grounding_result.relations_fuzzy
            print(f"  Relations: {r_total}/{t} ({_pct(r_total, t)}) "
                  f"[exact: {grounding_result.relations_exact}, fuzzy: {grounding_result.relations_fuzzy}]")
            print(f"  Entities:  {grounding_result.entities_exact}/{t} ({_pct(grounding_result.entities_exact, t)})")
            print(f"  Both:      {grounding_result.both_grounded}/{t} ({_pct(grounding_result.both_grounded, t)})")

    if args.mode in ("answer", "all"):
        if not args.zenodo_dir or not args.gold_file:
            print("\n[MODE: answer] SKIP — --zenodo_dir and/or --gold_file not provided")
        else:
            print("\n" + "-" * 40)
            print("MODE: Answer-level Evaluation")
            print("-" * 40)
            answer_result = eval_answers(
                preds, args.zenodo_dir, args.label_file or "",
                args.gold_file, top_k=args.top_k, fuzzy_cutoff=args.fuzzy_cutoff,
            )
            t = answer_result.total
            print(f"  Total:      {t}")
            print(f"  Answered:   {answer_result.answered} ({_pct(answer_result.answered, t)})")
            print(f"  F1:         {answer_result.f1_sum / t:.4f}" if t else "  F1: N/A")
            print(f"  Hits@1:     {answer_result.hits1_sum / t:.4f}" if t else "")
            print(f"  Accuracy:   {answer_result.acc_sum / t:.4f}" if t else "")
            if answer_result.temporal_total:
                tt = answer_result.temporal_total
                print(f"  Temporal:   {tt} questions")
                print(f"    F1:       {answer_result.temporal_f1 / tt:.4f}" if tt else "")
                print(f"    Hits@1:   {answer_result.temporal_hits1 / tt:.4f}" if tt else "")

    # ── Output ──

    json_report = to_json_report(
        validity=validity_result,
        grounding=grounding_result,
        answers=answer_result,
        pred_file=args.pred_file,
        mode=args.mode,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to: {args.output}")

    md_report = format_report(
        validity=validity_result,
        grounding=grounding_result,
        answers=answer_result,
        pred_file=args.pred_file,
        mode=args.mode,
    )

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write(md_report)
        print(f"Markdown report saved to: {args.report}")
    else:
        print("\n" + md_report)


if __name__ == "__main__":
    main()
