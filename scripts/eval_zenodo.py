"""Evaluate T-ChatKBQA using Zenodo triple files + entity labels.

Uses ChatKBQA's own S-expression parser (parse_s_expr, extract_entities,
extract_relations) to extract (relation, entity) pairs, then resolves
them against Zenodo FB+CVT+REV triples to get answer MIDs, converts to
human-readable labels, and compares with TempQuestions gold answers.

Usage:
    python scripts/eval_zenodo.py \
        --pred_file models/.../generated_predictions.jsonl \
        --zenodo_dir /workspace/data/zenodo/idirlab-freebases/FB+CVT+REV \
        --label_file /workspace/data/zenodo/idirlab-freebases/Metadata/entities_id_label.csv \
        --gold_file data/TempQuestions/generation/merged/TempQuestions_test.json \
        --output_file models/.../zenodo_eval_results.json
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from components.expr_parser import parse_s_expr, extract_entities, extract_relations


# — Zenodo loaders ——————————————————————

def load_relation_maps(zenodo_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """relation2id.txt (format: path,id)."""
    path_to_num: Dict[str, int] = {}
    num_to_path: Dict[int, str] = {}
    with open(os.path.join(zenodo_dir, "relation2id.txt")) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    path_to_num[parts[0]] = int(parts[1])
                    num_to_path[int(parts[1])] = parts[0]
                except ValueError:
                    continue
    return path_to_num, num_to_path


def load_entity_maps(zenodo_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """entity2id.txt (format: mid,id)."""
    mid_to_num: Dict[str, int] = {}
    num_to_mid: Dict[int, str] = {}
    ent_file = os.path.join(zenodo_dir, "entity2id.txt")
    print(f"  Loading entity2id.txt (~2.4GB)...")
    with open(ent_file) as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    mid_to_num[parts[0]] = int(parts[1])
                    num_to_mid[int(parts[1])] = parts[0]
                except ValueError:
                    continue
            if (i + 1) % 10_000_000 == 0:
                print(f"    {i + 1:,} entities...")
    print(f"  Loaded {len(mid_to_num):,} entities")
    return mid_to_num, num_to_mid


def load_entity_labels(label_file: str) -> Dict[str, str]:
    """entities_id_label.csv (format: mid,label,type_id)."""
    mid_to_label: Dict[str, str] = {}
    csv.field_size_limit(sys.maxsize)
    print(f"  Loading entity labels (~1.8GB)...")
    with open(label_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) >= 2:
                mid_to_label[row[0]] = row[1]
            if (i + 1) % 10_000_000 == 0:
                print(f"    {i + 1:,} labels...")
    print(f"  Loaded {len(mid_to_label):,} labels")
    return mid_to_label


def build_answer_index(
    zenodo_dir: str,
    queries: List[Tuple[int, int]],
    num_to_mid: Dict[int, str],
) -> Dict[Tuple[int, int], List[str]]:
    """Stream train.txt, collect tail MIDs for (head_num, rel_num) pairs."""
    query_set: Set[Tuple[int, int]] = set(queries)
    answers: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

    train_file = os.path.join(zenodo_dir, "train.txt")
    print(f"  Streaming train.txt (4.5GB) for {len(query_set)} patterns...")
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
                      f"{sum(len(v) for v in answers.values()):,} answers found")
    return {k: sorted(v) for k, v in answers.items()}


# — Metrics ————————————————————————————

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


# — S-expression parsing using ChatKBQA parser ——

def parse_sexpr_components(sexpr: str) -> Optional[Tuple[str, str]]:
    """Extract (relation_path, entity_mid) from S-expression.

    Uses ChatKBQA's parse_s_expr for validation, then extract_entities
    and extract_relations to get the components needed for Zenodo lookup.
    """
    try:
        parse_s_expr(sexpr)  # validate
    except Exception:
        return None

    try:
        entities = extract_entities(sexpr)
        relations = extract_relations(sexpr)
    except Exception:
        return None

    if not entities or not relations:
        return None

    # Take the first entity (head/topic entity) and first relation
    entity_mid = entities[0]
    relation_path = relations[0]

    return (relation_path, entity_mid)


# — Main ————————————————————————————————

def main():
    parser = argparse.ArgumentParser(description="Zenodo-based KBQA evaluation")
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--zenodo_dir", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # — 1. Load Zenodo mappings —
    print("=" * 60)
    print("STEP 1: Loading Zenodo mapping tables")
    print("=" * 60)
    path_to_num, num_to_path = load_relation_maps(args.zenodo_dir)
    print(f"  Relations: {len(path_to_num):,}")
    mid_to_num, num_to_mid = load_entity_maps(args.zenodo_dir)
    mid_to_label = load_entity_labels(args.label_file)

    # — 2. Load predictions —
    print("\nSTEP 2: Loading predictions")
    preds = []
    with open(args.pred_file) as f:
        for line in f:
            preds.append(json.loads(line))
    print(f"  Predictions: {len(preds)}")

    # — 3. Load gold answers —
    print("\nSTEP 3: Loading gold answers")
    with open(args.gold_file) as f:
        gold_items = json.load(f)
    print(f"  Gold entries: {len(gold_items)}")

    # — 4. Parse S-expressions & collect query patterns —
    print("\nSTEP 4: Parsing S-expressions with ChatKBQA parser")
    query_pairs: List[Tuple[int, int]] = []
    parsed_preds: List[dict] = []

    valid_sexprs = 0
    has_zenodo_match = 0

    # Helper: convert ChatKBQA dot-path to Zenodo slash-path
    def dot_to_zenodo(dot_path: str) -> str:
        return "/" + dot_path.replace(".", "/")

    # Helper: convert ChatKBQA MID to Zenodo MID format
    # m.xxx → /m/xxx  (Zenodo uses /m/ prefix with slashes)
    def mid_to_zenodo(mid: str) -> str:
        if mid.startswith("m."):
            return "/m/" + mid[2:]
        if mid.startswith("g."):
            return "/g/" + mid[2:]
        return mid

    for entry in preds:
        question = entry.get("question", "")
        predictions = entry.get("predictions", [])
        parsed = {"question": question, "candidates": []}

        for s_expr in predictions[:args.top_k]:
            components = parse_sexpr_components(s_expr)
            if components is None:
                parsed["candidates"].append({
                    "sexpr": s_expr[:150], "error": "parse_failed"
                })
                continue

            valid_sexprs += 1
            rel_path, ent_mid = components

            # Try both original dot format and converted slash format
            zenodo_rel = dot_to_zenodo(rel_path)
            rel_num = path_to_num.get(rel_path) or path_to_num.get(zenodo_rel)
            zenodo_mid = mid_to_zenodo(ent_mid)
            ent_num = mid_to_num.get(ent_mid) or mid_to_num.get(zenodo_mid)

            if rel_num is None or ent_num is None:
                parsed["candidates"].append({
                    "sexpr": s_expr[:150],
                    "rel_path": rel_path,
                    "ent_mid": ent_mid,
                    "error": f"not_in_zenodo (rel={rel_num is None}, ent={ent_num is None})"
                })
                continue

            has_zenodo_match += 1
            query_pairs.append((ent_num, rel_num))
            parsed["candidates"].append({
                "sexpr": s_expr,
                "rel_path": rel_path,
                "ent_mid": ent_mid,
                "rel_num": rel_num,
                "ent_num": ent_num,
            })

        parsed_preds.append(parsed)

    unique_queries = len(set(query_pairs))
    print(f"  Valid S-expressions (any beam): {valid_sexprs}")
    print(f"  With Zenodo mappings: {has_zenodo_match}")
    print(f"  Unique (entity, relation) queries: {unique_queries}")

    # — 5. Search triples —
    print(f"\nSTEP 5: Searching Zenodo triples")
    answer_index = build_answer_index(args.zenodo_dir, query_pairs, num_to_mid)
    print(f"  Answer patterns found: {len(answer_index)}")

    # — 6. Convert MIDs → labels, compute metrics —
    print("\nSTEP 6: Computing answer-level metrics vs gold")
    total = 0
    f1_sum = 0.0
    hits1_sum = 0
    acc_sum = 0
    answered = 0
    parseable = 0
    results = []

    for entry in parsed_preds:
        question = entry["question"]
        q_clean = question.replace("Question: { ", "").replace(" }", "").strip()

        # Match gold by question text
        gold_entry = None
        for gitem in gold_items:
            gq = gitem.get("question", "").strip()
            if q_clean.lower() == gq.lower():
                gold_entry = gitem
                break
        if gold_entry is None:
            continue

        total += 1
        gold_answers = gold_entry.get("answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        has_parseable = any("error" not in c for c in entry["candidates"])
        if has_parseable:
            parseable += 1

        # Try each candidate until one yields answers
        pred_labels: List[str] = []
        best_sexpr = None
        for cand in entry["candidates"]:
            if "error" in cand:
                continue
            key = (cand["ent_num"], cand["rel_num"])
            ans_mids = answer_index.get(key, [])
            if ans_mids:
                labels = [mid_to_label.get(mid, mid) for mid in ans_mids]
                pred_labels = labels
                best_sexpr = cand["sexpr"]
                break

        if pred_labels:
            answered += 1

        f1 = compute_f1(pred_labels, gold_answers)
        hits1 = compute_hits_at_1(pred_labels, gold_answers)
        acc = compute_accuracy(pred_labels, gold_answers)

        f1_sum += f1
        hits1_sum += hits1
        acc_sum += acc

        results.append({
            "question": q_clean,
            "gold": gold_answers,
            "pred": pred_labels[:10],
            "best_sexpr": best_sexpr,
            "f1": f1, "hits1": hits1, "accuracy": acc,
        })

    # — 7. Report —
    print("\n" + "=" * 60)
    print("ZENODO-BASED ANSWER-LEVEL EVALUATION")
    print("=" * 60)
    print(f"Total questions evaluated  : {total}")
    print(f"Parseable S-expressions    : {parseable} ({100*parseable/total:.1f}%)" if total else "")
    print(f"Questions with Zenodo ans  : {answered} ({100*answered/total:.1f}%)" if total else "")
    print(f"F1 score                   : {f1_sum/total:.4f}" if total else "N/A")
    print(f"Hits@1                     : {hits1_sum/total:.4f}" if total else "N/A")
    print(f"Accuracy                   : {acc_sum/total:.4f}" if total else "N/A")
    print("=" * 60)
    if total:
        print(f"\nError breakdown:")
        print(f"  Parse failed:   {total - parseable}")
        print(f"  No Zenodo data: {parseable - answered}")
        print(f"  Has answers:    {answered}")
        if answered > 0:
            print(f"\n  On {answered} questions with Zenodo answers:")
            ans_f1 = sum(r["f1"] for r in results if r["pred"]) / max(1, answered)
            ans_h1 = sum(r["hits1"] for r in results if r["pred"]) / max(1, answered)
            print(f"  F1 (answered subset): {ans_f1:.4f}")
            print(f"  H@1 (answered subset): {ans_h1:.4f}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({
                "summary": {
                    "total": total,
                    "parseable": parseable,
                    "answered": answered,
                    "f1": f1_sum / total if total else 0,
                    "hits1": hits1_sum / total if total else 0,
                    "accuracy": acc_sum / total if total else 0,
                },
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
