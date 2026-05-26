"""Grounded evaluation: fuzzy-match hallucinated relations to real Zenodo relations.

Key insight: the model generates near-correct relation names with small errors
(e.g., 'music.artist.albums' vs 'music.artist.album', 'book.author.works_written'
vs 'book.author.works_written'). Fuzzy matching bridges this gap without retraining.

Usage:
    python scripts/grounded_eval.py \
        --pred_file models/.../generated_predictions.jsonl \
        --zenodo_dir /workspace/data/zenodo/idirlab-freebases/FB+CVT+REV \
        --label_file /workspace/data/zenodo/idirlab-freebases/Metadata/entities_id_label.csv \
        --gold_file data/TempQuestions/generation/merged/TempQuestions_test.json \
        --output_file models/.../grounded_eval_results.json
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from difflib import get_close_matches
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.expr_parser import parse_s_expr, extract_entities, extract_relations


# — Zenodo loaders ——————————————————————

def load_relation_maps(zenodo_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
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
    mid_to_num: Dict[str, int] = {}
    num_to_mid: Dict[int, str] = {}
    ent_file = os.path.join(zenodo_dir, "entity2id.txt")
    print(f"  Loading entity2id.txt...")
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
    mid_to_label: Dict[str, str] = {}
    csv.field_size_limit(sys.maxsize)
    print(f"  Loading entity labels...")
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
                      f"{sum(len(v) for v in answers.values()):,} answers")
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


# — S-expression grounding ————————————

def parse_sexpr_components(sexpr: str) -> Optional[Tuple[str, str]]:
    try:
        parse_s_expr(sexpr)
    except Exception:
        return None
    try:
        entities = extract_entities(sexpr)
        relations = extract_relations(sexpr)
    except Exception:
        return None
    if not entities or not relations:
        return None
    return (relations[0], entities[0])


def dot_to_zenodo(dot_path: str) -> str:
    return "/" + dot_path.replace(".", "/")


def mid_to_zenodo(mid: str) -> str:
    if mid.startswith("m."):
        return "/m/" + mid[2:]
    if mid.startswith("g."):
        return "/g/" + mid[2:]
    return mid


def ground_relation(
    predicted_rel: str,
    zenodo_paths: List[str],
    cutoff: float = 0.6,
) -> Optional[str]:
    """Fuzzy-match a predicted relation to the closest real Zenodo relation.

    Tries both dot-format and slash-format matching.
    Returns the best-matching Zenodo slash-path, or None if no match >= cutoff.
    """
    # Try exact match first (with format conversion)
    slash_rel = dot_to_zenodo(predicted_rel)
    if slash_rel in zenodo_paths:
        return slash_rel
    if predicted_rel in zenodo_paths:
        return predicted_rel

    # Fuzzy match against all Zenodo relations
    # Use the last 2-3 segments for better matching (ignore common prefixes)
    pred_parts = predicted_rel.split(".")
    # Build a search key from distinctive parts
    search_key = predicted_rel

    matches = get_close_matches(slash_rel, zenodo_paths, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    # Try matching just the last 2 segments (more lenient)
    if len(pred_parts) >= 2:
        short_key = "/" + "/".join(pred_parts[-2:])
        # Filter candidates that end similarly
        candidates = [p for p in zenodo_paths if p.endswith("/" + pred_parts[-1])]
        if candidates:
            matches = get_close_matches("/" + "/".join(pred_parts[-3:]),
                                        candidates, n=1, cutoff=0.5)
            if matches:
                return matches[0]

    return None


def ground_entity(
    predicted_mid: str,
    zenodo_mids_set: Set[str],
) -> Optional[str]:
    """Try to find predicted MID in Zenodo entity set.

    Tries multiple format conversions.
    """
    # Direct match
    if predicted_mid in zenodo_mids_set:
        return predicted_mid
    # Zenodo slash format
    z_mid = mid_to_zenodo(predicted_mid)
    if z_mid in zenodo_mids_set:
        return z_mid
    return None


# — Main ————————————————————————————————

def main():
    parser = argparse.ArgumentParser(description="Grounded KBQA evaluation")
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--zenodo_dir", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--fuzzy_cutoff", type=float, default=0.6,
                        help="Minimum similarity for fuzzy relation match (0-1)")
    args = parser.parse_args()

    # — 1. Load Zenodo —
    print("=" * 60)
    print("STEP 1: Loading Zenodo")
    print("=" * 60)
    path_to_num, num_to_path = load_relation_maps(args.zenodo_dir)
    all_zenodo_rels = list(path_to_num.keys())
    print(f"  Relations: {len(path_to_num):,}")
    mid_to_num, num_to_mid = load_entity_maps(args.zenodo_dir)
    zenodo_mids_set: Set[str] = set(mid_to_num.keys())
    mid_to_label = load_entity_labels(args.label_file)

    # — 2. Load predictions —
    print("\nSTEP 2: Loading predictions")
    preds = []
    with open(args.pred_file) as f:
        for line in f:
            preds.append(json.loads(line))
    print(f"  Predictions: {len(preds)}")

    # — 3. Load gold —
    print("\nSTEP 3: Loading gold answers")
    with open(args.gold_file) as f:
        gold_items = json.load(f)
    print(f"  Gold entries: {len(gold_items)}")

    # — 4. Parse + Ground + Collect queries —
    print(f"\nSTEP 4: Parsing & Grounding (cutoff={args.fuzzy_cutoff})")
    query_pairs: List[Tuple[int, int]] = []
    parsed_preds: List[dict] = []

    valid_sexprs = 0
    grounded_rels = 0
    grounded_ents = 0
    grounded_both = 0
    grounding_failures: Dict[str, int] = defaultdict(int)

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
            pred_rel, pred_mid = components

            # Ground relation via fuzzy match
            real_rel = ground_relation(pred_rel, all_zenodo_rels, args.fuzzy_cutoff)
            if real_rel is None:
                grounding_failures[f"rel:{pred_rel.split('.')[-1]}"] += 1
                parsed["candidates"].append({
                    "sexpr": s_expr[:150],
                    "pred_rel": pred_rel,
                    "error": f"relation_not_grounded"
                })
                continue
            grounded_rels += 1

            # Ground entity
            real_mid = ground_entity(pred_mid, zenodo_mids_set)
            if real_mid is None:
                grounding_failures[f"ent:{pred_mid[:20]}"] += 1
                parsed["candidates"].append({
                    "sexpr": s_expr[:150],
                    "pred_rel": pred_rel, "grounded_rel": real_rel,
                    "pred_mid": pred_mid,
                    "error": "entity_not_grounded"
                })
                continue
            grounded_ents += 1
            grounded_both += 1

            rel_num = path_to_num[real_rel]
            ent_num = mid_to_num[real_mid]
            query_pairs.append((ent_num, rel_num))

            parsed["candidates"].append({
                "sexpr": s_expr,
                "pred_rel": pred_rel, "grounded_rel": real_rel,
                "pred_mid": pred_mid, "grounded_mid": real_mid,
                "rel_num": rel_num, "ent_num": ent_num,
            })

        parsed_preds.append(parsed)

    unique_queries = len(set(query_pairs))
    print(f"  Valid S-expressions: {valid_sexprs}")
    print(f"  Relations grounded: {grounded_rels}")
    print(f"  Entities grounded: {grounded_ents}")
    print(f"  Both grounded: {grounded_both}")
    print(f"  Unique (ent, rel) queries: {unique_queries}")
    if grounding_failures:
        print(f"  Top grounding failures:")
        for reason, count in sorted(grounding_failures.items(),
                                     key=lambda x: -x[1])[:10]:
            print(f"    {reason}: {count}")

    # — 5. Search triples —
    print(f"\nSTEP 5: Searching Zenodo triples")
    answer_index = build_answer_index(args.zenodo_dir, query_pairs, num_to_mid)
    print(f"  Answer patterns found: {len(answer_index)}")

    # — 6. Compute metrics —
    print("\nSTEP 6: Computing answer-level metrics")
    total = 0; f1_sum = 0.0; hits1_sum = 0; answered = 0
    results = []

    for entry in parsed_preds:
        question = entry["question"]
        q_clean = question.replace("Question: { ", "").replace(" }", "").strip()

        gold_entry = None
        for gitem in gold_items:
            if gitem.get("question", "").strip().lower() == q_clean.lower():
                gold_entry = gitem
                break
        if gold_entry is None:
            continue

        total += 1
        gold_answers = gold_entry.get("answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        pred_labels: List[str] = []
        best_sexpr = None
        best_grounded_rel = None
        for cand in entry["candidates"]:
            if "error" in cand:
                continue
            key = (cand["ent_num"], cand["rel_num"])
            ans_mids = answer_index.get(key, [])
            if ans_mids:
                pred_labels = [mid_to_label.get(mid, mid) for mid in ans_mids]
                best_sexpr = cand["sexpr"]
                best_grounded_rel = cand["grounded_rel"]
                break

        if pred_labels:
            answered += 1

        f1 = compute_f1(pred_labels, gold_answers)
        hits1 = compute_hits_at_1(pred_labels, gold_answers)
        f1_sum += f1
        hits1_sum += hits1

        results.append({
            "question": q_clean,
            "gold": gold_answers,
            "pred": pred_labels[:10],
            "best_sexpr": best_sexpr,
            "grounded_rel": best_grounded_rel,
            "f1": f1, "hits1": hits1,
        })

    # — 7. Report —
    print("\n" + "=" * 60)
    print("GROUNDED EVALUATION RESULTS")
    print("=" * 60)
    print(f"Fuzzy cutoff              : {args.fuzzy_cutoff}")
    print(f"Total questions           : {total}")
    print(f"Valid S-expressions       : {valid_sexprs}")
    print(f"Grounded (rel + ent)      : {grounded_both}")
    print(f"Questions with answers    : {answered} ({100*answered/total:.1f}%)" if total else "")
    print(f"F1 score                  : {f1_sum/total:.4f}" if total else "N/A")
    print(f"Hits@1                    : {hits1_sum/total:.4f}" if total else "N/A")
    print("=" * 60)

    if total and answered > 0:
        ans_f1 = sum(r["f1"] for r in results if r["pred"]) / max(1, answered)
        ans_h1 = sum(r["hits1"] for r in results if r["pred"]) / max(1, answered)
        print(f"\nOn {answered} answered questions:")
        print(f"  F1 (answered subset): {ans_f1:.4f}")
        print(f"  H@1 (answered subset): {ans_h1:.4f}")

        # Show some examples
        print(f"\nSample grounded predictions:")
        for r in results[:10]:
            if r["pred"]:
                print(f"  Q: {r['question'][:70]}")
                print(f"    Gold: {r['gold'][:3]}")
                print(f"    Pred: {r['pred'][:5]}")
                print(f"    F1: {r['f1']:.2f}")
                if r.get("grounded_rel"):
                    print(f"    Grounded rel: {r['grounded_rel']}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({
                "summary": {
                    "total": total,
                    "valid_sexprs": valid_sexprs,
                    "grounded_both": grounded_both,
                    "answered": answered,
                    "f1": f1_sum / total if total else 0,
                    "hits1": hits1_sum / total if total else 0,
                    "fuzzy_cutoff": args.fuzzy_cutoff,
                },
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
