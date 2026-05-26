#!/usr/bin/env python3
"""Build canonical temporal dataset artifacts for T-ChatKBQA.

New pipeline (primary):
  1. mine-facts          — SPARQL mining from Freebase Virtuoso
  2. generate-candidates — Template-based question + S-expression generation
  3. verify-samples      — Execute S-expressions, compare answers
  4. filter-and-export   — Dedup, cap, balance, export training data

Legacy commands (deprecated, kept for backward compat):
  standardize-tempquestions, build-relation-inventory,
  build-fact-mining-manifest, build-remote-jobs, run-remote-mining,
  build-synthetic-samples, filter-synthetic-samples, export-training-examples,
  audit-merged-tempquestions
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.utils import dump_json, load_json, mkdir_p
from src.temporal_data import (
    MinedTemporalFact,
    TemporalSample,
    apply_distribution_filter,
    build_temporal_training_examples,
    generate_candidates,
    mine_facts_from_zenodo,
    mine_facts_raw,
    summarize_candidates,
    summarize_distribution,
    summarize_mined_facts,
    summarize_training_examples,
    summarize_verification,
    verify_samples,
    load_whitelist,
)

# ── New Pipeline Commands ─────────────────────────────────────────────

def _cmd_mine_facts(args: argparse.Namespace) -> None:
    """Mine temporal facts from Freebase Virtuoso."""
    whitelist_path = args.whitelist or "configs/relation_whitelist.yaml"
    output_path = Path(args.output or "data/temporal/mined_facts.json")
    summary_path = Path(args.summary or "data/temporal/mined_facts.summary.json")

    if not Path(whitelist_path).exists():
        print(f"ERROR: Whitelist config not found at {whitelist_path}")
        sys.exit(1)

    print(f"Mining facts using whitelist: {whitelist_path}")
    facts = mine_facts_raw(
        whitelist_path=whitelist_path,
        limit_per_family=args.limit,
        family_filter=args.families.split(",") if args.families else None,
    )

    mkdir_p(str(output_path.parent))
    dump_json([f.to_dict() for f in facts], str(output_path), indent=2)
    dump_json(summarize_mined_facts(facts), str(summary_path), indent=2)

    print(f"Wrote {len(facts)} mined facts to {output_path}")
    print(f"Wrote summary to {summary_path}")


def _cmd_mine_facts_zenodo(args: argparse.Namespace) -> None:
    """Mine temporal facts from Zenodo idirlab/freebases dataset."""
    data_dir = args.data_dir or "data/zenodo/extracted"
    output_path = Path(args.output or "data/temporal/mined_facts_zenodo.json")
    summary_path = Path(args.summary or "data/temporal/mined_facts_zenodo.summary.json")

    if not os.path.isdir(data_dir):
        print(f"ERROR: Zenodo data directory not found at {data_dir}")
        sys.exit(1)

    print(f"Mining facts from Zenodo dataset at {data_dir}")
    facts = mine_facts_from_zenodo(
        data_dir=data_dir,
        variant=args.variant,
        max_triples_per_rel=args.max_per_rel,
        max_facts_total=args.max_total,
    )

    mkdir_p(str(output_path.parent))
    dump_json([f.to_dict() for f in facts], str(output_path), indent=2)
    dump_json(summarize_mined_facts(facts), str(summary_path), indent=2)

    print(f"Wrote {len(facts)} mined facts to {output_path}")
    print(f"Wrote summary to {summary_path}")


def _cmd_generate_candidates(args: argparse.Namespace) -> None:
    """Generate candidate samples from mined facts using template bank."""
    facts_path = Path(args.facts)
    whitelist_path = args.whitelist or "configs/relation_whitelist.yaml"
    output_path = Path(args.output or "data/temporal/candidate_samples.json")
    summary_path = Path(args.summary or "data/temporal/candidate_samples.summary.json")

    if not facts_path.exists():
        print(f"ERROR: Facts file not found at {facts_path}")
        sys.exit(1)
    if not Path(whitelist_path).exists():
        print(f"ERROR: Whitelist config not found at {whitelist_path}")
        sys.exit(1)

    whitelist_config = load_whitelist(whitelist_path)
    fact_dicts = load_json(str(facts_path))
    facts = [MinedTemporalFact(**f) for f in fact_dicts]
    print(f"Loaded {len(facts)} mined facts")

    candidates = generate_candidates(
        facts=facts,
        whitelist_config=whitelist_config,
        seed=args.seed,
    )

    mkdir_p(str(output_path.parent))
    dump_json([c.to_dict() for c in candidates], str(output_path), indent=2)
    dump_json(summarize_candidates(candidates), str(summary_path), indent=2)

    print(f"Wrote {len(candidates)} candidate samples to {output_path}")
    print(f"Wrote summary to {summary_path}")


def _cmd_verify_samples(args: argparse.Namespace) -> None:
    """Execute candidate S-expressions against Virtuoso and verify answers."""
    input_path = Path(args.input)
    output_passed = Path(args.output_passed or "data/temporal/verified_samples.json")
    output_failed = Path(args.output_failed or "data/temporal/failed_samples.json")
    summary_path = Path(args.summary or "data/temporal/verification_summary.json")

    if not input_path.exists():
        print(f"ERROR: Candidate file not found at {input_path}")
        sys.exit(1)

    records = load_json(str(input_path))
    candidates = [TemporalSample.from_dict(r) for r in records]
    print(f"Verifying {len(candidates)} candidate samples...")

    passed, failed = verify_samples(
        candidates,
        timeout=args.timeout,
        verbose=not args.quiet,
    )

    mkdir_p(str(output_passed.parent))
    dump_json([s.to_dict() for s in passed], str(output_passed), indent=2)
    dump_json([s.to_dict() for s in failed], str(output_failed), indent=2)
    dump_json(summarize_verification(passed, failed), str(summary_path), indent=2)

    print(f"Passed: {len(passed)}, Failed: {len(failed)}")
    print(f"Wrote passed to {output_passed}")
    print(f"Wrote failed to {output_failed}")


def _cmd_filter_and_export(args: argparse.Namespace) -> None:
    """Filter verified samples and export to training format."""
    input_path = Path(args.input)
    human_path = Path(args.human_input) if args.human_input else None
    output_path = Path(args.output or "LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json")
    summary_path = Path(args.summary or "LLMs/data/TChatKBQA_Freebase_NQ_train/summary.json")

    if not input_path.exists():
        print(f"ERROR: Verified samples not found at {input_path}")
        sys.exit(1)

    records = load_json(str(input_path))
    verified = [TemporalSample.from_dict(r) for r in records]
    print(f"Loaded {len(verified)} verified samples")

    # Apply distribution filter
    filtered = apply_distribution_filter(
        verified,
        max_per_family=args.max_per_family,
        max_per_type=args.max_per_type,
        max_per_date_bucket=args.max_per_date_bucket,
        seed=args.seed,
    )

    # Load human samples
    human_samples = None
    if human_path and human_path.exists():
        human_records = load_json(str(human_path))
        human_samples = [TemporalSample.from_dict(r) for r in human_records]
        print(f"Loaded {len(human_samples)} human samples")

    # Export
    examples = build_temporal_training_examples(
        human_samples=human_samples or [],
        synthetic_samples=filtered,
        split="train",
        max_synthetic=args.max_synthetic,
        include_metadata=True,
        exclude_suspicious_human=not args.allow_suspicious_human,
    )

    mkdir_p(str(output_path.parent))
    dump_json(examples, str(output_path), indent=2)
    dump_json(
        {
            "training_examples": summarize_training_examples(examples),
            "distribution": summarize_distribution(filtered),
        },
        str(summary_path),
        indent=2,
    )

    print(f"Wrote {len(examples)} training examples to {output_path}")
    print(f"Wrote summary to {summary_path}")


# ── Legacy Commands (re-imported for backward compat) ──────────────────

def _standardize_tempquestions(args: argparse.Namespace) -> None:
    from src.temporal_data import standardize_tempquestions_split, summarize_temporal_samples

    split = args.split
    merged_path = args.merged or f"data/TempQuestions/generation/merged/TempQuestions_{split}.json"
    origin_path = args.origin or f"data/TempQuestions/origin/TempQuestions.{split}.json"
    output_path = Path(args.output or f"data/temporal/TempQuestions/{split}.samples.json")
    summary_path = Path(args.summary or f"data/temporal/TempQuestions/{split}.summary.json")

    samples = standardize_tempquestions_split(merged_path, origin_path, split=split)
    payload = [s.to_dict() for s in samples]
    summary = summarize_temporal_samples(samples)

    mkdir_p(str(output_path.parent))
    dump_json(payload, str(output_path), indent=2)
    dump_json(summary, str(summary_path), indent=2)

    print(f"Wrote {len(samples)} temporal samples to {output_path}")


def _legacy_export(args: argparse.Namespace) -> None:
    from src.temporal_data import build_temporal_training_examples

    human_records = load_json(str(Path(args.human_input)))
    human_samples = [TemporalSample.from_dict(r) for r in human_records]

    synthetic_samples = None
    if args.synthetic_input:
        syn_records = load_json(str(Path(args.synthetic_input)))
        synthetic_samples = [TemporalSample.from_dict(r) for r in syn_records]

    examples = build_temporal_training_examples(
        human_samples=human_samples,
        synthetic_samples=synthetic_samples,
        split=args.split,
        max_synthetic=args.max_synthetic,
        include_metadata=args.include_metadata,
        exclude_suspicious_human=args.exclude_suspicious_human,
    )
    output_path = Path(args.output or f"LLMs/data/TChatKBQA_Freebase_NQ_{args.split}/examples.json")
    summary_path = Path(args.summary or f"LLMs/data/TChatKBQA_Freebase_NQ_{args.split}/summary.json")

    mkdir_p(str(output_path.parent))
    dump_json(examples, str(output_path), indent=2)
    dump_json(summarize_training_examples(examples), str(summary_path), indent=2)

    print(f"Wrote {len(examples)} training examples to {output_path}")


# ── Parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build temporal dataset artifacts for T-ChatKBQA"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── New pipeline ──

    mine_parser = subparsers.add_parser(
        "mine-facts",
        help="Mine temporal facts from Freebase Virtuoso",
    )
    mine_parser.add_argument("--whitelist", default=None, help="Path to relation_whitelist.yaml")
    mine_parser.add_argument("--limit", type=int, default=None, help="Override max facts per family")
    mine_parser.add_argument("--families", default=None, help="Comma-separated family names to mine")
    mine_parser.add_argument("--output", default=None, help="Where to write mined facts JSON")
    mine_parser.add_argument("--summary", default=None, help="Where to write mining summary")
    mine_parser.set_defaults(handler=_cmd_mine_facts)

    zenodo_parser = subparsers.add_parser(
        "mine-facts-zenodo",
        help="Mine temporal facts from Zenodo idirlab/freebases dataset (no Virtuoso needed)",
    )
    zenodo_parser.add_argument("--data-dir", default=None, help="Path to extracted Zenodo dataset")
    zenodo_parser.add_argument("--variant", default="FB+CVT+REV", help="Dataset variant")
    zenodo_parser.add_argument("--max-per-rel", type=int, default=200, help="Max triples per relation")
    zenodo_parser.add_argument("--max-total", type=int, default=10000, help="Max total facts")
    zenodo_parser.add_argument("--output", default=None, help="Where to write mined facts JSON")
    zenodo_parser.add_argument("--summary", default=None, help="Where to write mining summary")
    zenodo_parser.set_defaults(handler=_cmd_mine_facts_zenodo)

    gen_parser = subparsers.add_parser(
        "generate-candidates",
        help="Generate candidate samples from mined facts using template bank",
    )
    gen_parser.add_argument("--facts", required=True, help="Path to mined facts JSON")
    gen_parser.add_argument("--whitelist", default=None, help="Path to relation_whitelist.yaml")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--output", default=None, help="Where to write candidate samples")
    gen_parser.add_argument("--summary", default=None, help="Where to write candidate summary")
    gen_parser.set_defaults(handler=_cmd_generate_candidates)

    verify_parser = subparsers.add_parser(
        "verify-samples",
        help="Execute S-expressions against Virtuoso and verify answers",
    )
    verify_parser.add_argument("--input", required=True, help="Path to candidate samples JSON")
    verify_parser.add_argument("--timeout", type=int, default=30, help="SPARQL timeout per query")
    verify_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    verify_parser.add_argument("--output-passed", default=None, help="Where to write verified samples")
    verify_parser.add_argument("--output-failed", default=None, help="Where to write failed samples")
    verify_parser.add_argument("--summary", default=None, help="Where to write verification summary")
    verify_parser.set_defaults(handler=_cmd_verify_samples)

    export_parser = subparsers.add_parser(
        "filter-and-export",
        help="Filter verified samples and export to training format",
    )
    export_parser.add_argument("--input", required=True, help="Path to verified samples JSON")
    export_parser.add_argument("--human-input", default=None, help="Optional canonical human samples JSON")
    export_parser.add_argument("--max-per-family", type=int, default=500, help="Max samples per relation family")
    export_parser.add_argument("--max-per-type", type=int, default=1000, help="Max samples per reasoning type")
    export_parser.add_argument("--max-per-date-bucket", type=int, default=200, help="Max samples per decade")
    export_parser.add_argument("--max-synthetic", type=int, default=None, help="Absolute cap on synthetic examples")
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed for cap sampling")
    export_parser.add_argument("--allow-suspicious-human", action="store_true", help="Include phase0_suspicious human records")
    export_parser.add_argument("--output", default=None, help="Where to write examples.json")
    export_parser.add_argument("--summary", default=None, help="Where to write export summary")
    export_parser.set_defaults(handler=_cmd_filter_and_export)

    # ── Legacy commands ──

    stdz_parser = subparsers.add_parser(
        "standardize-tempquestions",
        help="[Legacy] Normalize TempQuestions into canonical schema",
    )
    stdz_parser.add_argument("--split", choices=["train", "test"], required=True)
    stdz_parser.add_argument("--merged", default=None)
    stdz_parser.add_argument("--origin", default=None)
    stdz_parser.add_argument("--output", default=None)
    stdz_parser.add_argument("--summary", default=None)
    stdz_parser.set_defaults(handler=_standardize_tempquestions)

    leg_export = subparsers.add_parser(
        "export-training-examples",
        help="[Legacy] Export human/synthetic samples to instruction-tuning format",
    )
    leg_export.add_argument("--human-input", required=True)
    leg_export.add_argument("--synthetic-input", default=None)
    leg_export.add_argument("--split", default="train")
    leg_export.add_argument("--max-synthetic", type=int, default=None)
    leg_export.add_argument("--include-metadata", action="store_true")
    leg_export.add_argument("--exclude-suspicious-human", action="store_true")
    leg_export.add_argument("--output", default=None)
    leg_export.add_argument("--summary", default=None)
    leg_export.set_defaults(handler=_legacy_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "handler"):
        parser.print_help()
        sys.exit(1)

    args.handler(args)


if __name__ == "__main__":
    main()
