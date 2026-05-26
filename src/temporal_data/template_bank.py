"""Template bank for temporal question generation.

Defines 6 reasoning families with multiple paraphrased question templates.
Slot-fills MinedTemporalFact records into templates to produce
TemporalSample candidates with valid S-expressions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
    infer_temporal_question_type,
)
from .sparql_miner import MinedTemporalFact


# ── Template definition ────────────────────────────────────────────────

@dataclass
class Template:
    """A single question template with slots for fact fields."""

    template_id: str
    family: str  # simple, during, before, after, first, last
    question_template: str  # with {topic_label}, {answer_label}, {year}, etc.
    s_expr_builder: str  # "join" | "tc" | "argmin" | "argmax" | "tc_lt" | "tc_gt"


# ── S-expression builders ──────────────────────────────────────────────

def _build_join_expr(fact: MinedTemporalFact) -> str:
    return f"(JOIN (R {fact.answer_relation}) {fact.topic_mid})"


def _build_tc_expr(fact: MinedTemporalFact) -> str:
    date = fact.temporal_start or fact.temporal_end or "2000"
    return f"(TC (JOIN (R {fact.answer_relation}) {fact.topic_mid}) {fact.retrieved_from_relation} {date})"


def _build_argmin_expr(fact: MinedTemporalFact) -> str:
    return f"(ARGMIN (JOIN (R {fact.answer_relation}) {fact.topic_mid}) {fact.retrieved_from_relation})"


def _build_argmax_expr(fact: MinedTemporalFact) -> str:
    return f"(ARGMAX (JOIN (R {fact.answer_relation}) {fact.topic_mid}) {fact.retrieved_from_relation})"


S_EXPR_BUILDERS: Dict[str, Callable] = {
    "join": _build_join_expr,
    "tc": _build_tc_expr,
    "argmin": _build_argmin_expr,
    "argmax": _build_argmax_expr,
}


# ── Template bank ──────────────────────────────────────────────────────

TEMPLATES: List[Template] = [
    # ── Simple (direction-agnostic) ──
    Template("simple_01", "simple",
             "What {role} is associated with {topic_label}?",
             "join"),
    Template("simple_02", "simple",
             "Find the {role} connected to {topic_label}.",
             "join"),
    Template("simple_03", "simple",
             "Name a {role} related to {topic_label}.",
             "join"),
    Template("simple_04", "simple",
             "Which {role} has a connection to {topic_label}?",
             "join"),

    # ── During (explicit date) — only used when fact has real temporal data ──
    Template("during_01", "during",
             "What {role} was associated with {topic_label} in {year}?",
             "tc"),
    Template("during_02", "during",
             "Name the {role} linked to {topic_label} as of {year}.",
             "tc"),
    Template("during_03", "during",
             "In {year}, what {role} was connected to {topic_label}?",
             "tc"),

    # ── First (earliest) ──
    Template("first_01", "first",
             "What was the earliest {role} associated with {topic_label}?",
             "argmin"),
    Template("first_02", "first",
             "Name the first {role} linked to {topic_label}.",
             "argmin"),
    Template("first_03", "first",
             "What {role} came first for {topic_label}?",
             "argmin"),
    Template("first_04", "first",
             "Find the original {role} connected to {topic_label}.",
             "argmin"),

    # ── Last (most recent) ──
    Template("last_01", "last",
             "What was the most recent {role} associated with {topic_label}?",
             "argmax"),
    Template("last_02", "last",
             "Name the latest {role} linked to {topic_label}.",
             "argmax"),
    Template("last_03", "last",
             "What {role} was most recent for {topic_label}?",
             "argmax"),
    Template("last_04", "last",
             "Find the last {role} connected to {topic_label}.",
             "argmax"),
]


# ── Slot filling ───────────────────────────────────────────────────────

def _extract_year(date_str: str) -> int:
    """Extract a year integer from a date string."""
    if not date_str:
        return 2000
    try:
        return int(date_str[:4])
    except (ValueError, IndexError):
        return 2000


def _format_slot(template: str, fact: MinedTemporalFact) -> str:
    """Fill template slots with values from a mined fact."""
    start_year = _extract_year(fact.temporal_start)
    end_year = _extract_year(fact.temporal_end) if fact.temporal_end else start_year

    # Derive role hint: use metadata if available, else extract from relation
    role = fact.metadata.get("role_hint", "")
    if not role:
        parts = fact.anchor_relation.split(".")
        role = parts[-2] if len(parts) >= 3 else (parts[-1] if parts else "entity")
        role = role.replace("_", " ")

    result = template
    result = result.replace("{topic_label}", fact.topic_label)
    result = result.replace("{answer_label}", fact.answer_label)
    result = result.replace("{role}", role)
    result = result.replace("{year}", str(start_year))
    result = result.replace("{start_year}", str(start_year))
    result = result.replace("{end_year}", str(end_year))
    return result


def _filter_templates_for_family(family_cfg: dict, fact: MinedTemporalFact) -> List[Template]:
    """Return templates compatible with a relation family and fact.

    Skips TC/during templates when fact has no real temporal data.
    """
    supported = set(family_cfg.get("supported_templates", []))
    has_date = bool(fact.temporal_start and fact.temporal_start != "2000")
    result = []
    for t in TEMPLATES:
        if t.family not in supported:
            continue
        # Only include during/tc templates when fact has a real date
        if t.family == "during" and not has_date:
            continue
        result.append(t)
    return result


def generate_candidates(
    facts: List[MinedTemporalFact],
    whitelist_config: dict,
    seed: int = 42,
) -> List[TemporalSample]:
    """Generate candidate TemporalSamples from mined facts and a template bank.

    Args:
        facts: Mined temporal facts from SPARQL mining.
        whitelist_config: Loaded relation_whitelist.yaml config.
        seed: Random seed for reproducibility.

    Returns:
        List of TemporalSample candidates ready for verification.
    """
    rng = random.Random(seed)

    # Build family lookup
    families_by_name: Dict[str, dict] = {}
    for fcfg in whitelist_config["families"]:
        families_by_name[fcfg["family"]] = fcfg

    candidates: List[TemporalSample] = []
    seen: set = set()

    for fact in facts:
        family_cfg = families_by_name.get(fact.fact_relation_family)
        if family_cfg is None:
            continue

        templates = _filter_templates_for_family(family_cfg, fact)
        if not templates:
            continue

        # Apply 1-3 random templates per fact
        num_templates = min(len(templates), rng.randint(1, 3))
        selected = rng.sample(templates, num_templates)

        for template in selected:
            question = _format_slot(template.question_template, fact)
            s_expr_builder = S_EXPR_BUILDERS.get(template.s_expr_builder)
            if s_expr_builder is None:
                continue

            s_expression = s_expr_builder(fact)
            temporal_type = infer_temporal_question_type(
                question=question,
                temporal_signal=template.family,
                s_expression=s_expression,
            )

            # Deduplicate
            dedup_key = (question, s_expression)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            sample = TemporalSample(
                sample_id=f"candidate-{fact.fact_id}-{template.template_id}",
                question=question,
                split="train",
                source=TemporalDataSource.SYNTHETIC,
                temporal_type=temporal_type,
                temporal_signal=template.family,
                topic_entity_mid=fact.topic_mid,
                s_expression=s_expression,
                sparql="",  # will be filled during verification
                answers=[fact.answer_mid],
                validation_status=ValidationStatus.RAW,
                source_dataset="SyntheticTemporal",
                relation_ids=[
                    fact.anchor_relation,
                    fact.answer_relation,
                    fact.retrieved_from_relation,
                ],
                entity_ids=[fact.topic_mid, fact.answer_mid],
                metadata={
                    "template_id": template.template_id,
                    "template_family": template.family,
                    "fact_relation_family": fact.fact_relation_family,
                    "topic_label": fact.topic_label,
                    "answer_label": fact.answer_label,
                    "temporal_start": fact.temporal_start,
                    "temporal_end": fact.temporal_end,
                    "supporting_row_hash": fact.supporting_row_hash,
                },
            )
            candidates.append(sample)

    return candidates


def summarize_candidates(samples: List[TemporalSample]) -> dict:
    """Summary statistics for generated candidates."""
    from collections import Counter

    by_family = Counter(s.metadata.get("template_family", "unknown") for s in samples)
    by_type = Counter(s.temporal_type.value for s in samples)
    by_relation = Counter(s.metadata.get("fact_relation_family", "unknown") for s in samples)

    return {
        "total_candidates": len(samples),
        "by_template_family": dict(sorted(by_family.items())),
        "by_temporal_type": dict(sorted(by_type.items())),
        "by_relation_family": dict(sorted(by_relation.items())),
        "unique_questions": len({s.question for s in samples}),
        "unique_expressions": len({s.s_expression for s in samples}),
    }
