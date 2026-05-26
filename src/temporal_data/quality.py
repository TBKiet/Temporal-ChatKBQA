"""Quality filtering for synthetic temporal samples."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Iterable
import re

from .schema import TemporalDataSource, TemporalQuestionType, TemporalSample, ValidationStatus


_MID_PATTERN = re.compile(r"\bm\.[A-Za-z0-9_]+\b")
_DATE_LITERAL_PATTERN = re.compile(r"\b\d{4}(?:-\d{2}-\d{2})?\b")


def review_synthetic_sample(sample: TemporalSample) -> list[str]:
    """Return filtering issues for one synthetic temporal sample."""
    issues: list[str] = []

    if sample.source != TemporalDataSource.SYNTHETIC:
        issues.append("sample source is not synthetic")
    if not sample.question.strip():
        issues.append("question is required")
    if not sample.s_expression.strip() or sample.s_expression.lower() == "null":
        issues.append("s_expression is missing or null")
    if sample.temporal_type == TemporalQuestionType.UNKNOWN:
        issues.append("temporal_type is unknown")
    if not sample.answers:
        issues.append("answers must not be empty")
    if not sample.topic_entity_mid.startswith("m."):
        issues.append("topic entity mid must look like a Freebase mid")
    if "UNKNOWN" in sample.s_expression:
        issues.append("s_expression contains UNKNOWN placeholder")
    if _MID_PATTERN.search(sample.question):
        issues.append("question still contains raw mids")
    if not sample.relation_ids:
        issues.append("relation_ids must not be empty")
    if not sample.metadata.get("source_seed_id"):
        issues.append("missing source_seed_id metadata")
    if not sample.metadata.get("mined_fact_id"):
        issues.append("missing mined_fact_id metadata")

    if sample.temporal_type == TemporalQuestionType.DURING:
        if "TC" not in sample.s_expression.upper():
            issues.append("during sample must use TC operator")
        if not _DATE_LITERAL_PATTERN.search(sample.question) and not _DATE_LITERAL_PATTERN.search(sample.s_expression):
            issues.append("during sample should include a temporal value")
    if sample.temporal_type == TemporalQuestionType.LAST and "ARGMAX" not in sample.s_expression.upper():
        issues.append("last sample must use ARGMAX")
    if sample.temporal_type == TemporalQuestionType.FIRST and "ARGMIN" not in sample.s_expression.upper():
        issues.append("first sample must use ARGMIN")

    return issues


def filter_synthetic_samples(
    samples: Iterable[TemporalSample],
) -> tuple[list[TemporalSample], list[TemporalSample]]:
    """Split synthetic temporal samples into accepted and rejected sets."""
    accepted: list[TemporalSample] = []
    rejected: list[TemporalSample] = []

    for sample in samples:
        issues = review_synthetic_sample(sample)
        if issues:
            rejected.append(
                replace(
                    sample,
                    validation_status=ValidationStatus.FAILED,
                    metadata={**sample.metadata, "filter_issues": issues},
                )
            )
            continue
        accepted.append(
            replace(
                sample,
                validation_status=ValidationStatus.NORMALIZED,
                metadata={**sample.metadata, "filter_issues": []},
            )
        )
    return accepted, rejected


def summarize_synthetic_filtering(
    accepted: Iterable[TemporalSample],
    rejected: Iterable[TemporalSample],
) -> dict[str, Any]:
    """Summarize synthetic filtering results and major rejection reasons."""
    accepted = list(accepted)
    rejected = list(rejected)
    by_type = Counter(sample.temporal_type.value for sample in accepted)
    rejection_reasons = Counter()
    for sample in rejected:
        for issue in sample.metadata.get("filter_issues", []):
            rejection_reasons[issue] += 1

    total = len(accepted) + len(rejected)
    acceptance_rate = (len(accepted) / total) if total else 0.0
    return {
        "total_samples": total,
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected),
        "acceptance_rate": round(acceptance_rate, 4),
        "accepted_by_temporal_type": dict(sorted(by_type.items())),
        "top_rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in rejection_reasons.most_common(20)
        ],
    }
