"""Training-data export helpers for T-ChatKBQA."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from src import STANDARD_INSTRUCTION, TEMPORAL_INSTRUCTION

from .quality import filter_synthetic_samples
from .schema import TemporalDataSource, TemporalQuestionType, TemporalSample


def _instruction_for_sample(sample: TemporalSample) -> str:
    if sample.temporal_type == TemporalQuestionType.UNKNOWN:
        return STANDARD_INSTRUCTION
    return TEMPORAL_INSTRUCTION


def sample_to_training_example(
    sample: TemporalSample,
    include_metadata: bool = False,
) -> Optional[dict[str, Any]]:
    """Convert a canonical temporal sample into instruction-tuning format."""
    if not sample.question.strip():
        return None
    if not sample.s_expression.strip() or sample.s_expression.lower() == "null":
        return None

    payload: dict[str, Any] = {
        "instruction": _instruction_for_sample(sample),
        "input": f"Question: {{ {sample.question} }}",
        "output": sample.s_expression,
        "history": [],
    }
    if include_metadata:
        payload["metadata"] = {
            "sample_id": sample.sample_id,
            "source": sample.source.value,
            "temporal_type": sample.temporal_type.value,
            "source_dataset": sample.source_dataset,
            "validation_status": sample.validation_status.value,
        }
    return payload


def build_temporal_training_examples(
    human_samples: Iterable[TemporalSample],
    synthetic_samples: Optional[Iterable[TemporalSample]] = None,
    split: str = "train",
    max_synthetic: Optional[int] = None,
    include_metadata: bool = False,
    require_filtered_synthetic: bool = False,
    exclude_suspicious_human: bool = False,
) -> list[dict[str, Any]]:
    """Build a train-ready temporal instruction dataset.

    The output matches the ``examples.json`` format used by existing ChatKBQA
    training scripts while preserving optional metadata for debugging.
    """
    examples: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    def _append_samples(samples: Iterable[TemporalSample], source: TemporalDataSource, limit: Optional[int] = None) -> None:
        count = 0
        for sample in samples:
            if sample.split != split:
                continue
            if sample.source != source:
                continue
            if source == TemporalDataSource.HUMAN and exclude_suspicious_human and sample.metadata.get("phase0_suspicious"):
                continue
            example = sample_to_training_example(sample, include_metadata=include_metadata)
            if example is None:
                continue
            dedupe_key = (example["input"], example["output"])
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            examples.append(example)
            count += 1
            if limit is not None and count >= limit:
                break

    _append_samples(human_samples, TemporalDataSource.HUMAN)
    if synthetic_samples is not None:
        if require_filtered_synthetic:
            synthetic_samples, _ = filter_synthetic_samples(synthetic_samples)
        _append_samples(synthetic_samples, TemporalDataSource.SYNTHETIC, limit=max_synthetic)
    return examples


def summarize_training_examples(examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a training export for quick sanity checks."""
    examples = list(examples)
    by_source = Counter()
    by_temporal_type = Counter()
    with_metadata = 0
    for example in examples:
        metadata = example.get("metadata")
        if not metadata:
            continue
        with_metadata += 1
        by_source[metadata.get("source", "unknown")] += 1
        by_temporal_type[metadata.get("temporal_type", "unknown")] += 1

    return {
        "total_examples": len(examples),
        "with_metadata": with_metadata,
        "by_source": dict(sorted(by_source.items())),
        "by_temporal_type": dict(sorted(by_temporal_type.items())),
    }
