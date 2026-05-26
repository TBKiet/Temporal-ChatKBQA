"""Builders for canonical temporal KBQA datasets and relation inventories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

from components.utils import (
    extract_mentioned_entities_from_sexpr,
    extract_mentioned_relations_from_sexpr,
    load_json,
)
from .schema import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
    infer_temporal_question_type,
    normalize_answers,
)


TEMPORAL_RELATION_HINTS = (
    "from",
    "to",
    "start",
    "end",
    "start_date",
    "end_date",
    "date",
    "year",
    "time",
    "datetime",
)


@dataclass
class RelationInventoryEntry:
    relation_id: str
    evidence_count: int
    temporal_hint: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relation_id": self.relation_id,
            "evidence_count": self.evidence_count,
            "temporal_hint": self.temporal_hint,
        }


def _safe_temporal_signal(raw_item: dict, normalized_item: dict) -> str:
    signal = raw_item.get("TemporalSignalNorm") or normalized_item.get("TemporalSignalNorm")
    if signal:
        return str(signal).lower()

    question = raw_item.get("Question") or normalized_item.get("question", "")
    q = question.lower()
    if "before" in q:
        return "before"
    if "after" in q:
        return "after"
    if "first" in q or "earliest" in q:
        return "first"
    if "last" in q or "latest" in q:
        return "last"
    return "during"


def _resolve_topic_entity(raw_item: dict, normalized_item: dict) -> str:
    topic_entity_mid = raw_item.get("TopicEntityMid", "")
    if topic_entity_mid and topic_entity_mid != "m.placeholder":
        return topic_entity_mid

    entity_ids = normalized_item.get("entities") or []
    return entity_ids[0] if entity_ids else ""


def standardize_tempquestions_split(
    merged_path: str | Path,
    origin_path: str | Path,
    split: str,
) -> list[TemporalSample]:
    """Normalize a TempQuestions split into the canonical temporal schema."""
    merged_path = Path(merged_path)
    audited_candidate = merged_path.with_name(merged_path.stem + ".audited" + merged_path.suffix)
    if audited_candidate.exists() and merged_path.name.startswith(f"TempQuestions_{split}"):
        merged_path = audited_candidate

    merged_examples = load_json(str(merged_path))
    origin_examples = load_json(str(origin_path))

    origin_by_id = {str(item["Id"]): item for item in origin_examples}
    samples: list[TemporalSample] = []

    for merged in merged_examples:
        sample_id = str(merged["ID"])
        raw = origin_by_id.get(sample_id, {})
        question = merged["question"]
        s_expression = merged.get("normed_sexpr") or merged.get("sexpr") or ""
        sparql = raw.get("Sparql", "")
        temporal_signal = _safe_temporal_signal(raw, merged)
        temporal_type = infer_temporal_question_type(
            question=question,
            temporal_signal=temporal_signal,
            s_expression=s_expression,
            sparql=sparql,
        )

        sample = TemporalSample(
            sample_id=sample_id,
            question=question,
            split=split,
            source=TemporalDataSource.HUMAN,
            temporal_type=temporal_type,
            temporal_signal=temporal_signal,
            topic_entity_mid=_resolve_topic_entity(raw, merged),
            s_expression=s_expression,
            sparql=sparql,
            answers=normalize_answers(merged.get("answer", [])),
            validation_status=ValidationStatus.NORMALIZED,
            source_dataset="TempQuestions",
            relation_ids=sorted(set(merged.get("relations", []))),
            entity_ids=sorted(
                set(merged.get("entities", [])) | set(extract_mentioned_entities_from_sexpr(s_expression))
            ),
            metadata={
                "raw_temporal_signal": raw.get("Temporal signal", []),
                "raw_type": raw.get("Type", []),
                "data_source": raw.get("Data source", ""),
                "question_creation_date": raw.get("Question creation date", ""),
                "phase0_suspicious": merged.get("phase0_suspicious", False),
                "phase0_suspicious_reasons": list(merged.get("phase0_suspicious_reasons", [])),
                "phase0_inferred_temporal_type": merged.get("phase0_inferred_temporal_type"),
                "phase0_origin_question_type": merged.get("phase0_origin_question_type"),
            },
        )
        issues = sample.validate()
        if issues:
            sample.validation_status = ValidationStatus.FAILED
            sample.metadata["validation_issues"] = issues
        samples.append(sample)

    return samples


def _hint_for_relation(relation_id: str) -> str:
    tokens = re.split(r"[._]", relation_id.lower())
    for hint in TEMPORAL_RELATION_HINTS:
        if hint in tokens:
            return hint
    return "sexpr_context"


def _is_temporal_relation(relation_id: str) -> bool:
    tokens = set(re.split(r"[._]", relation_id.lower()))
    if {"from", "to"} & tokens:
        return True
    if {"start", "end"} & tokens:
        return True
    if "date" in tokens or "datetime" in tokens:
        return True
    if "year" in tokens or "time" in tokens:
        return True
    if "start_date" in relation_id.lower() or "end_date" in relation_id.lower():
        return True
    return False


def build_temporal_relation_inventory(
    samples: Iterable[TemporalSample],
    ontology_path: str | Path | None = None,
) -> list[RelationInventoryEntry]:
    """Build a ranked inventory of temporalizable relations."""
    counter: Counter[str] = Counter()

    for sample in samples:
        relation_ids = set(sample.relation_ids)
        relation_ids.update(extract_mentioned_relations_from_sexpr(sample.s_expression))
        for relation_id in relation_ids:
            if _is_temporal_relation(relation_id):
                counter[relation_id] += 1

    if ontology_path is not None:
        ontology_file = Path(ontology_path)
        if ontology_file.exists():
            for line in ontology_file.read_text(encoding="utf8").splitlines():
                parts = line.split()
                for token in parts[1:]:
                    if _is_temporal_relation(token):
                        counter[token] += 1

    return [
        RelationInventoryEntry(
            relation_id=relation_id,
            evidence_count=count,
            temporal_hint=_hint_for_relation(relation_id),
        )
        for relation_id, count in counter.most_common()
    ]


def summarize_temporal_samples(samples: Iterable[TemporalSample]) -> dict:
    samples = list(samples)
    by_type = Counter(sample.temporal_type.value for sample in samples)
    by_status = Counter(sample.validation_status.value for sample in samples)
    return {
        "total_samples": len(samples),
        "by_temporal_type": dict(sorted(by_type.items())),
        "by_validation_status": dict(sorted(by_status.items())),
        "non_empty_topic_entities": sum(1 for sample in samples if sample.topic_entity_mid),
    }
