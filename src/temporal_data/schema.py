"""Canonical schema for temporal KBQA training and evaluation data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional
import re


class TemporalQuestionType(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    FIRST = "first"
    LAST = "last"
    EXPLICIT_DATE = "explicit_date"
    TEMPORAL_ANSWER = "temporal_answer"
    UNKNOWN = "unknown"


class TemporalDataSource(str, Enum):
    HUMAN = "human"
    SYNTHETIC = "synthetic"


class ValidationStatus(str, Enum):
    RAW = "raw"
    NORMALIZED = "normalized"
    EXECUTABLE = "executable"
    FAILED = "failed"


_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def infer_temporal_question_type(
    question: str,
    temporal_signal: Optional[str] = None,
    s_expression: Optional[str] = None,
    sparql: Optional[str] = None,
) -> TemporalQuestionType:
    """Infer a stable temporal question type from question and annotations."""
    text = " ".join(
        part for part in [
            question.lower() if question else "",
            (temporal_signal or "").lower(),
            (s_expression or "").lower(),
            (sparql or "").lower(),
        ]
        if part
    )

    if "argmin" in text or re.search(r"\bfirst\b|\bearliest\b", text):
        return TemporalQuestionType.FIRST
    if "argmax" in text or re.search(r"\blast\b|\blatest\b|\bmost recent\b", text):
        return TemporalQuestionType.LAST
    if re.search(r"\bbefore\b", text):
        return TemporalQuestionType.BEFORE
    if re.search(r"\bafter\b|\bsince\b", text):
        return TemporalQuestionType.AFTER
    if re.search(r"\bduring\b|\bwhen\b|\bin\b", text) and _YEAR_PATTERN.search(text):
        return TemporalQuestionType.DURING
    if "tc" in text or re.search(r"\bduring\b|\bwhen\b", text):
        return TemporalQuestionType.DURING
    if _YEAR_PATTERN.search(text):
        return TemporalQuestionType.EXPLICIT_DATE
    if re.search(r"\bwhen did\b|\bwhat year\b|\bwhat date\b", text):
        return TemporalQuestionType.TEMPORAL_ANSWER
    return TemporalQuestionType.UNKNOWN


def normalize_answers(answers: list[Any]) -> list[str]:
    """Normalize answers into a stable list[str] representation."""
    normalized: list[str] = []
    for answer in answers or []:
        if answer is None:
            continue
        if isinstance(answer, dict):
            if "AnswerArgument" in answer:
                normalized.append(str(answer["AnswerArgument"]))
            elif "EntityName" in answer:
                normalized.append(str(answer["EntityName"]))
            else:
                normalized.append(str(answer))
        else:
            normalized.append(str(answer))
    return normalized


@dataclass
class TemporalSample:
    """Canonical record for human and synthetic temporal supervision."""

    sample_id: str
    question: str
    split: str
    source: TemporalDataSource
    temporal_type: TemporalQuestionType
    temporal_signal: str
    topic_entity_mid: str
    s_expression: str
    sparql: str
    answers: list[str]
    validation_status: ValidationStatus = ValidationStatus.NORMALIZED
    source_dataset: str = "TempQuestions"
    relation_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return validation issues. Empty list means valid."""
        issues: list[str] = []
        if not self.sample_id:
            issues.append("sample_id is required")
        if not self.question.strip():
            issues.append("question is required")
        if not self.split.strip():
            issues.append("split is required")
        if not self.temporal_signal.strip():
            issues.append("temporal_signal is required")
        if not self.s_expression.strip() or self.s_expression == "null":
            issues.append("s_expression is missing or null")
        if not self.sparql.strip():
            issues.append("sparql is required")
        if not self.answers:
            issues.append("answers must not be empty")
        if self.temporal_type == TemporalQuestionType.UNKNOWN:
            issues.append("temporal_type should be resolved before export")
        return issues

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["temporal_type"] = self.temporal_type.value
        payload["validation_status"] = self.validation_status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalSample":
        return cls(
            sample_id=str(payload["sample_id"]),
            question=payload["question"],
            split=payload["split"],
            source=TemporalDataSource(payload["source"]),
            temporal_type=TemporalQuestionType(payload["temporal_type"]),
            temporal_signal=payload["temporal_signal"],
            topic_entity_mid=payload.get("topic_entity_mid", ""),
            s_expression=payload.get("s_expression", ""),
            sparql=payload.get("sparql", ""),
            answers=normalize_answers(payload.get("answers", [])),
            validation_status=ValidationStatus(
                payload.get("validation_status", ValidationStatus.NORMALIZED.value)
            ),
            source_dataset=payload.get("source_dataset", "TempQuestions"),
            relation_ids=list(payload.get("relation_ids", [])),
            entity_ids=list(payload.get("entity_ids", [])),
            metadata=dict(payload.get("metadata", {})),
        )
