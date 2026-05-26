"""Temporal data utilities for T-ChatKBQA.

Pipeline: mine facts → template generate → execute verify → filter → export.

Core modules:
  - sparql_miner:   Direct SPARQL fact mining from Freebase Virtuoso
  - template_bank:  Template-based question + S-expression generation
  - verifier:       Execution-based answer verification
  - distribution_filter: Dedup, cap, and balance
  - training:       Export to ChatKBQA instruction-tuning format
  - schema:         Canonical dataclasses (TemporalSample, etc.)
  - builder:        TempQuestions standardization helpers
  - quality:        Synthetic sample quality filtering
"""

from .schema import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
    infer_temporal_question_type,
)
from .builder import (
    build_temporal_relation_inventory,
    standardize_tempquestions_split,
    summarize_temporal_samples,
)
from .sparql_miner import (
    MinedTemporalFact,
    mine_facts_raw,
    mine_temporal_facts,
    summarize_mined_facts,
    load_whitelist,
)
from .template_bank import (
    generate_candidates,
    summarize_candidates,
)
from .verifier import (
    verify_sample,
    verify_samples,
    summarize_verification,
    normalize_answer,
    answers_match,
)
from .distribution_filter import (
    apply_distribution_filter,
    deduplicate,
    summarize_distribution,
)
from .zenodo_loader import (
    mine_facts_from_zenodo,
    load_relation_map,
    select_temporal_relation_ids,
    extract_temporal_triples,
)
from .training import (
    build_temporal_training_examples,
    sample_to_training_example,
    summarize_training_examples,
)
from .quality import (
    filter_synthetic_samples,
    review_synthetic_sample,
    summarize_synthetic_filtering,
)

__all__ = [
    # Schema
    "TemporalDataSource",
    "TemporalQuestionType",
    "TemporalSample",
    "ValidationStatus",
    "infer_temporal_question_type",
    # Builder
    "build_temporal_relation_inventory",
    "standardize_tempquestions_split",
    "summarize_temporal_samples",
    # SPARQL miner
    "MinedTemporalFact",
    "mine_facts_raw",
    "mine_temporal_facts",
    "summarize_mined_facts",
    "load_whitelist",
    # Template bank
    "generate_candidates",
    "summarize_candidates",
    # Verifier
    "verify_sample",
    "verify_samples",
    "summarize_verification",
    "normalize_answer",
    "answers_match",
    # Distribution filter
    "apply_distribution_filter",
    "deduplicate",
    "summarize_distribution",
    # Training
    "sample_to_training_example",
    "build_temporal_training_examples",
    "summarize_training_examples",
    # Zenodo loader
    "mine_facts_from_zenodo",
    "load_relation_map",
    "select_temporal_relation_ids",
    "extract_temporal_triples",
    # Quality
    "review_synthetic_sample",
    "filter_synthetic_samples",
    "summarize_synthetic_filtering",
]
