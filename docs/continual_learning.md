# Continual Learning & Monitoring Strategy — Temporal ChatKBQA

## Overview

This document outlines how the Temporal ChatKBQA system would evolve over time as new data becomes available, performance degrades, or requirements change. While full implementation of continual learning is beyond the scope of this project, we design the strategy such that it can be operationalized with minimal additional infrastructure.

## Data Collection Strategy

### Sources of New Data

1. **User feedback loop**: Each `/ask` API response includes the full provenance (SPARQL used, temporal constraint, retries). A feedback endpoint (`POST /ask/feedback`) could collect:
   - Thumbs up/down on answer correctness
   - User-provided correct answer
   - Free-text comment

2. **Query logs**: All SPARQL queries and their results are logged (anonymized). Queries that return empty results are flagged for review — these may indicate missing KB facts or parsing failures.

3. **New temporal questions**: As new temporal datasets emerge (e.g., CronQuestions, MultiTQ) or domain-specific temporal questions are collected, they can be added to the training pool.

### Data Collection Pipeline (Proposed)

```
User Feedback → Feedback DB → Periodic Review → Curated New Examples
Query Logs    → Error Log   → Manual/Semi-auto Annotation → Training Pool
New Datasets  → Preprocess  → Merge → Instruction-Tuning Format
```

## Retraining & Fine-Tuning Strategy

### Triggers for Retraining

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Temporal F1 drops | > 5% relative decline | Full retraining on updated dataset |
| New temporal signal types | Any new signal (e.g., "throughout", "meanwhile") | Data augmentation + fine-tune |
| KB schema change | Freebase update / migration to Wikidata | Reparse SPARQL → retrain |
| User feedback volume | > 100 new verified examples | Incremental fine-tune on new data |
| Scheduled refresh | Quarterly | Evaluate on held-out set, retrain if degraded |

### Retraining Approaches

1. **Incremental LoRA fine-tuning** (preferred for small data additions):
   - Load existing LoRA weights
   - Continue training on new examples for 5-10 epochs
   - Lower learning rate (1e-5) to avoid catastrophic forgetting
   - Validate on original test set + new holdout

2. **Full retraining** (for major changes):
   - Merge old and new training data
   - Train from base LLaMA-2-7b checkpoint
   - Use same LoRA configuration

3. **Model versioning**:
   - Each retrained checkpoint is tagged: `models/LLaMA2-7b-temporal/checkpoint-v{date}-{trigger}`
   - Configs track which checkpoint is active via `configs/inference.yaml`

## Performance Degradation Detection

### Monitoring Metrics (Production)

| Metric | Collection Method | Alert Threshold |
|--------|-------------------|----------------|
| Answer rate (% non-empty) | API response logging | < 60% over 1h window |
| Average retry count | Agent provenance field | > 1.0 avg over 100 requests |
| SPARQL execution errors | Exception logging | > 20% error rate |
| Inference latency (p50, p95, p99) | Request timing middleware | p99 > 10s |
| Temporal vs. standard routing ratio | Agent decision logging | Significant shift from baseline (drift signal) |

### Drift Detection

- **Distributional drift**: Compare token distributions of incoming questions vs. training data (Jensen-Shannon divergence). If divergence exceeds threshold, flag for review.
- **Semantic drift**: Track embedding similarity (via SimCSE) between new questions and training questions. Low similarity indicates out-of-domain queries.
- **Temporal scope drift**: Monitor the distribution of years referenced in questions. If questions increasingly reference post-2016 events, Freebase coverage becomes an issue.

## Mitigation Strategies

| Risk | Mitigation |
|------|-----------|
| Model overfits to new data | Keep original test set as holdout; validate after each retrain |
| KB staleness (Freebase archived 2016) | Plan migration path to Wikidata (actively maintained) |
| LoRA adapter drift | Maintain baseline non-temporal ChatKBQA as fallback |
| GPU unavailability for retraining | LoRA enables CPU fine-tuning (slower but feasible) |
| Concept drift in temporal expressions | Periodic review of temporal signal regex; add new patterns |

## Implementation Roadmap

| Phase | Timeline | Action |
|-------|----------|--------|
| Phase 1 | Now | Log all API responses with provenance |
| Phase 2 | Month 1-2 | Add `/ask/feedback` endpoint, build feedback DB |
| Phase 3 | Month 3 | First scheduled evaluation; establish baseline metrics |
| Phase 4 | Month 6 | First incremental retrain on collected feedback data |
| Phase 5 | Ongoing | Quarterly retrain + drift monitoring |
