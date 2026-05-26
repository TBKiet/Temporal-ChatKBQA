# T-ChatKBQA Overview

- Project: `T-ChatKBQA`
- Baseline: `ChatKBQA`
- Goal: Extend ChatKBQA to temporal KBQA
- Contribution: Temporal dataset construction + model adaptation
- Primary benchmark: `TempQuestions`
- Synthetic data role: Primary training source (human TempQuestions data is corrupted)
- Heavy compute environment: `Vast.ai` (Zenodo dataset, GPU training)
- Local role: Code, lightweight tests (32 tests, offline), and Streamlit demo
- Current phase: **Evaluation** — first trial complete (data→train→infer), 87.6% valid predictions. Zenodo answer-level eval attempted (2026-05-26): S-expression validity confirmed but answer-level F1 blocked by relation hallucination (only 22 training relations, model invents 177 unique relations).
- Current top priority: Report honest findings (validity 87.6%, hallucination analysis, operator distribution); v2: increase relation diversity to 50-100, add retrieval grounding, source dates for TC
- Next milestone: Submit v1 report with actual evaluation numbers and error analysis

## Pipeline (Current)

```
Zenodo idirlab/freebases (FB+CVT+REV, 244M triples, 122M entities)
        │
        ▼
[1] zenodo_loader.py     Extract temporal triples → resolve labels → MinedTemporalFact[]
        │
        ▼
[2] template_bank.py     15 direction-agnostic templates × 4 reasoning families
                          JOIN / ARGMIN / ARGMAX only (TC skipped: no dates)
        │
        ▼
[3] distribution_filter  Dedup → cap by family/type → balanced dataset
        │
        ▼
[4] training.py          Export to ChatKBQA instruction-tuning format
```

Key decisions:
- **No Virtuoso SPARQL** — dump lacks labels. Zenodo triple files are the primary fact source.
- **No TC/during templates** — Zenodo facts lack temporal dates. JOIN + ARGMAX + ARGMIN only.
- **Role hints from relation path** — last segment of relation path → human-readable role.
- **Direction-agnostic templates** — avoid subject/object confusion in questions.

## Read Order For New Sessions

1. `overview.md`
2. `progress.md`
3. `decisions.md`
4. `master_plan.md` if more detail is needed
