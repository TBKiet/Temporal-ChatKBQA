# Project Plan — Temporal ChatKBQA

## Scope

The submission scope is a production-oriented temporal extension of `ChatKBQA`, not a new KBQA architecture. The core deliverables are:

- a reproducible temporal data pipeline
- a fine-tuned logical-form generator for temporal KBQA
- an agentic inference surface with API/CLI/demo entrypoints
- benchmark, ablation, and error analysis on `TempQuestions`

## Milestones

| Phase | Goal | Status |
|---|---|---|
| Phase 1 | Audit TempQuestions inputs and establish project memory | Completed |
| Phase 2 | Build temporal data pipeline and synthetic sample export | Completed |
| Phase 3 | Run first end-to-end train/infer trial on Vast.ai | Completed |
| Phase 4 | Lock benchmark narrative, ablation, and report artifacts | In progress |
| Phase 5 | Optional v2 improvements: relation diversity, grounding, TC support | Planned |

## Work Breakdown

### 1. Data
- standardize TempQuestions artifacts
- mine temporal facts from Zenodo Freebase triples
- generate synthetic training examples
- filter and export train-ready instruction data

### 2. Modeling
- fine-tune `LLaMA-2-7B` with LoRA
- run beam-search inference on TempQuestions test
- analyze validity, operator drift, and hallucination patterns

### 3. Deployment
- keep the baseline ChatKBQA structure
- expose inference through FastAPI, CLI, and Streamlit
- document configuration, expected services, and remote execution flow

### 4. Evaluation
- report S-expression validity
- report answer-level evaluation outcomes honestly, including blocked cases
- include a component ablation:
  - generate-only baseline
  - + fuzzy relation grounding
  - + golden entity

## Submission Checklist

- Problem definition document
- Data description document
- Model/evaluation document with baseline and error analysis
- Continual learning and monitoring plan
- Privacy/robustness and ethics writeups
- Final report and slides
- Runnable repo with tests, API/CLI/demo entrypoints, and configs
