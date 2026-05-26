# Master Plan: T-ChatKBQA

## Project Objective

Build `T-ChatKBQA` by extending `ChatKBQA` into a temporal KBQA system with two linked contributions:

1. A temporal dataset construction pipeline over Freebase-aligned QA data
2. A temporal model adaptation pipeline for training, benchmarking, and demoing the improved system

## Scope Locked In

- Keep `ChatKBQA` as the framework core
- Use `TempQuestions` as the primary benchmark
- Generate synthetic temporal data as training augmentation
- Run Freebase-heavy jobs, model training, and large evaluation on `Vasi.ai`
- Deliver a lightweight `Streamlit` demo for presentation
- Include benchmark and ablation as required evidence

## Non-Goals

- Rebuilding the system around a different temporal KGQA architecture
- Making `CronQuestions` a primary implementation target in v1
- Replacing the Freebase-based pipeline with a Wikidata-native system
- Building a production-grade multi-user deployment

## Main Workstreams

### 1. Temporal Dataset Construction

- Define temporal question types for v1:
  - `before`
  - `after`
  - `first`
  - `last`
  - explicit year/date
- Mine temporalizable Freebase facts and relations
- Generate `question + logical form + answer` triples
- Validate generated samples via executable SPARQL
- Export training-ready augmented data with stable metadata

### 2. T-ChatKBQA Model Adaptation

- Keep the `generate-then-retrieve` pipeline
- Add temporal instruction/prompting
- Train temporal-aware logical form generation
- Support temporal operators:
  - `TC`
  - `ARGMAX`
  - `ARGMIN`
  - `gt`, `ge`, `lt`, `le`

### 3. Benchmark + Ablation

- Evaluate on `TempQuestions` test
- Compare:
  - baseline `ChatKBQA`
  - human-only temporal fine-tuning
  - temporal fine-tuning with synthetic augmentation
- Report:
  - `F1`
  - `Hits@1`
  - `Accuracy`
  - temporal subset metrics
- Add at least one useful ablation on augmentation or temporal adaptation

### 4. Streamlit Demo

- Build a simple demo surface for:
  - question input
  - answer
  - temporal signal
  - logical form
  - SPARQL
  - reasoning/provenance
- Prepare stable sample questions for presentation

### 5. Project Cleanup

- Standardize dependency management
- Remove hard-coded endpoints and move them to config/env vars
- Align docs with actual implementation
- Keep local demo workflow separate from remote heavy compute workflow

## Milestones

### Milestone 1: Memory + Planning Foundation

- Create persistent project memory files
- Lock decisions, current status, and next steps

Acceptance criteria:
- New sessions can recover project context from `docs/project_memory/`

### Milestone 2: Temporal Data Construction Spec

- Define data schema
- Define temporal fact mining workflow
- Define validation/export process

Acceptance criteria:
- Temporal data pipeline is precise enough to implement without new scope decisions

### Milestone 3: Temporal Dataset Build

- Implement data construction scripts
- Produce validated augmented temporal training data

Acceptance criteria:
- Synthetic data artifacts exist and match the agreed schema

### Milestone 4: T-ChatKBQA Training + Inference

- Train temporal model on `Vasi.ai`
- Produce temporal checkpoints and inference outputs

Acceptance criteria:
- Temporal checkpoint loads and produces temporal logical forms/inference artifacts

### Milestone 5: Benchmark + Ablation

- Run benchmark on `TempQuestions`
- Produce summary metrics and ablation results

Acceptance criteria:
- Results show baseline vs temporal improvement story clearly

### Milestone 6: Streamlit Demo + Final Cleanup

- Build the presentation-ready demo
- Finalize docs and workflow notes

Acceptance criteria:
- Demo is stable enough to present and explain end-to-end
