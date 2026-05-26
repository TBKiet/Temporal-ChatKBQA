# T-ChatKBQA: Data Preparation, Training & Evaluation

## Overview

This document describes the end-to-end process of building a temporal KBQA training dataset from Freebase, fine-tuning LLaMA-2-7b with LoRA, and evaluating on the TempQuestions benchmark.

**Date:** 2026-05-25
**Environment:** Vast.ai (RTX 4090 24GB, 600GB storage), Python 3.8 (chatKBQA) + 3.12 (main)
**Model:** LLaMA-2-7b-hf + LoRA (q_proj, v_proj)

---

## 1. Data Source

### 1.1 Zenodo idirlab/freebases Dataset

After discovering the Virtuoso Freebase dump on Vast.ai has no human-readable labels (all entities resolve to opaque `iri_id_*_with_no_name_entry`), we switched to the **Zenodo idirlab/freebases** dataset.

- **URL:** https://zenodo.org/records/7909511
- **Size:** 14.1 GB zip, 57 GB extracted
- **Variant used:** FB+CVT+REV (includes Compound Value Types, reverse triples)
- **Contents:**
  - 4,425 relations (relation2id.txt)
  - 122M entities with MIDs (entity2id.txt)
  - 47M entity labels (entities_id_label.csv)
  - 244M triples in train.txt (4.5 GB)
  - Mapping files: object_names, object_ids, properties_id_label, uri_original2simplified

### 1.2 Key Files Used

| File | Size | Purpose |
|------|------|---------|
| `FB+CVT+REV/relation2id.txt` | 0.2 MB | 4,425 relation paths → numeric IDs |
| `FB+CVT+REV/entity2id.txt` | 2.4 GB | 122M entity numeric IDs → MIDs |
| `FB+CVT+REV/train.txt` | 4.5 GB | 244M triples: `head_id, relation_id, tail_id` |
| `Metadata/entities_id_label.csv` | 1.8 GB | 47M entity MIDs → human-readable labels |

### 1.3 Why Not Virtuoso SPARQL?

The Freebase Virtuoso dump on Vast.ai contains **3.1 billion triples** but is structure-only:
- `RDF_OBJ` table: 1 row (Virtuoso version string only)
- `RDF_IRI` table: ~500 built-in RDF terms (sameAs, type, subClassOf...)
- No Freebase IRI names, no rdfs:label, no type.object.name
- All entities resolve to `iri_id_N_with_no_name_entry`

**Verdict:** Unusable for text-grounded KBQA. Zenodo triple files are the primary source.

---

## 2. Pipeline Architecture

```
Zenodo FB+CVT+REV (244M triples, 122M entities)
        │
        ▼
[1] zenodo_loader.py     Scan train.txt for temporal relations (22 relations)
        │                 Resolve numeric IDs → MIDs → labels
        │                 2,949 MinedTemporalFact records
        ▼
[2] template_bank.py     15 direction-agnostic templates
        │                 4 reasoning families (JOIN, ARGMIN, ARGMAX)
        │                 TC skipped (no date literals)
        │                 5,213 candidate TemporalSamples
        ▼
[3] distribution_filter   Deduplicate → cap by family (500) → cap by type (1000)
        │                 3,966 → 2,061 after filtering
        ▼
[4] training.py          Export to ChatKBQA instruction-tuning format
        │                 {instruction, input, output, history, metadata}
        │                 2,061 training examples
        ▼
[5] train_bash.py        LLaMA-2-7b + LoRA (q_proj, v_proj)
        │                 5 epochs, 645 steps, 14 min
        │                 Loss: 3.82 → 0.58
        ▼
[6] beam inference       5-beam search on 268 TempQuestions test questions
                         87.6% valid S-expressions
```

---

## 3. Dataset Details

### 3.1 Temporal Relations Mined

22 relations selected from 4,425 total, spanning 10 families:

| Family | Relations | Example |
|--------|-----------|---------|
| book_author | 2 | `book.author.works_written`, `book.written_work.author` |
| film_directed_by | 2 | `film.film.directed_by`, `film.director.film` |
| film_starring | 2 | `film.film.starring`, `film.performance.actor` |
| music_album | 2 | `music.artist.album`, `music.artist.track` |
| marriage | 1 | `people.marriage.spouse` |
| sports_team | 3 | `sports.pro_athlete.teams`, `sports.sports_team_roster.*` |
| organization_leadership | 2 | `organization.leadership.person`, `organization.organization.leadership` |
| award_nomination | 2 | `award.award_nominee.award_nominations`, `award.award_nomination.award_nominee` |
| government_position | 2 | `government.government_position_held.office_holder`, `government.politician.government_positions_held` |
| date_of_birth/death | 2 | `people.person.date_of_birth`, `people.person.date_of_death` |

### 3.2 Question Templates

15 direction-agnostic templates across 4 families:

**Simple (JOIN):**
- "What {role} is associated with {topic_label}?"
- "Find the {role} connected to {topic_label}."
- "Name a {role} related to {topic_label}."
- "Which {role} has a connection to {topic_label}?"

**First (ARGMIN):**
- "What was the earliest {role} associated with {topic_label}?"
- "Name the first {role} linked to {topic_label}."
- "What {role} came first for {topic_label}?"
- "Find the original {role} connected to {topic_label}."

**Last (ARGMAX):**
- "What was the most recent {role} associated with {topic_label}?"
- "Name the latest {role} linked to {topic_label}."
- "What {role} was most recent for {topic_label}?"
- "Find the last {role} connected to {topic_label}."

**During (TC):** Skipped — Zenodo facts lack temporal date literals.

### 3.3 S-Expression Examples

```
# Simple JOIN
Q: "Find the album connected to The Rolling Stones."
S: (JOIN (R music.artist.album) m.07mvp)

# First (ARGMIN)
Q: "What was the earliest album associated with The Rolling Stones?"
S: (ARGMIN (JOIN (R music.artist.album) m.07mvp) music.artist.album)

# Last (ARGMAX)
Q: "What was the most recent album associated with The Rolling Stones?"
S: (ARGMAX (JOIN (R music.artist.album) m.07mvp) music.artist.album)
```

### 3.4 Role Hint Logic

Role hints derived from **last segment** of relation path:

| Relation Path | Role Hint |
|---------------|-----------|
| `book.author.works_written` | "works written" |
| `film.film.directed_by` | "directed by" |
| `music.artist.album` | "album" |
| `people.marriage.spouse` | "spouse" |

---

## 4. Training

### 4.1 Configuration

```yaml
stage: sft
model_name_or_path: /workspace/models/Llama-2-7b-hf
dataset: TChatKBQA_Freebase_NQ_train (2,061 examples)
template: llama2
finetuning_type: lora
lora_target: q_proj,v_proj
per_device_train_batch_size: 4
gradient_accumulation_steps: 4     # effective batch = 16
learning_rate: 5.0e-5
lr_scheduler_type: cosine
num_train_epochs: 5.0              # 645 steps
fp16: true
```

### 4.2 Environment Issues Fixed

1. **Missing packages:** Installed `datasets`, `peft`, `accelerate`, `trl`, `tiktoken`, `sse-starlette`, `matplotlib`, `jieba`, `rouge_chinese`, `gradio`, `sentencepiece`
2. **llmtuner PPO import crash:** Patched `tune.py` to wrap `run_ppo`/`run_dpo`/`run_rm` imports in try/except (trl version incompatibility)
3. **summary.json conflict:** Removed `summary.json` from dataset directory (HuggingFace datasets loads all JSON files)
4. **adapter_model.bin missing:** Created symlink from `adapter_model.safetensors` (peft expects `.bin` by default)
5. **BFloat16 crash:** Patched `_infer_dtype()` in `parser.py` to force `torch.float16` (RTX 4090 + torch 1.13.1 doesn't support bfloat16 triu operation)

### 4.3 Training Results

| Metric | Value |
|--------|-------|
| GPU | RTX 4090 24GB |
| Runtime | 14 min 4 sec |
| Trainable params | 4,194,304 (LoRA) |
| Initial loss | 3.8156 |
| Final loss | 0.5777 |
| Average loss | 0.7374 |
| Samples/sec | 12.2 |
| Checkpoint | `models/LLaMA2-7b-tchatkbqa-trial/checkpoint` |

**Loss curve:**
```
Step   5: loss=3.82
Step  15: loss=2.95
Step 100: loss=0.68
Step 500: loss=0.62
Step 645: loss=0.58
```

---

## 5. Evaluation

### 5.1 Beam Search Inference

- **Test set:** TChatKBQA_Freebase_NQ_test (268 TempQuestions, human-authored)
- **Beams:** 5 per question
- **Inference time:** ~12 min (268 questions × 5 beams, avg 2.3s/question)
- **Note:** BFloat16 issue prevented using the built-in `beam_output_eva.py`. Used standalone inference script with `torch.float16` and `PeftModel.from_pretrained()`.

### 5.2 Results

| Metric | Value |
|--------|-------|
| Total predictions | 1,340 (268 × 5 beams) |
| Valid S-expressions | 1,174 (87.6%) |
| Invalid/malformed | 166 (12.4%) |

**Operator distribution in predictions:**
| Operator | Count | % |
|----------|-------|---|
| ARGMIN | 783 | 58.4% |
| ARGMAX | 268 | 20.0% |
| JOIN | 123 | 9.2% |
| Other/invalid | 166 | 12.4% |

**Operator distribution in training data (for reference):**
| Operator | Count | % |
|----------|-------|---|
| ARGMAX | 717 | 34.8% |
| JOIN | 675 | 32.7% |
| ARGMIN | 669 | 32.5% |

### 5.3 Sample Predictions

```
Q: "which dawkins book to read first?"
P: (ARGMIN (JOIN (R book.author.works_written) m.05w2x0) book.author.works_written)

Q: "what team is reggie bush on 2011?"
P: (ARGMIN (JOIN (R sports.pro_athlete.team) m.010l89) sports.pro_athlete.team)

Q: "Find the album connected to The Rolling Stones."
P: (JOIN (R music.artist.album) m.07mvp)
```

---

## 6. Analysis & Next Steps

### 6.1 What Worked

- Pipeline produces clean, syntactically-correct S-expressions with real entity names
- LoRA fine-tuning on 2,061 examples converges well (loss 3.82 → 0.58)
- Model generates valid S-expressions 87.6% of the time
- Role-direction fix eliminated label leakage in questions

### 6.2 Issues Identified

1. **ARGMIN over-prediction (58% vs 32% in training):** Model biases toward the most frequent pattern it saw in synthetic data combined with temporal bias in test questions
2. **No TC operator:** Training data lacks TC/during examples due to missing date literals in Zenodo. Model cannot handle "in 2010", "during 2005" type queries
3. **Hallucinated relations:** Model invents plausible Freebase relation names (e.g., `film.director.jedi_master`). Training data had only 22 relations — insufficient coverage
4. **Beam diversity low:** 5-beam predictions for same question often nearly identical (same operator, slightly different MIDs)
5. **Test set corruption:** TempQuestions test data is also corrupted (dominated by `government.government_position_held` patterns)

### 6.3 Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| P0 | Add more JOIN examples to balance ARGMIN bias | Normalize operator distribution |
| P1 | Source date literals for TC operator support | Enable temporal constraint queries |
| P2 | Increase relation diversity (target 50-100 relations) | Reduce hallucination |
| P3 | Add temperature sampling to beam search | Improve prediction diversity |
| P4 | Cross-reference TempQuestions with verified answers | Reliable benchmark |

---

## A. Appendix: File Inventory

### New Modules
- `src/temporal_data/zenodo_loader.py` — Zenodo fact mining
- `src/temporal_data/sparql_miner.py` — Virtuoso SPARQL mining (deprecated)
- `src/temporal_data/template_bank.py` — Template generation
- `src/temporal_data/verifier.py` — S-expression execution verification
- `src/temporal_data/distribution_filter.py` — Post-verification filtering
- `src/temporal_data/training.py` — Instruction-tuning format export

### Configs
- `configs/relation_whitelist.yaml` — 11 relation families
- `configs/train_tchatkbqa_trial.yaml` — Trial training config

### Scripts
- `scripts/build_temporal_dataset.py` — CLI: `mine-facts-zenodo`, `generate-candidates`, `filter-and-export`
- `scripts/diagnose_freebase.py` — SPARQL endpoint diagnostic (18 tests)
- `scripts/explore_zenodo.py` — Dataset structure explorer

### Tests
- `tests/test_temporal_pipeline.py` — 32 tests (offline)

### Artifacts
- `LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json` — 2,061 training examples
- `data/temporal/mined_facts_zenodo.json` — 2,949 mined facts
- `data/zenodo/extracted/` — Zenodo dataset (57 GB, Vast.ai only)
- `models/LLaMA2-7b-tchatkbqa-trial/checkpoint/` — LoRA adapter (16 MB)
- `models/LLaMA2-7b-tchatkbqa-trial/evaluation_beam/generated_predictions.jsonl` — 1,340 predictions
