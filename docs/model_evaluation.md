# Model Architecture & Evaluation — Temporal ChatKBQA

## Model Selection

### Architecture: LLaMA-2-7B with LoRA Fine-Tuning

We use **Meta's LLaMA-2-7B** (decoder-only transformer) fine-tuned with **LoRA** (Low-Rank Adaptation) for semantic parsing — converting natural language questions into S-expression logical forms with temporal operators.

**Why this model fits the business problem**:
1. **Strong instruction-following**: LLaMA-2 performs well on structured generation tasks after SFT
2. **Parameter efficiency**: LoRA fine-tunes only ~0.1% of parameters (q_proj, v_proj), enabling training on a single GPU with ~16GB VRAM
3. **Temporal compatibility**: The model learns to emit TC, ARGMAX, and ARGMIN operators from instruction prompts — no architecture changes needed
4. **Open-source**: No API costs, deployable on-premise for data-sensitive environments
5. **Proven on parent task**: ChatKBQA (ACL 2024) validated LLaMA-2-7B + LoRA on WebQSP and CWQ (SOTA results)

### Model Details

| Component | Specification |
|-----------|--------------|
| Base model | meta-llama/Llama-2-7b-hf |
| Fine-tuning | LoRA (r=8, alpha=16) |
| Target modules | q_proj, v_proj |
| Trainable params | ~4.2M (0.06% of 7B) |
| Precision | FP16 |
| Training epochs | 5 |
| Batch size | 4 per GPU × 4 gradient accumulation = 16 effective |
| Learning rate | 5e-5 (cosine schedule) |
| Optimizer | AdamW |
| Warmup ratio | 0.05 |

### Inference

| Parameter | Value |
|-----------|-------|
| Beam size | 5 |
| Max new tokens | 256 |
| Generation strategy | Beam search with early stopping |
| Device | CUDA (GPU) |

## Training Procedure

1. **Data preparation**: TempQuestions SPARQL → S-expression via `parse_sparql_tempquestions.py` → instruction-tuning format via `process_NQ.py`
2. **Training**: Run `LLMs/LLaMA/src/train_bash.py` with the trial config; the completed run is confirmed by trainer artifacts at 5 epochs / 645 steps
3. **Inference**: Run `LLMs/LLaMA/src/beam_output_eva.py` with beam_size=5
4. **Post-processing**: `run_generator_final.py` converts beam output to top-k predictions JSON
5. **Evaluation**: `eval_temporal.py` executes each candidate against Freebase, computes metrics

## Evaluation Results (Trial 1 — Zenodo Synthetic Data)

*Run date: 2026-05-25 (inference) / 2026-05-26 (Zenodo answer-level eval)*

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training examples | 2,061 (from 22 Zenodo relations: JOIN 675, ARGMAX 717, ARGMIN 669) |
| Model | LLaMA-2-7B + LoRA (q_proj, v_proj, r=8, alpha=16) |
| Trainable params | ~4.2M |
| Epochs | 5 (645 steps) |
| Batch size | 4 × 4 gradient accumulation = 16 effective |
| Learning rate | 5e-5 (cosine schedule) |
| Precision | FP16 |
| GPU | RTX 4090 24GB |
| Training time | 14 min |
| Loss | 3.82 → 0.58 |

### S-expression Generation Quality

Inference: 268 TempQuestions test questions × 5-beam search (1,340 predictions total)

| Metric | Value |
|--------|-------|
| Valid S-expressions | 1,174 / 1,340 (**87.6%**) |
| Invalid / malformed | 166 (12.4%) |

**Operator distribution:**

| Operator | Predictions | Training Data | Drift |
|----------|-------------|---------------|-------|
| ARGMIN | 783 (58.4%) | 669 (32.5%) | +25.9pp |
| ARGMAX | 268 (20.0%) | 717 (34.8%) | −14.8pp |
| JOIN | 123 (9.2%) | 675 (32.7%) | −23.5pp |
| Invalid | 166 (12.4%) | — | — |

### Answer-Level Evaluation (Zenodo Triple Lookup)

Evaluator: `scripts/eval_zenodo.py` — parses S-expressions, maps to Zenodo numeric IDs, searches 244M triples for answers, converts MIDs to labels via 47M-label CSV, compares with TempQuestions gold.

| Metric | Value |
|--------|-------|
| Zenodo relations available | 4,425 |
| Zenodo entities available | 122M |
| Predictions with relation+entity in Zenodo | 5 / 995 (0.5%) |
| Questions with answers found | 1 / 268 (0.4%) |
| **Raw F1 / Hits@1 / Accuracy** | **0.0 / 0.0 / 0.0** |

### Baseline Comparison

For the submitted v1, the baseline is the **generate-only** system: the fine-tuned model emits S-expressions without any additional grounding assistance beyond its own output. This is the cleanest deployable baseline because it isolates what the LoRA generator can do by itself on TempQuestions.

We then compare that baseline against two progressively stronger variants:

- `+ fuzzy relation grounding`: adds a retrieval-style post-processing step that maps generated relation strings to the closest supported relation
- `+ golden entity`: keeps the relation grounding step and replaces the entity with the gold topic entity to isolate the next bottleneck

This is a component baseline rather than a full re-run of unmodified ChatKBQA on TempQuestions. That limitation should be stated explicitly in the report.

| Configuration | Entity | Relation | Answered | F1 | Insight |
|--------|--------|----------|----------|----|---------|
| Generate-only | Model (0%) | Model (0.5%) | 0.4% | 0.0 | No retrieval → fail almost completely |
| + Fuzzy relation grounding | Model (5.8%) | Fuzzy (88.9%) | 0.7% | 0.0 | Relation fixed, entity still hallucinated |
| + Golden entity | Gold (100%) | Fuzzy (60.2%) | 2.6% | 0.0 | Entity fixed, but relation relevance and KB coverage still weak |

## Error Analysis (Actual — from Trial 1)

| Error Type | Impact | Detail |
|------------|--------|--------|
| Relation hallucination | **Critical** (~85%) | 177 unique relations generated, most non-existent. Examples: `film.director.jedi_master`, `foreign.exchange.rate.history.currency` |
| ARGMIN over-prediction | High (58.4%) | Model biases toward most frequent training operator |
| Invalid S-expr syntax | Medium (12.4%) | Truncated, malformed, or recursive patterns |
| No TC support | Limiting (100%) | Training data lacks date literals → model cannot learn temporal constraints |
| Beam diversity low | Medium | 5-beam predictions often nearly identical for same question |

### Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| P0 | Increase relation diversity to 50-100 relations | Reduce hallucination from ~85% to <20% |
| P1 | Add retrieval/grounding step to constrain relations to whitelist | Eliminate hallucinated relations entirely |
| P1 | Source date literals for TC operator support | Enable temporal constraint queries |
| P2 | Balance operator distribution in training data | Fix ARGMIN/ARGMAX/JOIN drift |
| P2 | Add temperature/diversity sampling to beam search | Improve prediction diversity |

## Trade-off Discussion

### Accuracy vs. Speed
- The locked v1 run used beam size 5
- Inference time was about 12 minutes for 268 questions, or ~2.3s/question on RTX 4090
- Larger beams may improve coverage, but they were not the submitted configuration
- Lazy LLM loading in API mode defers ~10s startup to first request
- The current system is fast enough for demo/API use, but answer quality is not yet strong enough for production deployment without grounding improvements

### Model Complexity vs. Maintainability
- LoRA adds only ~4.2M parameters (vs. 7B base) → checkpoint size is ~16MB, easy to version and deploy
- Full fine-tuning would yield slightly better accuracy (+2-3 F1 points) but requires multi-GPU setup and 14GB checkpoint storage
- LoRA chosen for practical deployability

## Submission Positioning

The current evidence supports the following honest claim:

- the system is implemented end-to-end
- the generator learns syntactic logical-form structure reasonably well
- the main failure boundary is downstream grounding, not just syntax generation

The current evidence does **not** support claiming successful answer-level temporal QA on TempQuestions yet, because the locked v1 answer-level metrics remain `0.0 / 0.0 / 0.0`.
