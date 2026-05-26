# Submission Notes — T-ChatKBQA

> Read this first if you're evaluating this repository. It explains what runs, what doesn't, and why.

## What you can run immediately (no GPU, no Freebase)

| Command | What it does |
|---------|-------------|
| `python -m unittest tests.test_temporal_parser -v` | 19 unit tests; all pass |
| `python -m src.cli --demo` | Agent routing demo: detects temporal signals in 6 sample questions |
| `streamlit run src/streamlit_app.py` | Interactive demo using recorded trial artifacts (no live model) |

## What requires a full environment

For live inference (LLM generation + Freebase SPARQL execution), you need:

| Asset | Role | Size |
|-------|------|------|
| `meta-llama/Llama-2-7b-hf` | Base language model | ~13 GB |
| LoRA adapter checkpoint | Fine-tuned weights | `models/LLaMA2-7b-tchatkbqa-trial/checkpoint/` (~16 MB) |
| Freebase/Virtuoso SPARQL | Knowledge base backend | `localhost:8890/sparql` (~53 GB DB) |
| ELQ entity linking | Entity mention → Freebase MID | `localhost:5688/entity_linking` |

The LoRA checkpoint is included in this repo. The base model, Freebase, and ELQ are not.

## Current benchmark status

| Metric | Value | Explanation |
|--------|-------|-------------|
| Valid S-expression rate | **87.6%** (1,174/1,340) | Model generates syntactically correct logical forms |
| Answer-level F1 | **≈0.0** | Grounding bottleneck: model invents plausible-but-invalid Freebase relations |

### Why answer-level F1 is 0.0

The model was trained on only **22 Freebase relations** (from Zenodo temporal facts). At inference, it generates **177 unique relations** — most are hallucinated strings that look like Freebase relations but don't exist in the KB. When SPARQL execution hits a non-existent relation, it returns empty results.

**The bottleneck is relation grounding, not generation quality.** The syntax pipeline works; the KB lookup fails because the model has too few training relations and no retrieval-constrained decoding.

## Degraded vs. live demo

- **Streamlit demo** (`src/streamlit_app.py`): Uses recorded trial artifacts for presentation. No LLM, no GPU, no Freebase needed. Shows architecture, ablation results, and an example walkthrough with temporal signal detection.
- **CLI `--demo`** (`src/streamlit_app.py` → `src/agent.py`): Real signal detection with no external dependencies.
- **Full API** (`src/api.py`): Requires the complete stack listed above.

## Repository structure

```
src/          Core source (agent, API, CLI, pipeline, Streamlit)
data/         Temporal data, TempQuestions
models/       Trained LoRA checkpoint
configs/      YAML configs (training, inference)
tests/        19 unit tests
docs/         7 required documents + report.pdf + slides.pdf
Dockerfile    Containerized deployment
```

## Training reproduction

The v1 model was trained on Vast.ai (RTX 4090, 24 GB):

```bash
CUDA_VISIBLE_DEVICES=0 python -u LLMs/LLaMA/src/train_bash.py \
  --stage sft \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_dir LLMs/data \
  --dataset TChatKBQA_Freebase_NQ_train \
  --template llama2 \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir models/LLaMA2-7b-tchatkbqa-trial/checkpoint \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --fp16
```

Training: 14 minutes, 645 steps, loss 3.82 → 0.58.
