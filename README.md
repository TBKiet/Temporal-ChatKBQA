# Temporal ChatKBQA

**Agentic Temporal Knowledge Base Question Answering** — extending [ChatKBQA](https://github.com/LHRLAB/ChatKBQA) (ACL 2024 Findings) to handle time-sensitive queries over Freebase with temporal operators, agentic routing, and production-ready deployment.

> **NLP Subject Final Project** — based on ChatKBQA: *"A Generate-then-Retrieve Framework for KBQA with Fine-tuned LLMs"* by Haoran Luo, Haihong E, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guanting Dong, Meina Song, Wei Lin, Yifan Zhu, Luu Anh Tuan. **Findings of ACL 2024** [\[paper\]](https://aclanthology.org/2024.findings-acl.122/).

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://aclanthology.org/2024.findings-acl.122.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deploy-009688.svg)](src/api.py)

## Overview

Temporal ChatKBQA is a **Generate-then-Retrieve** KBQA framework extended with temporal reasoning capabilities. It fine-tunes LLaMA-2 to generate S-expression logical forms from natural language temporal questions, then uses unsupervised retrieval to replace entity/relation placeholders with actual Freebase IDs before executing SPARQL queries. Temporal operators (TC, ARGMAX/ARGMIN, gt/ge/lt/le) handle time constraints such as before/after, first/last, and date ranges.

### Key Features

- **Agentic AI**: 4-step reasoning — detect temporal signals → route → execute → refine with iterative retry
- **Full provenance**: Every answer includes SPARQL used, temporal constraint, and reasoning trail
- **Production-ready**: REST API (FastAPI) + CLI + Docker deployment
- **Config-driven**: YAML-based configuration for training and inference
- **Tested**: Unit tests for signal detection, agent routing, and SPARQL parsing

### Documentation

| Document                                           | Description                                      |
| -------------------------------------------------- | ------------------------------------------------ |
| [Problem Definition](docs/problem_definition.md)   | Business context, stakeholders, success metrics  |
| [Data Description](docs/data_description.md)       | Data sources, preprocessing, splits, limitations |
| [Model & Evaluation](docs/model_evaluation.md)     | Architecture, training, metrics, error analysis  |
| [Continual Learning](docs/continual_learning.md)   | Data collection, retraining, drift detection     |
| [Privacy & Robustness](docs/privacy_robustness.md) | PII handling, adversarial inputs, mitigations    |
| [Project Plan](docs/project_plan.md)               | Timeline, milestones, and submission scope       |
| [Ethics Statement](docs/ethics_statement.md)       | Beneficiaries, harms, bias, explainability       |
| [Written Report](docs/report.pdf)                     | Comprehensive final report (PDF)                 |
| [Presentation Slides](docs/slides.pdf)                | Presentation slide deck (PDF)                    |

Note: the trained LoRA checkpoint is not committed to the repository. The code, configs, and runbooks are included so the experiment can be reproduced; lightweight unit tests run locally, while full training and Freebase-heavy evaluation are expected to run on Vast.ai.

![](./figs/F1.drawio.png)

##  General Setup

### Environment Setup

```bash
conda create -n chatKBQA python=3.8
conda activate chatKBQA
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt

# Additional dependencies for Temporal KBQA deployment
pip install -r requirements.txt
```

### TempQuestions Dataset Download

```bash
# Download the TempQuestions dataset
python scripts/download_tempquestions.py
```

### Required External Services

These must be running for the full pipeline to function:

| Service                  | URL/Port                        | Purpose                       |
| ------------------------ | ------------------------------- | ----------------------------- |
| Freebase Virtuoso SPARQL | `localhost:8890/sparql`         | KB query execution            |
| Freebase ODBC            | port 13001                      | Direct KB access via ODBC     |
| ELQ Entity Linking       | `localhost:5688/entity_linking` | Entity mention → Freebase MID |

Configured in `config.py`. For Freebase setup instructions, see [Freebase-Setup](Freebase-Setup/).

###  Freebase KG Setup

Below steps are according to [Freebase Virtuoso Setup](https://github.com/dki-lab/Freebase-Setup).
#### How to install virtuoso backend for Freebase KG.

1. Clone from `dki-lab/Freebase-Setup`:
```
cd Freebase-Setup
```

2. Processed [Freebase](https://developers.google.com/freebase) Virtuoso DB file can be downloaded from [Dropbox](https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip) or [Baidu Netdisk](https://pan.baidu.com/s/1F0ytk74p8PGQ0tgAMu9--g?pwd=cp1j) (WARNING: 53G+ disk space is needed):
```
tar -zxvf virtuoso_db.zip
```

3. Managing the Virtuoso service:

To start service at `localhost:3001/sparql`:
```
python3 virtuoso.py start 3001 -d virtuoso_db
```

and to stop a currently running service at the same port:
```
python3 virtuoso.py stop 3001
```

A server with at least 100 GB RAM is recommended.

#### Download FACC1 mentions for Entity Retrieval.

- Download the mention information (including processed [FACC1](https://github.com/HXX97/GMT-KBQA/blob/main/data/common_data/facc1/README.md) mentions and all entity alias in Freebase) from [OneDrive](https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye) or [Baidu Netdisk](https://pan.baidu.com/s/1qbKP2DV1lo9jlYoBxpyTHA?pwd=qzb7) to `data/common_data/facc1/`.

```
ChatKBQA/
└── data/
    ├── common_data/
        ├── facc1/
            ├── entity_list_file_freebase_complete_all_mention
            └── surface_map_file_freebase_complete_all_mention
```

## Dataset

Experiments are conducted on 2 KBQA benchmarks WebQSP, CWQ.

### WebQSP

[WebQSP](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) dataset has been downloaded under `data/WebQSP/origin`.

```
ChatKBQA/
└── data/
    ├── WebQSP
        ├── origin
            ├── WebQSP.train.json
            └── WebQSP.test.json
```

### CWQ

[CWQ](https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AACuu4v3YNkhirzBOeeaHYala) dataset has been downloaded under `data/CWQ/origin`.
```
ChatKBQA/
└── data/
    ├── CWQ
        ├── origin
            ├── ComplexWebQuestions_train.json
            ├── ComplexWebQuestions_dev.json
            └── ComplexWebQuestions_test.json
```


## Data Processing

(1) **Parse SPARQL queries to S-expressions**

- WebQSP:

Run `python parse_sparql_webqsp.py` and the augmented dataset files are saved as `data/WebQSP/sexpr/WebQSP.test[train].json`.

- CWQ:

Run `python parse_sparql_cwq.py` and the augmented dataset files are saved as `data/CWQ/sexpr/CWQ.test[train].json`.


(2) **Prepare data for training and evaluation**

- WebQSP:

Run `python data_process.py --action merge_all --dataset WebQSP --split test` and `python data_process.py --action merge_all --dataset WebQSP --split train`. The merged data file will be saved as `data/WebQSP/generation/merged/WebQSP_test[train].json`.

Run `python data_process.py --action get_type_label_map --dataset WebQSP --split train`. The merged data file will be saved as `data/WebQSP/generation/label_maps/WebQSP_train_type_label_map.json`.

- CWQ:

Run `python data_process.py --action merge_all --dataset CWQ --split test` and `python data_process.py --action merge_all --dataset CWQ --split train`. The merged data file will be saved as `data/CWQ/generation/merged/CWQ_test[train].json`.

Run `python data_process.py --action get_type_label_map --dataset CWQ --split train`. The merged data file will be saved as `data/CWQ/generation/label_maps/CWQ_train_type_label_map.json`.

**Note:** You can also get the ChatKBQA processed data from [TeraBox](https://1024terabox.com/s/1T3ckf32YJJ-SJCJd1zUNlg) or [Baidu Netdisk](https://pan.baidu.com/s/1ikNCCCtYd9izN0Ok3ozdZA?pwd=j6jk), which should be set in `data/`.
```
ChatKBQA/
└── data/
    ├── CWQ/
        ├── generation/
        ├── origin/
        └── sexpr/
    └── WebQSP/
        ├── generation/
        ├── origin/
        └── sexpr/
```

(3) **Prepare data for LLM model**

- WebQSP:

Run `python process_NQ.py --dataset_type WebQSP`. The merged data file will be saved as `LLMs/data/WebQSP_Freebase_NQ_test[train]/examples.json`.

- CWQ:

Run `python process_NQ.py --dataset_type CWQ` The merged data file will be saved as `LLMs/data/CWQ_Freebase_NQ_test[train]/examples.json`.

**Note:** You can also get the processed ChatKBQA SFT data from [TeraBox](https://1024terabox.com/s/1XHzWc5qQoaq2ncMfh81PxQ) or [Baidu Netdisk](https://pan.baidu.com/s/1dazyWIQ8nYt5YiLt8yjFSw?pwd=uvd9), which should be set in `LLMs/data`.
```
ChatKBQA/
└── LLMs/
    ├── data/
        ├── CWQ_Freebase_NQ_test/
        ├── CWQ_Freebase_NQ_train/
        ├── WebQSP_Freebase_NQ_test/
        ├── WebQSP_Freebase_NQ_train/
        └── dataset_info.json
```

## Fine-tuning, Retrieval and Evaluation

The following is an example of [LLaMa2-7b](README.md) fine-tuning and retrieval (num_beam = 15) on WebQSP and [LLaMa2-13b](README.md) fine-tuning and retrieval (num_beam = 8) on CWQ, respectively.

(1) **Train and test LLM model for Logical Form Generation**

- WebQSP:

Train LLMs for Logical Form Generation:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-7b-hf --do_train  --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_train --template llama2  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 100.0 --plot_loss  --fp16 >> train_LLaMA2-7b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_dir LLMs/data --dataset WebQSP_Freebase_NQ_test --template llama2 --finetuning_type lora --checkpoint_dir Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/checkpoint --num_beams 15 >> predbeam_LLaMA2-7b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl
```

- CWQ:

Train LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u LLMs/LLaMA/src/train_bash.py --stage sft --model_name_or_path meta-llama/Llama-2-13b-hf --do_train  --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_train --template default  --finetuning_type lora --lora_target q_proj,v_proj --output_dir Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --overwrite_cache --per_device_train_batch_size 4 --gradient_accumulation_steps 4  --lr_scheduler_type cosine --logging_steps 10 --save_steps 1000 --learning_rate 5e-5  --num_train_epochs 10.0 --plot_loss  --fp16 >> train_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Beam-setting LLMs for Logical Form Generation:
```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u LLMs/LLaMA/src/beam_output_eva.py --model_name_or_path meta-llama/Llama-2-13b-hf --dataset_dir LLMs/data --dataset CWQ_Freebase_NQ_test --template default --finetuning_type lora --checkpoint_dir Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/checkpoint --num_beams 8 >> predbeam_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```
```bash
python run_generator_final.py --data_file_name Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/generated_predictions.jsonl
```

(2) **Evaluate KBQA result with Retrieval**

- WebQSP:

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_LLaMA2-7b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u eval_final.py --dataset WebQSP --pred_file Reading/LLaMA2-7b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_LLaMA2-7b_WebQSP_Freebase_NQ_lora_epoch100.txt 2>&1 &
```

- CWQ:

Evaluate KBQA result with entity-retrieval and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=4 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json >> predfinal_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

Evaluate KBQA result with golden-entities and relation-retrieval:
```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u eval_final_cwq.py --dataset CWQ --pred_file Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json --golden_ent >> predfinalgoldent_LLaMA2-13b_CWQ_Freebase_NQ_lora_epoch10.txt 2>&1 &
```

**Note:** You can also get the ChatKBQA checkpoints and evaluations from [TeraBox](https://1024terabox.com/s/1ZCCauq7KC5vys-zSjdcNdA) or [Baidu Netdisk](https://pan.baidu.com/s/1I5j_FktPF5R0hI1In1qlkQ?pwd=53p5), which should be set in `Reading/`.
```
ChatKBQA/
└── Reading/
    ├── LLaMA2-7b/
        └── WebQSP_Freebase_NQ_lora_epoch100/
            ├── checkpoint/
            └── evaluation_beam/
    └── LLaMA2-13b/
        └── CWQ_Freebase_NQ_lora_epoch10/
            ├── checkpoint/
            └── evaluation_beam/
```

## BibTex

If you find this work is helpful for your research, please cite:

```bibtex
@inproceedings{luo2024chatKBQA,
    title = "{C}hat{KBQA}: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models",
    author = "Luo, Haoran  and
      E, Haihong  and
      Tang, Zichen  and
      Peng, Shiyao  and
      Guo, Yikai  and
      Zhang, Wentai  and
      Ma, Chenghao  and
      Dong, Guanting  and
      Song, Meina  and
      Lin, Wei  and
      Zhu, Yifan  and
      Luu, Anh Tuan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.122",
    pages = "2039--2056",
    abstract = "Knowledge Base Question Answering (KBQA) aims to answer natural language questions over large-scale knowledge bases (KBs), which can be summarized into two crucial steps: knowledge retrieval and semantic parsing. However, three core challenges remain: inefficient knowledge retrieval, mistakes of retrieval adversely impacting semantic parsing, and the complexity of previous KBQA methods. To tackle these challenges, we introduce ChatKBQA, a novel and simple generate-then-retrieve KBQA framework, which proposes first generating the logical form with fine-tuned LLMs, then retrieving and replacing entities and relations with an unsupervised retrieval method, to improve both generation and retrieval more directly. Experimental results show that ChatKBQA achieves new state-of-the-art performance on standard KBQA datasets, WebQSP, and CWQ. This work can also be regarded as a new paradigm for combining LLMs with knowledge graphs (KGs) for interpretable and knowledge-required question answering.",
}
```

For further questions, please contact: haoran.luo@ieee.org.

---

## Temporal KBQA Extension

This repository extends ChatKBQA with **temporal question answering** over Freebase, supporting time-sensitive queries (before/after/during/first/last/year).

### Dataset

Download [TempQuestions](https://github.com/jzjg99/TempQA) (Freebase-based, ~1,271 temporal questions) and place files in `data/TempQuestions/origin/`. Or use the download script:

```bash
python scripts/download_tempquestions.py
```

### Temporal Data Preprocessing

```bash
# 1. Parse TempQuestions SPARQL → S-expressions with TC/ARGMAX operators
python parse_sparql_tempquestions.py

# 2. Merge data with entity/relation labels
python data_process.py --action merge_all --dataset TempQuestions --split train
python data_process.py --action merge_all --dataset TempQuestions --split test

# 3. Build LLM instruction-tuning data
python process_NQ.py --dataset_type TempQuestions
```

### Temporal LLM Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python -u LLMs/LLaMA/src/train_bash.py \
  --stage sft \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --do_train \
  --dataset_dir LLMs/data \
  --dataset TempQuestions_Freebase_NQ_train \
  --template llama2 \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir models/LLaMA2-7b-temporal/checkpoint \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --learning_rate 5e-5 \
  --num_train_epochs 50.0 \
  --fp16
```

### Beam Search Inference

```bash
CUDA_VISIBLE_DEVICES=0 python -u LLMs/LLaMA/src/beam_output_eva.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_dir LLMs/data \
  --dataset TempQuestions_Freebase_NQ_test \
  --template llama2 \
  --finetuning_type lora \
  --checkpoint_dir models/LLaMA2-7b-temporal/checkpoint \
  --num_beams 15
```

### Temporal Evaluation

```bash
python eval_temporal.py \
  --pred_file Reading/LLaMA2-7b-temporal/evaluation_beam/beam_test_top_k_predictions.json \
  --dataset TempQuestions
```

Metrics: **F1**, **Hits@1**, **Accuracy**, and **Temporal F1** (subset of temporal questions only).

### Deployment

**REST API:**
```bash
pip install fastapi uvicorn pyyaml
# Edit configs/inference.yaml with your model path
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Test:
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who was US president before Obama?"}'
```

**CLI:**
```bash
# Agent routing demo (no LLM/Freebase required):
python -m src.cli --demo

# Answer a single question:
python -m src.cli --question "Who held the position of US president during WWII?"

# Interactive mode:
python -m src.cli --interactive
```

**Docker:**
```bash
docker build -t temporal-KBQA .
docker run -p 8000:8000 temporal-KBQA
```

**Streamlit Demo:**
```bash
pip install streamlit
streamlit run src/streamlit_app.py
```

The Streamlit app provides an interactive presentation/demo surface with:
- Real temporal signal detection from `src/agent.py`
- Recorded trial artifacts from the locked v1 experiment
- Architecture and ablation summaries
- Degraded/offline fallback when full LLM + Freebase assets are not available

### Agentic AI Component

`src/agent.py` implements `TemporalQuestionAgent` with 4-step reasoning:
1. **Detect** temporal signals in the question
2. **Route** to temporal or standard KBQA pipeline
3. **Refine** iteratively if no answer is found (relaxing temporal constraints)
4. **Return** answer with full provenance (SPARQL + temporal constraint used)

### Project Structure

```
ChatKBQA/
├── src/                     # Deployment + agentic code
│   ├── agent.py             # Agentic routing component
│   ├── pipeline.py          # End-to-end inference pipeline
│   ├── api.py               # FastAPI REST API
│   ├── cli.py               # CLI interface
│   └── streamlit_app.py     # Streamlit demo
├── configs/                 # Configuration files
│   ├── train_temporal.yaml  # LLM training config
│   └── inference.yaml       # Inference config
├── tests/                   # Unit tests
│   └── test_temporal_parser.py
├── data/TempQuestions/      # Temporal dataset
├── models/                  # Model checkpoints
├── parse_sparql_tempquestions.py  # SPARQL→S-expr for TempQuestions
└── eval_temporal.py               # Temporal evaluation script
```

### Running Tests

```bash
python -m pytest tests/ -v
# Or without pytest:
python -m unittest tests.test_temporal_parser -v
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning), [SimCSE](https://github.com/princeton-nlp/SimCSE), [GMT-KBQA](https://github.com/HXX97/GMT-KBQA) and [DECAF](https://github.com/awslabs/decode-answer-logical-form). Thanks for their wonderful works.
