# Data Description Document — Temporal ChatKBQA

## Data Sources

### 1. TempQuestions Dataset
- **Source**: [TempQuestions](https://github.com/jzjg99/TempQA) (Jia et al., EMNLP 2018)
- **License**: Academic research use (distributed with the TempQuestions paper)
- **Description**: 1,271 temporal questions over Freebase, each annotated with:
  - Natural language question
  - SPARQL query (executable against Freebase)
  - Answer entities
  - Topic entity MID
  - Temporal signal type (before, after, during, first, last, etc.)
- **Language**: English

### 2. Freebase Knowledge Graph
- **Source**: Google Freebase (archived; served via Virtuoso SPARQL endpoint)
- **Size**: ~3 billion facts, ~50M entities, ~1.9M relation types
- **Relevance**: The underlying KB that answers are retrieved from
- **Temporal facts**: Encoded as Compound Value Types (CVTs) with date attributes

### 3. FACC1 Entity Mentions
- **Source**: [FACC1](https://github.com/HXX97/GMT-KBQA/blob/main/data/common_data/facc1/README.md) (Gabrilovich et al.)
- **Description**: Surface-form-to-MID mappings for entity linking (e.g., "Barack Obama" → m.02mjmr)
- **License**: Freebase-compatible

## Dataset Size & Splits

| Split | Questions | Percentage |
|-------|-----------|------------|
| Train  | ~890      | 70%        |
| Test   | ~381      | 30%        |

The split follows the original TempQuestions paper convention. The official TempQuestions test split remains the held-out benchmark. However, the current v1 training run does **not** rely directly on the raw TempQuestions train split as its main supervision source because the bundled human artifacts were found to be heavily corrupted during audit. Instead, the locked v1 model was trained on a synthetic temporal dataset generated from Zenodo Freebase triples, while TempQuestions test remained the human evaluation set.

### Actual Training Data Used In V1

| Source | Role | Size |
|-------|------|------|
| TempQuestions train | Human reference and parser audit source | ~890 questions |
| TempQuestions test | Held-out benchmark | 268 evaluated questions in the locked run |
| Zenodo synthetic temporal data | Main v1 training supervision | 2,061 examples |

## Preprocessing Pipeline

### Stage 1: SPARQL → S-expression
`parse_sparql_tempquestions.py` converts each SPARQL query into an S-expression logical form. Temporal patterns recognized:
- **CVT date ranges** (FILTER NOT EXISTS with from/to) → `(TC ...)` operator
- **Ordinal/superlative** (ORDER BY DESC/ASC + LIMIT 1) → `(ARGMAX ...)` / `(ARGMIN ...)`
- **Comparisons** (FILTER with <, >, <=, >= on dates) → `gt` / `lt` / `ge` / `le` operators

### Stage 2: Merge & Label
`data_process.py --action merge_all --dataset TempQuestions` joins S-expressions with original questions, entity labels, and answer sets into a unified JSON format for auditing and benchmark preparation.

### Stage 3: LLM Instruction Format
For the original ChatKBQA path, `process_NQ.py --dataset_type TempQuestions` converts merged data into instruction-tuning format. For the current v1 temporal run, the main training data instead comes from the Zenodo-backed pipeline under `src/temporal_data/`, which exports the same instruction-tuning shape from verified synthetic samples:
```json
{
  "instruction": "Generate a Logical Form query for this temporal question, using TC operators for date constraints and ARGMAX/ARGMIN for ordinal constraints.",
  "input": "Question: { what was the last award ... }",
  "output": "(ARGMAX (JOIN ...) ...)",
  "history": []
}
```

## Train/Validation/Test Justification

The evaluation split remains the TempQuestions predefined test set because it is the project's primary benchmark. No separate validation set is carved out from TempQuestions because the human train split is both small and partially corrupted. For the current v1 experiment:

- training uses 2,061 Zenodo-derived synthetic examples
- TempQuestions test is kept as the only locked benchmark split
- hyperparameters are conservative and follow the parent ChatKBQA setup, with only lightweight tuning of epoch count, beam size, and LoRA configuration

## Known Limitations & Biases

1. **Dataset size**: 1,271 questions is small for LLM fine-tuning. We use LoRA (low-rank adaptation) to mitigate overfitting.
2. **Freebase coverage**: Freebase was archived in 2016. Facts after 2016 are absent. Temporal questions about recent events cannot be answered.
3. **English-only**: TempQuestions is English only. Temporal expressions vary across languages.
4. **Question type distribution**: The dataset skews toward explicit temporal expressions (e.g., "in 2012"). Implicit temporal questions ("during WWII") are less frequent and harder.
5. **CVT dependency**: Temporal answers depend on CVT date attributes being populated. Missing dates in Freebase cause false negatives.
6. **Entity bias**: Question entities skew toward popular topics (politicians, movies, countries). Rare entities may lack sufficient surface forms in FACC1.
7. **Human artifact corruption**: Bundled TempQuestions raw/merged artifacts contain placeholder and collapse errors, so they are not trustworthy as direct v1 supervision without auditing.
8. **Synthetic training skew**: The current Zenodo-derived dataset contains only 22 relations and omits TC/date-literal supervision, which contributes to relation hallucination and weak temporal-constraint handling.
