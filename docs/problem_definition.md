# Problem Definition Document — Temporal ChatKBQA

## Business Context & Motivation

Organizations across finance, legal, healthcare, and intelligence sectors routinely need to answer time-sensitive questions over structured knowledge: *"Who was CEO of Apple before Tim Cook?"*, *"Which drugs were approved by the FDA after 2020 for treating Alzheimer's?"*, *"What companies did Microsoft acquire during the 2008 recession?"*

Traditional search engines and chatbots struggle with these queries because they require:
- Understanding **temporal signals** (before, after, during, first, last, since)
- Reasoning over **time-annotated knowledge** (facts that are true only within date ranges)
- Executing **structured queries** against knowledge bases that model time

Knowledge bases like Freebase encode temporal information via Compound Value Types (CVTs) — facts with `from`/`to` date attributes. However, converting a natural language temporal question into an executable SPARQL query over these structures is non-trivial. It requires: (1) parsing the temporal intent, (2) generating the correct logical form with temporal operators (TC, ARGMAX, ARGMIN), (3) grounding entity/relation placeholders to KB IDs, and (4) binding timestamp constraints for temporal scoping.

ChatKBQA (Luo et al., ACL 2024 Findings) addressed the non-temporal version of this problem with a generate-then-retrieve framework. **Temporal ChatKBQA** extends this to the temporal domain, enabling organizations to query knowledge bases with time-sensitive natural language questions.

## Target Users & Stakeholders

| Stakeholder | Role | Value |
|---|---|---|
| Data analysts / researchers | Query Freebase with natural language instead of SPARQL | 10x faster query formulation |
| Business intelligence teams | Answer time-sensitive business questions from structured data | Reduced dependency on DB engineers |
| Legal/compliance officers | Trace entity relationships with temporal constraints | Accurate timeline reconstruction |
| Knowledge graph engineers | Deploy KBQA as an internal service | Reusable, API-first architecture |

## Problem Description

**Input**: Natural language question with temporal constraints over Freebase.
**Output**: A set of answer entities (or timestamps) retrieved from Freebase.

**Example**:
- Q: *"What was the last award Harry Potter and the Philosopher's Stone received?"*
- Temporal signal: `last` (ordinal — most recent in chronological order)
- Generated logical form: `(ARGMAX (JOIN (R award.award_nominee.award_nominations) m.02q4m) award.award_nomination.award_nomination_year)`
- SPARQL executes ORDER BY DESC + LIMIT 1 over CVT date attribute
- Answer: `Nestlé Children's Book Prize`

**Why NLP is required**: Freebase contains ~3 billion facts across ~50M entities. Manual SPARQL authoring is slow and error-prone. The temporal operators (TC, ARGMAX, ARGMIN) are particularly unintuitive in SPARQL. NLP bridges natural language to structured queries, making temporal KBs accessible to non-technical users.

## Success Metrics

### Business Metrics
| Metric | Target | Measurement |
|---|---|---|
| Query formulation time reduction | 90% reduction vs. manual SPARQL | Time from question to answer |
| User self-service rate | 80% of temporal queries answered without DB engineer | Query log analysis |
| Answer coverage | 70% of TempQuestions answered correctly | Evaluation benchmark |

### Technical Metrics
| Metric | Target | How Measured |
|---|---|---|
| F1 Score (TempQuestions) | > 0.60 | Set overlap between predicted and gold answers |
| Hits@1 (TempQuestions) | > 0.50 | Top-ranked answer in gold set |
| Temporal F1 (subset) | > 0.55 | F1 on temporal-only questions |
| Inference latency | < 5 seconds per question | End-to-end wall clock |
| API availability | 99.9% uptime | Health check monitoring |
