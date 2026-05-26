# Ethics Impact Statement — Temporal ChatKBQA

## Who Benefits from This System

**Positive beneficiaries**:
- **Researchers & analysts**: Faster access to structured knowledge without learning SPARQL. Democratizes access to large knowledge bases.
- **Non-technical professionals**: Journalists fact-checking timelines, legal professionals researching case chronologies, educators preparing materials with accurate historical facts.
- **Open-source community**: The full pipeline is open-source and reproducible, enabling further research in temporal KBQA.
- **Low-resource language communities**: The approach (LLM + retrieval) is language-agnostic — the same framework can be adapted for non-English temporal KBQA with appropriate training data.

## Who Could Be Harmed

**Potential harms and negative impacts**:
- **Misinformation amplification**: If the KB contains incorrect facts (Freebase has known inaccuracies), the system confidently returns wrong answers without indicating uncertainty. Users may not verify results.
- **Over-reliance on automated answers**: In high-stakes domains (legal, medical), users might treat the system's output as authoritative without human verification.
- **Exclusion of contemporary knowledge**: Freebase was archived in 2016. The system cannot answer questions about events after 2016, which users may not realize.
- **Privacy through aggregation**: Even though Freebase contains public data, aggregating and making it easily queryable could surface non-obvious connections about individuals that were previously hard to discover.
- **Language exclusion**: English-only training data excludes non-English speakers from benefiting.

## Bias & Fairness Risks

| Bias Type | Manifestation | Mitigation |
|-----------|--------------|------------|
| **Entity coverage bias** | Freebase has better coverage of Western entities (US politicians, Hollywood movies) than non-Western ones | Acknowledge in documentation; monitor answer rate by entity region |
| **Temporal coverage bias** | Freebase temporal facts are denser for recent history (1900-2016) than ancient history | Flag pre-1900 queries with lower confidence |
| **Gender bias** | Training data may contain more questions about male entities (historical figures, politicians) | Monitor question entity gender distribution |
| **Question type bias** | TempQuestions skews toward explicit temporal expressions; implicit temporal questions are harder | Report separate metrics for explicit vs. implicit questions |
| **LLM pre-training bias** | LLaMA-2 was trained on internet text and inherits societal biases present in that data | LoRA fine-tuning on factual KB data partially mitigates; bias auditing recommended before production use |

## Explainability for Non-Technical Stakeholders

The system is designed for transparency:
- **Provenance tracking**: Every answer includes the SPARQL query used and the temporal constraint applied. Users can trace *how* the answer was derived.
- **S-expression output**: The generated logical form is a human-readable tree structure showing the reasoning steps (JOIN, AND, ARGMAX, TC).
- **Agent reasoning trail**: `TemporalQuestionAgent.run()` returns a `reasoning_steps` list documenting each decision (signal detected → routed to X pipeline → retry N times → final answer).
- **Limitation**: The LLM's *internal* reasoning about why it generated a particular S-expression is not explainable — this is inherent to neural generation.

**Plain-language explanation for stakeholders**: "The system reads your question, detects if it's about time (using keywords like 'before', 'after', 'during'), generates a structured query in a language called S-expression, translates it to SPARQL (a database query language), runs it against Freebase (a giant encyclopedia of facts), and returns the answer. Every step is logged so you can see exactly what happened."

## Potential Misuse

| Misuse Scenario | Risk Level | Prevention |
|----------------|-----------|------------|
| Automated disinformation generation | Medium | System only retrieves from KB, doesn't generate free text |
| Surveillance / profiling | Low | KB data is public; no private data is indexed |
| Competitive intelligence abuse | Low | Same as surveillance — all data is public |
| Plagiarism / academic dishonesty | Medium | Students submitting KBQA answers as original research |
| Social engineering | Low | KB facts are already publicly accessible |

## Ethical Principles Adhered To

1. **Transparency**: Full provenance for every answer. Open-source code.
2. **Privacy by design**: No user data collection. Read-only KB access.
3. **Accountability**: Clear documentation of system limitations and failure modes.
4. **Fairness**: Acknowledged biases with monitoring proposals.
5. **Human-in-the-loop**: Designed for assistive use, not autonomous decision-making.
6. **Reproducibility**: All configs, training scripts, and evaluation code are public and version-controlled.

## Recommendations Before Production Deployment

1. Add confidence scores to answers (e.g., based on beam search rank, retrieval similarity)
2. Implement answer verification against external sources for high-stakes queries
3. Add a "I don't know" response when no executable S-expression is found or beam candidates are empty
4. Conduct a bias audit across entity gender, nationality, and temporal coverage
5. Support a feedback mechanism so users can report incorrect answers
6. Add a disclaimer: "Answers are retrieved from Freebase (archived 2016) and may be incomplete or outdated"
