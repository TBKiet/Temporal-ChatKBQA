# Privacy & Robustness Analysis — Temporal ChatKBQA

## Data Privacy & Security

### Personal Identifiable Information (PII) Assessment

**Training data (TempQuestions)**: The dataset contains only natural language questions, SPARQL queries, and Freebase entity IDs. No real user data, no email addresses, no IP addresses, no personal messages. Questions reference public figures (politicians, actors, athletes) via Freebase — this is already public knowledge.

**User queries (API requests)**: User-submitted questions could theoretically contain PII, but:
- Questions are processed ephemerally and not stored by default
- The system does not associate queries with user identities
- No authentication is required (deployment is assumed internal/VPN)

**Freebase results**: Returned entities are Freebase MIDs and labels — all public data derived from Wikipedia and other open sources.

### Anonymization & Minimization Strategies

| Data Category | Strategy |
|--------------|----------|
| API request logs | Strip client IPs; retain only question text and timestamp |
| SPARQL query logs | Log only the SPARQL, not the originating question (reversible only via hash) |
| User feedback | Store against anonymous session token, not user identity |
| Training data | Already anonymized — no user-derived data |
| Model checkpoints | No memorization risk for LoRA (only ~4M params trained on public facts) |

### Data Storage & Retention

- **Query logs**: Retained for 30 days for debugging, then auto-purged
- **Feedback data**: Retained indefinitely (opt-in, anonymized)
- **Model checkpoints**: Versioned in git-lfs or cloud storage; no PII exposure risk
- **No external data sharing**: All processing is local; no third-party APIs are called at inference time

## Model Robustness

### Known Failure Modes

1. **Out-of-domain questions**: Questions that don't map to Freebase (e.g., "What's the best pizza in New York?") will produce empty results or nonsensical S-expressions. The agent's retry mechanism mitigates this by falling back to standard mode.

2. **Adversarial inputs**:
   - **Malformed SPARQL injection**: If a user crafts a question designed to produce malicious SPARQL, the ODBC executor only has read access to Freebase — no INSERT/UPDATE/DELETE is possible.
   - **Prompt injection**: The LLM prompt uses fixed instruction templates. A question containing "ignore previous instructions" could theoretically affect generation but would result in unparseable S-expressions that fail at SPARQL conversion.
   - **Extremely long questions**: Tokenized input is capped by the model's context window (4096 tokens for LLaMA-2).

3. **Edge cases in temporal parsing**:
   - Multiple temporal signals in one question ("first president after WWI before 1950") — the LLM may handle only one
   - Relative dates ("three years ago") — no knowledge of current date
   - Vague temporal expressions ("recently", "earlier", "back then") — not mapped to concrete ranges
   - Year-only vs. full date ambiguity — Freebase dates may be YYYY, YYYY-MM, or YYYY-MM-DD

### Robustness Testing Results

| Input Type | Example | Expected Behavior |
|-----------|---------|-------------------|
| Normal temporal | "Who was president before Obama?" | Correct answer via temporal pipeline |
| Non-temporal factual | "What is H2O?" | Routes to standard pipeline, may return empty (not about Freebase entities) |
| Gibberish | "asdfghjkl 12345" | Empty S-expression → empty result → graceful empty response |
| Multiple temporal signals | "Who first visited France after 2000 before 2010?" | LLM may handle one constraint; partial answer |
| Extremely long question | 500+ word query | Truncated to 4096 tokens; may degrade |
| SQL injection attempt | "'; DROP TABLE--" as question | Contained within string literal; ODBC read-only |
| Non-English | "Wer war Präsident vor Obama?" | German input → LLM may still generate S-expression (multilingual training data) |

### Mitigation Strategies

| Risk | Mitigation |
|------|-----------|
| Empty results on valid questions | Agent retry with relaxed temporal constraint |
| Incorrect S-expression | Beam search generates 15 candidates; best executable one selected |
| Entity linking failure | Fallback to direct SPARQL execution (MID placeholders may match) |
| KB incompleteness | Return partial results with confidence flag; log for KB curation |
| Prompt injection | Stripped prompt text before S-expression parsing; fixed instruction template |
| Resource exhaustion | API rate limiting; max token generation cap (256 tokens) |

## Deployment Security

| Layer | Measure |
|-------|---------|
| Network | Internal deployment behind VPN; no public internet exposure |
| API | No authentication (internal); add API key auth for external deployment |
| Freebase | ODBC connection is read-only; SPARQL endpoint is read-only |
| Docker | Non-root user in container; health check catches crashes |
| Dependencies | `requirements.txt` with pinned versions; `pip audit` for known CVEs |
