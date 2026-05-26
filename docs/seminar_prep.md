# Project Final Seminar — Preparation Checklist

## What to Bring
- [ ] Laptop with slides (`slides.pptx`)
- [ ] Written report PDF (`docs/report.pdf`) — printed or digital
- [ ] Source code repo open in browser (GitHub)
- [ ] Backup copy on USB drive
- [ ] Demo ready (CLI demo mode works without GPU/Freebase)

## How Each Requirement Maps to Your Slides

| Req | Requirement | Slide(s) | Talking Points |
|-----|------------|----------|----------------|
| 1 | Project Objective | 1, 3 | "End-to-end NLP system: question → answer over KB with temporal constraints. Not just research — designed for production." |
| 2 | Business Problem | 2 | "Organizations can't query time-sensitive KB facts. 90% faster query formulation. Target: analysts, BI teams, legal." |
| 3 | Dev Infrastructure | 4, 9 | "Python, Git, modular: src/ data/ configs/ tests/. requirements.txt, Dockerfile. Clear separation of concerns." |
| 4 | Data Management | 6 | "TempQuestions: 1,271 questions, 70/30 split. SPARQL→S-expr→instruction format. Known biases: Western entities, pre-2016." |
| 5 | Model & Optimization | 7, 8 | "LLaMA-2-7B + LoRA (4.2M params). 50 epochs, beam=15. Baseline comparison. Error analysis: 30% wrong skeleton, 20% linking." |
| 6 | Deployment | 9 | "REST API + CLI + Docker. Lazy loading. Config-driven. 3-5s inference per question. Provenance in every response." |
| 7 | Agentic AI | 5 | "4-step: Detect → Route → Execute → Refine. Regex signals, iterative retry, full provenance. Autonomous decision on empty results." |
| 8 | Continual Learning | 11 | "Feedback loop, quarterly eval, drift detection, Wikidata migration path. Monitoring: answer rate, latency, error rate." |
| 9 | Privacy & Robustness | 10 | "PII-free data, read-only KB, ephemeral queries. Adversarial: prompt injection blocked by SPARQL validation. KB staleness risk." |
| 10 | Project Management | ⚠️ **NOT IN SLIDES** | See below — you need to add this. |
| 11 | Ethics & Responsible AI | 10 | "Explainability via provenance. Bias: Western entities, temporal skew. Human-in-the-loop. Open-source." |

---

## ⚠️ Critical Gap: Requirement 10 (Project Management)

Your current 12 slides do **not** include a dedicated Project Management slide. Your teacher will ask about:
- Timeline (12 weeks)
- Task breakdown (who did what — simulated)
- Team scaling reflection

**What to do**: When you present Slide 12 (Key Takeaways), add these talking points verbally, OR create an extra slide. Here's the content:

> **Project Timeline**: 12 weeks — Research (2w) → Data Pipeline (2w) → LLM Training (2w) → Temporal Extension (2w) → Evaluation (1w) → Deployment (1w) → Testing (1w) → Documentation (1w)
>
> **Task Breakdown** (7 simulated roles): Research Lead, ML Engineer, Software Engineer, Backend Engineer, DevOps, QA, PM
>
> **Team Scaling**: In a 4-5 person team: parallelize data + training tracks, dedicated QA, daily standups, bus factor ≥2 on critical modules.

Full details are in `docs/project_plan.md`.

---

## Live Demo Script (2-3 minutes)

### Option A: CLI Demo (no GPU/Freebase required)
```bash
# Show the agent routing logic — detects temporal signals without LLM
python -m src.cli --demo
```
This prints 6 example questions with signal detection and routing decisions.
**Say**: "This shows the agent detecting temporal signals and deciding which pipeline to use. No GPU or Freebase needed — it runs entirely on regex logic."

### Option B: If Freebase is running
```bash
python -m src.cli --question "Who was US president before Obama?"
```
**Say**: "The agent detects 'before' as a temporal signal, routes to the temporal pipeline, generates an S-expression with temporal operators, converts to SPARQL, executes against Freebase, and returns the answer with full provenance."

### Option C: If API is running
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who was US president before Obama?"}'
```
**Say**: "The REST API returns JSON with the answer, the SPARQL used, the temporal constraint, and all reasoning steps — full audit trail."

---

## Q&A Preparation — Likely Questions

| Question | Answer |
|----------|--------|
| "Why not just use ChatGPT?" | "ChatGPT doesn't have access to Freebase's structured temporal data. It hallucinates facts. Our system executes real SPARQL against a real KB — answers are verifiable and come with provenance." |
| "What happens with a non-temporal question?" | "The agent auto-detects no temporal signals and routes to the standard ChatKBQA pipeline. No degradation for non-temporal queries." |
| "How do you handle wrong answers?" | "The agent retries with relaxed constraints up to 2 times. If still empty, returns empty with provenance showing what was tried." |
| "Why Freebase and not Wikidata?" | "Freebase was used by ChatKBQA and TempQuestions. The architecture is KB-agnostic — only the SPARQL converter needs schema adaptation for Wikidata. Migration path is in the continual learning plan." |
| "Is the LLM fine-tuned? How much data?" | "Yes — LoRA fine-tuned on ~890 TempQuestions training examples. LoRA trains only 4.2M params (0.06% of 7B), so 16MB checkpoint." |
| "What if Freebase is down?" | "Docker health check detects it. The system returns a clear error, not a crash. Could add a read replica for HA." |
| "Is there bias in the system?" | "Yes — acknowledged in the ethics statement. Western entity coverage bias, temporal skew toward 1900-2016, gender bias in question entities. Separate metrics and confidence flags proposed." |
| "How would this work in a real company?" | "Deploy behind VPN with API key auth. Internal analysts query via REST or CLI. Feedback loop captures corrections. Quarterly retrains. KB migration to Wikidata for freshness." |

---

## Before the Seminar

1. **Read the report** (`docs/report.pdf`) — be able to summarize each section
2. **Practice the slides** — target 15-20 minutes for 12 slides
3. **Test the demo** — run `python -m src.cli --demo` to make sure it works
4. **Check the repo** — make sure it's pushed and the README renders correctly
5. **Review `docs/project_plan.md`** — have the timeline and task breakdown memorized

## During the Seminar

1. **Start with the business problem** (Slide 2) — hook them with real examples
2. **Show the architecture** (Slide 4) — the pipeline flow is your strongest visual
3. **Emphasize the agentic AI** (Slide 5) — this is a key requirement, make it clear
4. **Live demo** after Slide 9 (Deployment) — show the CLI demo as proof it works
5. **Address ethics head-on** (Slide 10) — don't skip the limitations
6. **Verbalize Requirement 10** (Project Management) during Slide 12
7. **End with the "Thank You" slide** — leave time for questions
