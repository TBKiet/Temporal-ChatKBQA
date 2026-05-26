# Decision Log

## D-001

- Date: `2026-05-23`
- Decision: Keep `ChatKBQA` as the framework core.
- Reason: The goal is to extend the baseline into `T-ChatKBQA`, not replace the underlying architecture.
- Impact: All temporal work must fit the `generate-then-retrieve` design.

## D-002

- Date: `2026-05-23`
- Decision: Use `TempQuestions` as the primary benchmark.
- Reason: It is the clearest human-authored temporal benchmark aligned with the current Freebase-based system.
- Impact: Final headline metrics must be reported on `TempQuestions` test.

## D-003

- Date: `2026-05-23`
- Decision: Use synthetic temporal data to augment training.
- Reason: The project should contribute not only temporal fine-tuning, but also temporal data construction.
- Impact: A temporal dataset generation pipeline is part of the core deliverable.

## D-004

- Date: `2026-05-23`
- Decision: `CronQuestions` is related work and a future extension, not the main v1 implementation target.
- Reason: The current project is Freebase-centric and must stay scoped enough to finish cleanly.
- Impact: `CronQuestions` informs design decisions but does not become a co-primary benchmark or pipeline.

## D-005

- Date: `2026-05-23`
- Decision: Synthetic supervision must include `question + logical form + answer`.
- Reason: This matches the training objective and structure of `ChatKBQA`.
- Impact: Data generation must validate both executable logical forms and answer outputs.

## D-006

- Date: `2026-05-23`
- Decision: Heavy compute jobs will run on `Vasi.ai`.
- Reason: Freebase loading, temporal mining, model training, and large evaluation are too RAM/GPU intensive for local execution.
- Impact: Scripts, configs, and artifacts must support a local-dev / remote-execution workflow.

## D-007

- Date: `2026-05-23`
- Decision: The primary demo surface is `Streamlit`.
- Reason: It gives a presentation-friendly interface without distracting from the data/model core.
- Impact: Demo planning should optimize for a simple web app with provenance output.

## D-008

- Date: `2026-05-23`
- Decision: Benchmark + ablation are mandatory.
- Reason: The project must show measurable improvement, not just qualitative examples.
- Impact: Evaluation planning must include baseline comparison and at least one meaningful ablation.

## D-009

- Date: `2026-05-23`
- Decision: Repo-based memory in `docs/project_memory/` is the source of truth across sessions.
- Reason: It is easy for future sessions to read directly from the project without relying on external tools.
- Impact: Future progress and major decisions should be recorded here first.

## D-010

- Date: `2026-05-23`
- Decision: Treat the bundled TempQuestions raw SPARQL files as placeholder-corrupted inputs unless proven otherwise.
- Reason: The checked-in raw files use `m.placeholder` for every topic entity and do not reproduce valid `sexpr` outputs by raw parsing alone.
- Impact: Phase 0 repair may rely on aligned merged artifacts as a temporary trusted reference while parser correctness and merged corruption are audited.

## D-011

- Date: `2026-05-25`
- Decision: Audited merged TempQuestions artifacts are the preferred human-data source for Phase 0 canonical temporal exports.
- Reason: The audit layer backfills temporal metadata from origin records and marks suspicious merged cases before they flow into training artifacts.
- Impact: Canonical TempQuestions standardization should prefer `TempQuestions_{split}.audited.json` when available, and major runs should check audit summaries first.

## D-012

- Date: `2026-05-25`
- Decision: Phase 0-clean training exports must exclude `phase0_suspicious` human records by default for the first remote experiments.
- Reason: The dominant merged-data corruption family is not repaired yet, so early `Vasi.ai` runs should optimize for cleaner supervision over maximum coverage.
- Impact: Runbooks and export commands for the first `T-ChatKBQA` experiments should opt into suspicious-record exclusion explicitly, while full-data comparisons remain separate follow-up experiments.

## D-013

- Date: `2026-05-25`
- Decision: Treat collapsed TempQuestions origin SPARQL templates as the primary corruption family, not just merged-record anomalies.
- Reason: Traceback from exported training examples showed the dominant errors originate in bundled `origin` SPARQL files, which collapse 888 train questions into 41 repeated placeholder templates and then propagate through parser fallback, merged artifacts, canonical samples, and training export.
- Impact: `phase0_suspicious` auditing must consider raw origin-template collapse directly; current human-only exports are suitable for workflow validation but not trustworthy supervision until stronger filtering or a cleaner upstream source is available.

## D-014

- Date: `2026-05-25`
- Decision: Scrap the dual-pipeline (legacy `process_NQ.py` + over-engineered `src/temporal_data/` manifest→jobs→specs→callback) in favor of a clean 4-step pipeline: mine facts directly from Virtuoso → template generate → execute verify → distribution filter → export.
- Reason: The old pipeline had 7 critical problems: dual inconsistent paths, dead synthetic branch (mined_facts.json = []), stricter audit collapsing 888→16 examples, dataset namespace mismatch, format inconsistency, and no KB-backed verification. The new approach is simpler, verifiable, and aligned with project scope.
- Impact: Old modules (`miner.py`, `remote_executor.py`, `runbook.py`, `merged_audit.py`, `scripts/run_tchatkbqa_vasi.py`) are deprecated. New modules (`sparql_miner.py`, `template_bank.py`, `verifier.py`, `distribution_filter.py`) form the core pipeline.

## D-015

- Date: `2026-05-25`
- Decision: Use a relation whitelist config (`configs/relation_whitelist.yaml`) as the single source of truth for temporal fact mining and template generation.
- Reason: Heuristic temporal relation detection (keyword matching `from/to/date/year`) caused the old pipeline to assign `government.government_position_held` to questions about film, music, and death — the dominant corruption source. Explicit declaration per family prevents this.
- Impact: Only relations declared in the whitelist are mined and used for training data. Adding a new relation family requires a config change, not code change.

## D-016

- Date: `2026-05-25`
- Decision: Execution verification is a hard gate — only S-expression samples whose SPARQL execution returns matching answers enter the training set.
- Reason: Without KB-backed verification, the old pipeline exported all generated samples regardless of correctness. The verifier converts each S-expression to SPARQL via `lisp_to_sparql()`, executes against Virtuoso, and compares answers with normalized matching (MID exact, label, date, multi-answer set).
- Impact: Expected 60-80% pass rate. Failed samples are logged for debugging but excluded from training.

## D-017

- Date: `2026-05-25`
- Decision: Vast.ai replaces Vasi.ai as the remote compute environment for Freebase Virtuoso, SPARQL mining, and training.
- Reason: User preference — Vast.ai is cheaper/more familiar for renting high-RAM GPU instances.
- Impact: Scripts use `config.py` FREEBASE_SPARQL_WRAPPER_URL which points to `localhost:8890/sparql` — works on any host running Virtuoso. No provider-specific code.

## D-018

- Date: `2026-05-25`
- Decision: Use the remote `/venv/main` Python `3.12` environment for current repo tests and Freebase smoke checks, while treating the existing `/venv/chatKBQA` Python `3.8` env as legacy-only until the environments are unified.
- Reason: The current codebase uses built-in generic annotations such as `list[str]` and `dict[str, Any]`, which fail under Python `3.8`, while the validated remote test stack now passes under Python `3.12`.
- Impact: Remote verification, parser tests, and lightweight backend checks should run with `/venv/main/bin/python`; older training commands that still depend on the legacy stack must be treated separately and documented explicitly.

## D-019

- Date: `2026-05-25`
- Decision: Treat the current processed Virtuoso Freebase dump on Vast.ai as structure-only and unsuitable for natural-language synthetic generation or semantic answer verification.
- Reason: Live inspection showed `RDF_OBJ` contains only a Virtuoso version string, `RDF_IRI` contains only a small built-in RDF vocabulary set, and the loaded triples resolve only to opaque `iri_id_*_with_no_name_entry` identifiers rather than Freebase-readable IRIs, labels, or relation names.
- Impact: The current dump can support transport/connectivity checks only. It cannot yet support English question generation, label-aware answer verification, or meaningful retrieval debugging. The next remote-data step must be either a better dump with usable IRI/label mappings or an auxiliary label/name source wired into the pipeline.

## D-020

- Date: `2026-05-25`
- Decision: Use Zenodo idirlab/freebases triple files as the primary fact source instead of Virtuoso SPARQL mining. TC/during templates are disabled until real date literals become available.
- Reason: The Virtuoso dump has no labels. Zenodo provides 122M entities with labels, 4,425 relations, and 244M triples. However, date entities (CVT nodes) also have no labels, so temporal operators (TC) cannot be populated with real dates. JOIN + ARGMAX + ARGMIN are generated, TC is skipped.
- Impact: The training dataset has 2,061 examples across 3 S-expression operators (no TC). Adding dates requires either the raw Freebase RDF dump or Wikidata cross-reference.

## D-021

- Date: `2026-05-25`
- Decision: Derive question role hints from the LAST segment of the relation path (e.g., `book.author.works_written` → role="works written"), and use direction-agnostic templates to avoid subject/object confusion.
- Reason: Early templates leaked answer labels into questions. Deriving from the MIDDLE segment (old approach) gave wrong roles for many relations. The last segment reliably names the answer entity's role.
- Impact: Templates use "{role} associated with {topic_label}" patterns instead of "Who is the {role} of {topic_label}?".

## D-022

- Date: `2026-05-25`
- Decision: First trial training completed successfully — LLaMA-2-7b + LoRA, 2,061 examples, 5 epochs, 14 min on RTX 4090. Beam search inference achieved 87.6% valid S-expression rate. The model is ready for iterative improvement.
- Reason: Trial proves the pipeline works end-to-end: Zenodo facts → templates → training → inference. Key issues identified: ARGMIN over-prediction (58%), no TC operator, hallucinated relations.
- Impact: Next iteration should focus on balancing operator distribution, adding TC support, and increasing relation diversity.

## D-023

- Date: `2026-05-25`
- Decision: Document the full data→train→eval process in `docs/tchatkbqa_data_training_eval.md` as the single reference document for the project's experimental workflow.
- Reason: Provides a complete, reproducible record of data sources, pipeline architecture, training configuration, environment fixes, and evaluation results.
- Impact: Future sessions can reference this document to understand the current state and reproduce results.

## D-024

- Date: `2026-05-26`
- Decision: Answer-level F1/Hits@1 evaluation against Zenodo triples confirms that the current T-ChatKBQA trial model generates 87.6% valid S-expressions but cannot produce meaningful answer-level metrics because it hallucinates relation names outside the 22 training relations. The honest evaluation finding is the error analysis itself: 177 unique relations generated, most non-existent in Freebase/Zenodo (e.g., `film.director.jedi_master`). Zenodo-based evaluation is technically viable but requires fixing relation hallucination first.
- Reason: A full Zenodo evaluation run (4,425 relations, 122M entities, 244M triples, 47M labels) on the Vast.ai machine showed that only 5 out of 995 valid S-expressions had both relation and entity mappings in Zenodo, and only 1 returned answers from triples. The gap is not in the evaluation infrastructure but in the model's training: 22 relations are too few, and the LLM invents plausible-sounding Freebase relations instead of constraining to known ones.
- Impact: The v1 report should present S-expression validity (87.6%), operator distribution, training metrics, and hallucination analysis as the primary evaluation. Answer-level F1 is blocked until relation diversity is increased and/or a retrieval step grounds generated relations to a whitelist. The `scripts/eval_zenodo.py` evaluator is ready for v2 when these issues are addressed.

## D-025

- Date: `2026-05-26`
- Decision: Use `generate-only` vs. `generate + fuzzy relation grounding` as the primary ablation story for v1.
- Reason: The ablation produces a clean causal result. Adding even a simple grounding step increases relation grounding from `0.5%` to `88.9%`, proving that retrieval is essential. However, end-to-end answer quality still remains at `F1 = 0.0` because entity MID hallucination becomes the next blocker. This isolates the failure boundary much more clearly than a data-source ablation would.
- Impact: The report and slides should frame v1 as evidence that both halves of the ChatKBQA paradigm matter: generation provides syntactic logical forms, retrieval/grounding makes them executable. The next implementation priority after relation grounding is entity grounding.

## D-026

- Date: `2026-05-26`
- Decision: Expand the v1 ablation into a full component decomposition: generate-only, +fuzzy relation grounding, and +golden entity.
- Reason: The third setting proves that even when entities are fixed externally, end-to-end F1 still stays at `0.0`. This means the pipeline bottleneck is not only relation hallucination or entity hallucination; limited relation relevance and KB coverage also matter. The decomposition therefore identifies the four practical requirements for T-ChatKBQA: valid syntax generation, relation grounding, entity grounding, and enough KB coverage.
- Impact: The final report should present the ablation as evidence of systematic analysis rather than a simple failed benchmark. The v2 roadmap should prioritize entity linking and broader relation/fact coverage after relation grounding.

## D-027

- Date: `2026-05-26`
- Decision: Treat FastAPI and CLI as the primary inference surfaces, and Streamlit as a live-capable presentation/demo layer with explicit degraded/offline fallback.
- Reason: A fully live classroom demo requires large external assets and services: base LLaMA model, LoRA adapter, Freebase/Virtuoso, and retrieval assets. Streamlit should therefore be honest about runtime readiness instead of pretending missing assets are available.
- Impact: Report and slides should use consistent deployment wording: API/CLI/Docker are the main production-oriented surfaces, while Streamlit demonstrates the workflow and can fall back to recorded artifacts when the full stack is unavailable.
