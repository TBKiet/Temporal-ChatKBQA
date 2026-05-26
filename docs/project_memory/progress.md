# Progress Tracker

## Current Status

**Final submission packaging in progress.** First trial is complete: 2,061 examples trained on LLaMA-2-7b (LoRA), beam search inference on TempQuestions test (268 questions), and 87.6% valid S-expression diagnostic rate. Report and Beamer slides now frame v1 as submission-ready engineering evidence, not a production-quality QA model; slides have been reduced to the required 15-page deck.

## Completed

- All previous milestones
- **Pipeline v2**: Zenodo loader, fixed role direction, direction-agnostic templates
- **Dataset**: 2,061 examples (ARGMAX 717, JOIN 675, ARGMIN 669)
- **Training**: LLaMA-2-7b + LoRA, 5 epochs, 14 min, RTX 4090, loss 3.82→0.58
- **Inference**: Beam search (5 beams) on 268 TempQuestions test questions
  - 87.6% valid S-expressions (1,174/1,340)
  - ARGMIN 783, ARGMAX 268, JOIN 123, Invalid 166
- **Final report/slides cleanup**:
  - report updated with explicit data licensing/language, train/dev/test split, baseline roles, tuning notes, deployment wording, and 0.0-result framing
  - Beamer deck reduced from 24+ pages to 15 pages
  - Streamlit positioned as a live-capable presentation layer with degraded/offline fallback when full assets are unavailable
- Part B (dates) investigated — infeasible with Zenodo
- Zenodo on Vast.ai (57GB), 32 local tests

## In Progress

- Final submission polish and evidence capture
- Screenshot capture for Streamlit, FastAPI health, and CLI/API demo surfaces

## Blocked / Risks

- TC operator missing from training → model can't learn temporal constraints
- ARGMIN over-prediction (58%) due to imbalanced training distribution
- Hallucinated relations: model invents plausible Freebase relation names
- Human TempQuestions data heavily corrupted — test set may not be reliable benchmark

- No KB-backed verification — S-expressions are syntactically correct but not executed
- TC/during operators not represented in training data — model won't learn temporal constraints
- Human TempQuestions data heavily corrupted — synthetic data is the sole training source
- Vast.ai GPU environment compatibility with the ChatKBQA training stack (torch 1.13.1, LLaMA-2-7b)

## Next 3 Tasks

1. Capture final demo screenshots: Streamlit degraded/live status, Streamlit walkthrough output, FastAPI `/health`, and CLI/API query output.
2. Run final repository checklist against required `src/`, `data/`, `models/`, `configs/`, `tests/`, and dependency files.
3. If time remains, reduce remaining LaTeX layout warnings in report tables and add screenshots to slides/report if required by the instructor.

## Last Updated

- Date: `2026-05-26`
- By: `Claude Code`

### Session 2026-05-26 (Final Report and Slide Packaging)

- Goal:
  - Bring the final report and slide deck into closer alignment with `prj_requirments.pdf` and reduce obvious grading risks.
- Completed:
  - Reduced the Beamer deck to 15 pages and removed outline/rubric-style/meta slides.
  - Fixed the overlapping architecture comparison diagram on Slide 8 by resolving both the row-level overlap (using robust relative positioning `below=1.6cm of B0` for `T0`) and the label/delta annotation overlap (naming delta nodes and incorporating them in G2's background fit parameter to push the label above cleanly).
  - Executed a premium publication-grade redesign of the TikZ architecture comparison diagram, grid-aligning both pipelines vertically, adding UI pill badges for delta annotations, drawing a beautiful Emerald database card for training data, styling them with elegant dashed bounding boxes and local indigo/emerald RGB color themes, and shifting the Legend box to the bottom-right corner (`G2.south east`) to resolve a secondary overlap with the database block and establish perfect visual symmetry.
  - Reframed the 0.0 answer-level result as a diagnostic finding: v1 is submission-ready as an engineering artifact, not production-ready as a QA model.
  - Clarified that the same-dataset generate-only TempQuestions pipeline is the main baseline; ChatKBQA paper numbers are reference-only.
  - Updated report deployment wording to align FastAPI, CLI, Docker, and Streamlit degraded/live demo behavior.
  - Shortened detailed report tables and rebuilt `report_template/main.pdf`, `ChatKBQA_Beamer/main.pdf`, and `ChatKBQA_Beamer/final_slides.pdf`.
- Artifacts Produced:
  - `report_template/main.tex`
  - `report_template/main.pdf`
  - `ChatKBQA_Beamer/main.tex`
  - `ChatKBQA_Beamer/sections/01_chatkbqa.tex`
  - `ChatKBQA_Beamer/sections/02_temporal_kbqa.tex`
  - `ChatKBQA_Beamer/images/comparison_tikz.tex` (fixed overlapping architecture diagram)
  - `ChatKBQA_Beamer/main.pdf`
  - `ChatKBQA_Beamer/final_slides.pdf`
- Blockers:
  - Final screenshot evidence still needs to be captured from the user's environment.
- Decisions Made:
  - Treat Streamlit as a live-capable presentation/demo layer with explicit degraded/offline fallback; API and CLI remain the primary inference surfaces.
- Next Actions:
  - Capture final screenshots and optionally insert them into the slide deck/report if needed.
  - Run final repo/package sanity checks before submission.

### Session 2026-05-25 (Pipeline v2: Fixes + Clean Dataset + Trial Training)

### Session 2026-05-25 (TempQuestions Corruption Traceback)

- Goal:
  - Trace repeated logical-form export failures back to the first corrupted stage in the TempQuestions pipeline
- Completed:
  - Compared exported training examples against canonical temporal samples, audited merged artifacts, sexpr artifacts, and raw origin records
  - Confirmed the corruption begins in bundled `data/TempQuestions/origin/TempQuestions.train.json`
  - Quantified the collapse:
    - 888 train questions
    - 41 unique SPARQL templates
    - 779 `merged_reference` sexpr fallbacks
  - Tightened merged auditing to flag collapsed placeholder-origin templates directly
  - Rebuilt the audited train split, canonical train samples, and human-only export
  - Measured the stricter retained subset:
    - 16 kept human examples
    - still contains residual semantic mismatches on spot check
- Artifacts Produced:
  - updated `src/temporal_data/merged_audit.py`
  - updated `tests/test_temporal_merged_audit.py`
  - regenerated `data/TempQuestions/generation/merged/TempQuestions_train.audited.json`
  - regenerated `data/TempQuestions/generation/merged/TempQuestions_train.audit.summary.json`
  - regenerated `data/temporal/TempQuestions/train.samples.json`
  - regenerated `LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json`
  - regenerated `LLMs/data/TChatKBQA_Freebase_NQ_train/summary.json`
- Blockers:
  - The retained high-confidence human subset is currently too small and still not fully clean
- Decisions Made:
  - Collapsed origin-template corruption is the primary family to defend against, not only merged-record anomalies
- Next Actions:
  - Add stronger semantic mismatch checks for residual survivors
  - Decide whether to recover cleaner human supervision from another source or deprioritize human-only training

### Session 2026-05-25 (Vast.ai Freebase Bring-Up + Remote Test Validation)

- Goal:
  - Bring up a live Freebase backend on Vast.ai, validate remote Python envs, and run a small agent/backend smoke test
- Completed:
  - Downloaded and unpacked the processed Virtuoso Freebase DB on the Vast.ai machine
  - Started Virtuoso with project-aligned ports:
    - SPARQL HTTP: `8890`
    - ODBC/server: `13001`
  - Validated both transport paths against the live backend:
    - HTTP SPARQL query returned a row
    - ODBC query returned the same row
  - Found and documented the remote Python env split:
    - `/venv/chatKBQA`: Python `3.8`, incompatible with current `list[str]` / `dict[str, Any]` syntax
    - `/venv/main`: Python `3.12`, validated for current repo tests
  - Installed the minimum repo/runtime packages in `/venv/main` for remote verification:
    - `transformers`
    - `SPARQLWrapper`
    - `pyodbc`
  - Ran remote temporal tests successfully:
    - `tests.test_temporal_parser`: `19` passing
    - remaining temporal suite: `41` passing
  - Ran a lightweight agent + live Freebase smoke test and confirmed:
    - temporal question routes to temporal mode
    - non-temporal question routes to standard mode
    - stub pipeline can return a live backend answer through ODBC
- Artifacts Produced:
  - `scripts/smoke_test_remote_env.py`
  - `docs/vast_ai_remote_env_runbook.md`
- Blockers:
  - Live backend rows currently resolve to `iri_id_*_with_no_name_entry`, so retrieval correctness still needs a deeper semantic sanity check
- Decisions Made:
  - Use `/venv/main` as the validated remote test/smoke env for current code
- Next Actions:
  - Probe real retrieval helper behavior against the live Freebase dump
  - Run `mine-facts` on Vast.ai
  - Measure the first live pipeline yields before training

### Session 2026-05-25 (Freebase Dump Content Validation)

- Goal:
  - Determine whether the live Virtuoso dump is semantically usable for synthetic data generation and answer verification
- Completed:
  - Inspected the live dump contents beyond transport-level smoke tests
  - Confirmed `RDF_OBJ` effectively contains no useful label/value content beyond a Virtuoso version string
  - Confirmed `RDF_IRI` only exposes a small built-in RDF vocabulary set rather than Freebase-readable IRIs
  - Confirmed live query results resolve only to opaque `iri_id_*_with_no_name_entry` identifiers
- Artifacts Produced:
  - updated project memory and remote runbook documentation
- Blockers:
  - The current dump is not semantically usable for English synthetic question generation or label-aware answer verification
- Decisions Made:
  - Treat the current dump as structure-only for project purposes
- Next Actions:
  - Source a better dump or a separate mapping layer for human-readable Freebase names and IRIs
  - Re-run live retrieval sanity checks once that layer exists

## Session Summary

### Session 2026-05-23

- Goal:
  - Establish persistent memory for project continuity across sessions
- Completed:
  - Designed and created the project memory structure in `docs/project_memory/`
  - Seeded overview, plan, decisions, progress, and session template files
  - Recorded the current project direction and immediate next planning tasks
- Artifacts Produced:
  - `overview.md`
  - `master_plan.md`
  - `progress.md`
  - `decisions.md`
  - `session_template.md`
- Blockers:
  - None for memory setup
- Decisions Made:
  - Repo-based memory is the source of truth for future sessions
  - Progress will be maintained with manual session checkpoints
- Next Actions:
  - Specify temporal synthetic data schema
  - Specify temporal fact mining workflow
  - Specify benchmark and ablation protocol

### Session 2026-05-23 (Coding Kickoff)

- Goal:
  - Start coding the temporal dataset foundation for `T-ChatKBQA`
- Completed:
  - Added a canonical temporal sample schema under `src/temporal_data/`
  - Added a builder to standardize `TempQuestions` into the canonical schema
  - Added a relation inventory builder to support future temporal fact mining
  - Added unit tests for schema inference, validation, and relation inventory behavior
- Artifacts Produced:
  - `src/temporal_data/schema.py`
  - `src/temporal_data/builder.py`
  - `src/temporal_data/__init__.py`
  - `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_dataset.py`
- Blockers:
  - Synthetic temporal fact mining itself is not implemented yet
  - Large-scale validation still depends on remote Freebase-heavy execution on `Vasi.ai`
- Decisions Made:
  - Canonical temporal sample export will be the first shared interface between data construction, training, and evaluation
- Next Actions:
  - Refine the temporal sample schema if needed after first real exports
  - Define and implement the temporal fact mining workflow
  - Connect canonical temporal exports to the training data pipeline

### Session 2026-05-23 (Temporal Fact Mining Manifest)

- Goal:
  - Extend the dataset foundation into a first usable temporal fact mining workflow
- Completed:
  - Added temporal fact seed extraction from canonical temporal samples
  - Added a fact mining manifest builder for remote `Vasi.ai` augmentation jobs
  - Added summary generation for manifest artifacts
  - Added unit tests for `TC` and `ARGMAX` seed extraction
  - Smoke-tested the full local workflow:
    - `standardize-tempquestions`
    - `build-relation-inventory`
    - `build-fact-mining-manifest`
  - Confirmed the current TempQuestions train split produces 462 structured fact-mining seeds
- Artifacts Produced:
  - `src/temporal_data/miner.py`
  - updates to `src/temporal_data/__init__.py`
  - updates to `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_miner.py`
- Blockers:
  - Fact mining manifest is a structured precursor; actual Freebase-heavy mining and synthetic sample generation still need implementation on top of it
- Decisions Made:
  - Remote temporal mining on `Vasi.ai` will be driven by exported manifests instead of ad hoc local querying
- Next Actions:
  - Turn fact seeds into actual remote mining jobs/specs
  - Implement synthetic temporal sample generation from mined facts
  - Hook synthetic outputs back into the training data pipeline

### Session 2026-05-23 (Remote Jobs + Synthetic Sample Path)

- Goal:
  - Extend the manifest workflow into a full remote-job and synthetic-sample pipeline
- Completed:
  - Added batched remote mining job specs for `Vasi.ai`
  - Added a normalized mined-fact schema
  - Added synthetic temporal sample generation from mined facts
  - Added summaries for remote jobs and synthetic sample outputs
  - Smoke-tested the extended workflow and confirmed:
    - 462 manifest seeds from TempQuestions train
    - 28 batched remote mining jobs
    - synthetic sample generation works from normalized mined-fact inputs
- Artifacts Produced:
  - `src/temporal_data/generator.py`
  - updates to `src/temporal_data/__init__.py`
  - updates to `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_generator.py`
- Blockers:
  - The remote mining executor that actually queries Freebase and produces mined-fact records is still missing
  - Synthetic samples currently derive from mined facts; they still need a real remote fact extraction stage
- Decisions Made:
  - The augmentation pipeline will separate:
    - seed extraction
    - remote mining jobs
    - mined fact normalization
    - synthetic sample generation
- Next Actions:
  - Implement the remote mining executor spec/runner interface for `Vasi.ai`
  - Define mined-fact export format expected from remote jobs
  - Connect synthetic sample outputs into training-data conversion

### Session 2026-05-23 (Remote Mining Runner)

- Goal:
  - Add the missing execution layer between remote mining jobs and normalized mined facts
- Completed:
  - Added a remote mining runner that expands batched jobs into deterministic per-query specs
  - Added backend-agnostic query payload generation for later `Vasi.ai` execution
  - Added normalized mined-fact conversion from executor rows into canonical `MinedTemporalFact` records
  - Added support for `dry-run` and fixture-backed local execution so the workflow can be tested without live Freebase services
  - Added a new CLI subcommand:
    - `run-remote-mining`
  - Verified the new command works end-to-end with a local fixture payload
- Artifacts Produced:
  - `src/temporal_data/remote_executor.py`
  - updates to `src/temporal_data/__init__.py`
  - updates to `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_remote_executor.py`
- Blockers:
  - Real Freebase-backed execution on `Vasi.ai` still needs an injected executor callback or service wrapper
  - Synthetic mined facts still need stronger validation and filtering before large-scale augmentation
- Decisions Made:
  - The remote execution layer will stay backend-agnostic locally and accept a pluggable executor for `Vasi.ai`
  - Fixture-backed execution is the default local verification path for remote mining workflow changes
- Next Actions:
  - Connect normalized mined facts into training-data conversion
  - Add quality filters for synthetic temporal samples derived from remote mining
  - Define the benchmark and ablation protocol around `TempQuestions`

### Session 2026-05-23 (Training Export Path)

- Goal:
  - Connect canonical human and synthetic temporal samples into the LLM instruction-tuning format used by ChatKBQA training
- Completed:
  - Added a training-data exporter for canonical temporal samples
  - Added metadata-aware conversion from `TemporalSample` to `examples.json` records
  - Added support for combining human `TempQuestions` data with synthetic temporal augmentation
  - Added deduplication and optional synthetic-cap control for export
  - Added a new CLI subcommand:
    - `export-training-examples`
  - Added a dedicated training config for `TChatKBQA_Freebase_NQ_train`
  - Verified the export command locally with fixture human + synthetic inputs
- Artifacts Produced:
  - `src/temporal_data/training.py`
  - updates to `src/temporal_data/__init__.py`
  - updates to `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_training.py`
  - `configs/train_tchatkbqa.yaml`
- Blockers:
  - Synthetic temporal samples still need stronger quality filtering before large-scale export
  - The remote Vasi.ai workflow still needs real mined-fact artifacts to produce a full augmented training corpus
- Decisions Made:
  - `T-ChatKBQA` training exports will use a separate dataset namespace:
    - `TChatKBQA_Freebase_NQ_train`
  - Metadata can stay in exported examples during development for easier auditability
- Next Actions:
  - Add synthetic-sample quality filters before large-scale training export
  - Define benchmark and ablation protocol for `TempQuestions`
  - Wire the new export path into the Vasi.ai training runbook

### Session 2026-05-23 (Synthetic Quality Filtering)

- Goal:
  - Add a quality gate for synthetic temporal samples before they are exported into `T-ChatKBQA` training data
- Completed:
  - Added a synthetic-sample review layer with explicit filtering rules
  - Added accepted/rejected split logic with validation status updates
  - Added rejection-reason summaries for quick debugging and dataset audits
  - Added a new CLI subcommand:
    - `filter-synthetic-samples`
  - Added optional filtered-synthetic enforcement in training export
  - Verified the filtering command locally with one accepted and one rejected fixture sample
- Artifacts Produced:
  - `src/temporal_data/quality.py`
  - updates to `src/temporal_data/__init__.py`
  - updates to `src/temporal_data/training.py`
  - updates to `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_quality.py`
- Blockers:
  - Quality filters are currently heuristic and should be stress-tested on real mined artifacts from `Vasi.ai`
  - The project still needs a remote train/eval runbook to connect filtered data exports to actual model training
- Decisions Made:
  - Synthetic temporal samples must pass an explicit quality gate before they are trusted for large-scale augmentation
  - Training export can optionally re-filter synthetic inputs as a safety check
- Next Actions:
  - Connect filtered exports to the Vasi.ai training workflow
  - Define benchmark and ablation protocol for `TempQuestions`
  - Add a lightweight runbook or script wrapper for remote train/eval execution

### Session 2026-05-23 (Vasi.ai Runbook Wrapper)

- Goal:
  - Turn the filtered temporal data pipeline into a stage-based remote training and evaluation workflow for `Vasi.ai`
- Completed:
  - Added runbook helpers to register datasets in `LLMs/data/dataset_info.json`
  - Added command builders for `prepare`, `train`, `infer`, `eval`, and `full` workflow stages
  - Added a dedicated Vasi.ai wrapper script with dry-run and execute modes
  - Registered `TChatKBQA_Freebase_NQ_train` and `TChatKBQA_Freebase_NQ_test` in the dataset registry
  - Verified the wrapper with a full dry-run command sequence
- Artifacts Produced:
  - `src/temporal_data/runbook.py`
  - `scripts/run_tchatkbqa_vasi.py`
  - `tests/test_temporal_runbook.py`
  - updates to `LLMs/data/dataset_info.json`
- Blockers:
  - No real Vasi.ai training or evaluation artifacts have been produced yet in this local session
  - Benchmark and ablation protocol still needs to be locked before large-scale experiment execution
- Decisions Made:
  - Vasi.ai orchestration will be stage-based so we can rerun `prepare`, `train`, `infer`, or `eval` independently
  - The wrapper defaults to dry-run mode to reduce accidental heavy-job launches
- Next Actions:
  - Launch the first end-to-end Vasi.ai run with filtered temporal data
  - Define benchmark and ablation protocol for `TempQuestions`
  - Add result collection/checkpoint logging conventions for remote experiments

### Session 2026-05-23 (Improvements Documentation)

- Goal:
  - Write one consolidated document describing all implemented T-ChatKBQA improvements
- Completed:
  - Added a single markdown document summarizing the full improvement set from memory system to Vasi.ai orchestration
  - Documented the baseline-to-T-ChatKBQA transition, new modules, CLI commands, tests, and remaining gaps
- Artifacts Produced:
  - `docs/tchatkbqa_improvements.md`
- Blockers:
  - The document reflects implementation status up to this point; future benchmark results still need to be added after real Vasi.ai runs
- Decisions Made:
  - This document will act as the high-level implementation summary for future reporting, slide writing, and handoff
- Next Actions:
  - Reuse the new improvements doc when writing benchmark and ablation sections
  - Keep it updated when real training/evaluation artifacts arrive

### Session 2026-05-23 (Phase 0 Parser Repair)

- Goal:
  - Fix the broken TempQuestions preprocessing path so the repo can regenerate non-null temporal S-expressions
- Completed:
  - Confirmed the checked-in TempQuestions raw SPARQL is placeholder-corrupted and does not parse cleanly on its own
  - Updated `parse_sparql_tempquestions.py` to normalize compact `SELECT ... WHERE {` layout before parsing
  - Added a controlled repair fallback from aligned merged artifacts when raw parsing fails
  - Regenerated `data/TempQuestions/sexpr/TempQuestions.train.expr.json` and `TempQuestions.test.expr.json`
  - Restored both splits from all-null to fully non-null outputs
  - Recorded parser source breakdown:
    - train: `109` direct parser, `779` merged-reference repair
    - test: `47` direct parser, `336` merged-reference repair
  - Updated `process_NQ.py` so TempQuestions exports structured `sexpr` by default instead of bracket-linearized output
  - Tightened parser tests so they no longer swallow assertion failures
- Artifacts Produced:
  - updates to `parse_sparql_tempquestions.py`
  - updates to `process_NQ.py`
  - updates to `tests/test_temporal_parser.py`
- Blockers:
  - Many merged TempQuestions `sexpr` entries still appear semantically suspicious and need targeted auditing
  - The current repair path restores reproducibility, but not full confidence in raw-source correctness
- Decisions Made:
  - Phase 0 repair is mandatory before trusting downstream model experiments
  - Structured `sexpr` is now the default TempQuestions training target to remove train/inference mismatch
- Next Actions:
  - Audit suspicious merged `sexpr` cases and repair high-impact corruption patterns
  - Define benchmark and ablation protocol for `TempQuestions`
  - Launch the first Vasi.ai run only after Phase 0 sanity checks are acceptable

### Session 2026-05-23 (Merged TempQuestions Audit)

- Goal:
  - Add a structured audit layer for merged TempQuestions records and quantify how much training data still looks suspicious after parser recovery
- Completed:
  - Added a merged-data audit module that backfills temporal metadata from aligned origin records
  - Added heuristic suspicious flags for obvious corruption patterns, especially generic government relations in non-government questions
  - Added a new CLI subcommand:
    - `audit-merged-tempquestions`
  - Generated audited merged artifacts for both train and test:
    - `TempQuestions_train.audited.json`
    - `TempQuestions_test.audited.json`
  - Generated audit summaries for both splits
  - Updated `process_NQ.py` so it can exclude `phase0_suspicious` TempQuestions records during export
  - Measured current suspicious-data footprint:
    - train: `291 / 888`
    - test: `115 / 383`
  - Rebuilt TempQuestions training export with suspicious records removed:
    - train examples reduced to `597`
    - outputs remain structured `sexpr` rather than bracket-linearized forms
- Artifacts Produced:
  - `src/temporal_data/merged_audit.py`
  - updates to `src/temporal_data/__init__.py`
  - updates to `scripts/build_temporal_dataset.py`
  - updates to `process_NQ.py`
  - `tests/test_temporal_merged_audit.py`
  - `data/TempQuestions/generation/merged/TempQuestions_train.audited.json`
  - `data/TempQuestions/generation/merged/TempQuestions_test.audited.json`
  - corresponding audit summary JSON files
- Blockers:
  - Suspicious flags are heuristic and do not yet repair the underlying semantics of bad records
  - We still need targeted fixes for the most frequent corruption family rather than only excluding it
- Decisions Made:
  - Phase 0 can proceed with a cleaner subset by excluding `phase0_suspicious` records during TempQuestions export
  - Merged-data audit summaries are now a required sanity check before major training runs
- Next Actions:
  - Repair the dominant corruption family behind generic government-relation failures
  - Define benchmark and ablation protocol for `TempQuestions`
  - Launch the first Vasi.ai run with the Phase 0-clean training subset

### Session 2026-05-23 (Phase 0-Clean Canonical Export)

- Goal:
  - Make sure Phase 0 suspicious-record filtering propagates through the newer canonical temporal pipeline, not only the legacy `process_NQ.py` path
- Completed:
  - Updated canonical TempQuestions standardization to automatically prefer audited merged artifacts when available
  - Propagated `phase0_suspicious` metadata into canonical `TemporalSample` records
  - Added support in temporal training export to drop suspicious human samples with:
    - `--exclude-suspicious-human`
  - Verified the canonical human sample export path end-to-end:
    - canonical train samples: `888`
    - suspicious canonical human samples: `291`
    - clean exported human training examples: `597`
- Artifacts Produced:
  - updates to `src/temporal_data/builder.py`
  - updates to `src/temporal_data/training.py`
  - updates to `scripts/build_temporal_dataset.py`
  - updates to `tests/test_temporal_dataset.py`
  - updates to `tests/test_temporal_training.py`
- Blockers:
  - This still isolates suspicious records rather than repairing their underlying semantics
  - A true full-data temporal benchmark still depends on fixing the dominant corruption family instead of filtering it out
- Decisions Made:
  - Audited merged TempQuestions artifacts are now the preferred source for canonical human sample generation
  - Phase 0-clean human exports should exclude suspicious records by default for early experiments
- Next Actions:
  - Repair the dominant corruption family behind generic government-relation failures
  - Define benchmark and ablation protocol for `TempQuestions`
  - Launch the first Vasi.ai run with the Phase 0-clean canonical training subset

### Session 2026-05-25 (State Sync + Phase 0-Clean Readiness Review)

- Goal:
  - Sync project memory with the actual implementation state and identify concrete gaps before the first real Phase 0-clean `Vasi.ai` run
- Completed:
  - Reviewed `project_memory`, `docs/tchatkbqa_improvements.md`, temporal pipeline code, and the current `Vasi.ai` wrapper behavior
  - Confirmed the local temporal pipeline remains green with targeted unit coverage:
    - `40` temporal tests passing
  - Confirmed audited merged TempQuestions artifacts are present for both train and test splits
  - Confirmed the current `Vasi.ai` wrapper dry-run prepare command still omits:
    - `--exclude-suspicious-human`
  - Confirmed the canonical artifact inputs expected by the new pipeline are still missing locally:
    - `data/temporal/TempQuestions/train.samples.json`
    - `data/temporal/TempQuestions/test.samples.json`
    - `data/temporal/synthetic.filtered.samples.json`
    - `LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json`
  - Updated `project_memory` top-level status and decisions to reflect the implemented Phase 0 audit/clean-export workflow
- Artifacts Produced:
  - updates to `docs/project_memory/overview.md`
  - updates to `docs/project_memory/progress.md`
  - updates to `docs/project_memory/decisions.md`
- Blockers:
  - The first real remote run still needs generated canonical artifacts, not just code paths
  - The runbook currently defaults to a non-Phase-0-clean export command
  - Real remote mining is still backend-incomplete without a concrete Freebase executor on `Vasi.ai`
- Decisions Made:
  - Audited merged TempQuestions artifacts are the preferred Phase 0 source for canonical human exports
  - The first remote experiments should exclude suspicious human records by default
- Next Actions:
  - Wire Phase 0-clean defaults into the `Vasi.ai` prepare path
  - Materialize canonical human and optional synthetic temporal artifacts under `data/temporal/`
  - Launch the first real `prepare/train/infer/eval` run on `Vasi.ai`

### Session 2026-05-25 (Phase 0-Clean Wrapper Default)

- Goal:
  - Make the `Vasi.ai` wrapper default to the Phase 0-clean human training subset instead of requiring a manual export flag
- Completed:
  - Updated runbook command generation so `prepare` includes:
    - `--exclude-suspicious-human`
    - by default
  - Added a wrapper escape hatch:
    - `--allow-suspicious-human`
    - for later full-data or ablation runs
  - Added tests covering both the new default and the override path
  - Verified the wrapper dry-run now prints a Phase 0-clean export command by default
- Artifacts Produced:
  - updates to `src/temporal_data/runbook.py`
  - updates to `scripts/run_tchatkbqa_vasi.py`
  - updates to `tests/test_temporal_runbook.py`
- Blockers:
  - Canonical temporal artifact files still need to be materialized before a real remote run
  - Remote mining still needs a real executor backend on `Vasi.ai` if synthetic augmentation is part of the first run
- Decisions Made:
  - The first remote wrapper path now optimizes for the clean audited human subset by default
  - Full-data comparisons should opt in explicitly via `--allow-suspicious-human`
- Next Actions:
  - Generate canonical `data/temporal/TempQuestions/*.samples.json` artifacts
  - Export `LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json`
  - Launch the first Phase 0-clean `Vasi.ai` run

### Session 2026-05-25 (Pipeline Rewrite — SPARQL Mining + Template + Verify + Filter)

- Goal:
  - Replace the broken dual-pipeline approach with a clean 4-step pipeline: mine facts directly from Virtuoso → template generate → execute verify → distribution filter → export
- Completed:
  - Audited the existing code and identified 7 root problems (dual pipelines, dead synthetic path, namespace mismatch, etc.)
  - Designed new pipeline with user incorporating feedback:
    - Relation whitelist config instead of heuristic guessing
    - Execution verifier with answer normalization (MID, label, date, multi-answer)
    - Distribution filter after verify, not before
    - Enriched fact schema with provenance fields
  - Created 5 new files:
    - `configs/relation_whitelist.yaml` (11 families, ~80% from real TempQuestions patterns)
    - `src/temporal_data/sparql_miner.py` (SPARQL query builders, raw multi-var mining, MinedTemporalFact)
    - `src/temporal_data/template_bank.py` (32 templates, 6 families, S-expression builders)
    - `src/temporal_data/verifier.py` (lisp_to_sparql + execute + normalized compare)
    - `src/temporal_data/distribution_filter.py` (dedup, cap by family/type/date)
  - Updated `src/temporal_data/__init__.py` — clean imports, no backward compat
  - Rewrote `scripts/build_temporal_dataset.py` — 4 new subcommands, kept legacy as deprecated
  - Removed `scripts/run_tchatkbqa_vasi.py`
  - Created `tests/test_temporal_pipeline.py` — 32 tests, all passing, offline
  - All imports verified working locally (lazy Virtuoso deps)
- Artifacts Produced:
  - `configs/relation_whitelist.yaml`
  - `src/temporal_data/sparql_miner.py`
  - `src/temporal_data/template_bank.py`
  - `src/temporal_data/verifier.py`
  - `src/temporal_data/distribution_filter.py`
  - updated `src/temporal_data/__init__.py`
  - rewritten `scripts/build_temporal_dataset.py`
  - `tests/test_temporal_pipeline.py`
- Blockers:
  - Need Vast.ai instance with Virtuoso to test real SPARQL mining and verification
  - Freebase CVT schema may need query adjustments after first real run
- Decisions Made:
  - Vast.ai replaces Vasi.ai as remote compute environment (user preference)
  - Relation whitelist is the single source of truth for which facts to mine and templates to apply
  - Execution verification is a hard gate — only KB-verified samples enter training
  - Old modules (miner, remote_executor, runbook, merged_audit) kept on disk but removed from package imports
- Next Actions:
  - Rent Vast.ai instance and deploy Virtuoso
  - Run `mine-facts` with the 11-family whitelist
  - Run full pipeline end-to-end and measure yields

### Session 2026-05-25 (Phase 0-Clean Artifact Prep)

- Goal:
  - Materialize the remaining local artifacts and write an exact operator-facing runbook for the first `Vasi.ai` execution
- Completed:
  - Generated canonical TempQuestions temporal samples:
    - `train`: `888`
    - `test`: `383`
  - Generated temporal relation inventory:
    - `1609` relations
  - Generated temporal fact-mining manifest:
    - `687` seeds
  - Generated batched remote mining jobs:
    - `26` jobs
  - Expanded remote mining jobs into deterministic query specs via dry-run:
    - `687` query specs
  - Exported the Phase 0-clean human training dataset:
    - `597` examples
  - Exported the Phase 0-clean canonical test dataset namespace:
    - `268` examples
  - Added a preflight checker for `Vasi.ai` readiness and verified it reports:
    - `READY`
  - Added a dedicated markdown runbook with exact preflight, prepare, train, infer, and eval commands for `Vasi.ai`
- Artifacts Produced:
  - `data/temporal/TempQuestions/train.samples.json`
  - `data/temporal/TempQuestions/train.summary.json`
  - `data/temporal/TempQuestions/test.samples.json`
  - `data/temporal/TempQuestions/test.summary.json`
  - `data/temporal/relation_inventory.json`
  - `data/temporal/relation_inventory.summary.json`
  - `data/temporal/fact_mining_manifest.json`
  - `data/temporal/fact_mining_manifest.summary.json`
  - `data/temporal/remote_mining_jobs.json`
  - `data/temporal/remote_mining_jobs.summary.json`
  - `data/temporal/remote_query_specs.json`
  - `data/temporal/mined_facts.json`
  - `data/temporal/mined_facts.summary.json`
  - `LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json`
  - `LLMs/data/TChatKBQA_Freebase_NQ_train/summary.json`
  - `LLMs/data/TChatKBQA_Freebase_NQ_test/examples.json`
  - `LLMs/data/TChatKBQA_Freebase_NQ_test/summary.json`
  - `scripts/preflight_tchatkbqa_vasi.py`
  - `docs/vasi_phase0_clean_runbook.md`
- Blockers:
  - Synthetic augmentation still stops short of real mined facts because the remote mining executor is not wired to a live backend yet
  - The first train/infer/eval artifacts still need to be produced on `Vasi.ai`
- Decisions Made:
  - The first remote runbook will target the human-only Phase 0-clean subset by default
  - Mining manifests and remote job specs can be prepared locally even before the live executor path exists
- Next Actions:
  - Sync the prepared artifacts to `Vasi.ai`
  - Execute the first human-only Phase 0-clean remote run
  - Record metrics and checkpoint paths back into project memory

### Session 2026-05-26 (Zenodo Answer-Level Evaluation on Vast.ai)

- Goal:
  - Run answer-level evaluation of the T-ChatKBQA trial model against TempQuestions using Zenodo triple files as ground-truth KB, bypassing the opaque-iri Virtuoso dump
- Completed:
  - Provisioned new Vast.ai instance (RTX 4090 24GB, 2TB RAM, 600GB disk)
  - Set up Virtuoso Freebase from the processed dump — confirmed SPARQL HTTP responds, but all IRIs are opaque `iri_id_*_with_no_name_entry` (re-confirmed D-019)
  - Patched `config.py` to make torch/transformers imports optional for evaluation-only path
  - Downloaded and extracted Zenodo `idirlab-freebases` dataset (14GB zip, 57GB extracted) containing 4,425 relations, 122M entities, 244M triples, 47M labels
  - Built `scripts/eval_zenodo.py` — a custom evaluator that:
    - Parses S-expressions using ChatKBQA's own `parse_s_expr` / `extract_entities` / `extract_relations`
    - Converts ChatKBQA dot-format relations (`music.artist.album`) to Zenodo slash-format (`/music/artist/album`)
    - Converts MIDs (`m.xxxxx`) to Zenodo format (`/m/xxxxx`)
    - Streams 244M Zenodo triples to find answer MIDs for each (entity, relation) query pair
    - Converts answer MIDs to human-readable labels via `entities_id_label.csv` (47M labels)
    - Compares predicted labels with TempQuestions gold answer strings
  - Ran evaluation on 268 TempQuestions test questions x 5 beams
- Evaluation Results:
  - **S-expression validity**: 995/1,340 (74.3%) via ChatKBQA parser; 87.6% via broader lisp_to_sparql criterion
  - **With Zenodo mappings**: 5 unique queries (out of 995 valid) had both relation AND entity found in Zenodo
  - **Answer patterns found in triples**: 1 question (out of 5 mapped)
  - **F1 / Hits@1 / Accuracy**: 0.0 (the 1 question with answers did not match gold)
  - **Root cause**: Model hallucinates relation names — 177 unique relations in predictions, vast majority do not exist in Zenodo's 4,425 relations
  - Example hallucinations: `film.director.jedi_master`, `film.performance.who_is_film_performance_jennifer_lawrence_boyfriend`, `foreign.exchange.rate.history.currency`
  - **Operator distribution**: ARGMIN 58.4%, ARGMAX 20.0%, JOIN 9.2%, Invalid 12.4%
  - **Training metrics**: Loss 3.82 -> 0.58, 5 epochs, 14 min on RTX 4090
- Artifacts Produced:
  - `scripts/eval_zenodo.py` — reusable Zenodo-based evaluator
  - `models/LLaMA2-7b-tchatkbqa-trial/evaluation_beam/zenodo_eval_results.json` — per-question results
  - Patched `config.py` (lazy torch/transformers imports)
- Blockers:
  - Model trained on only 22 Zenodo relations — insufficient coverage for TempQuestions benchmark
  - No retrieval/grounding step during inference — the LLM generates raw relation names rather than selecting from a whitelist
  - TC operator absent from training data (no date literals in Zenodo facts)
- Decisions Made:
  - Zenodo-based evaluation is viable for future iterations IF relation hallucination is addressed
  - Current model is useful for S-expression generation quality analysis but not for answer-level F1
  - The primary contribution for v1 is the temporal data pipeline + training workflow, with honest error analysis as the evaluation finding
- Next Actions:
  - Report these findings honestly: 87.6% validity, operator distribution, hallucination analysis
  - For v2: increase relation diversity to 50-100, add retrieval step, source date literals for TC

### Session 2026-05-26 (Grounding Ablation Follow-up)

- Goal:
  - Test whether adding a lightweight retrieval-style grounding step improves executable predictions after generate-only inference
- Completed:
  - Ran an ablation comparing generate-only inference against a second pass with fuzzy relation grounding
  - Verified that relation grounding is not the final blocker once retrieval is added; entity MID hallucination becomes the dominant remaining issue
- Evaluation Results:
  - **Trial 1: Generate-only**
    - Relations grounded: `0.5%`
    - Entities grounded: `0%`
    - Questions with answers: `0.4%` (`1` question)
    - F1: `0.0`
  - **Trial 2: + Fuzzy relation grounding**
    - Relations grounded: `88.9%` (`885/995`)
    - Entities grounded: `5.8%` (`58/995`)
    - Questions with answers: `0.7%` (`2` questions)
    - F1: `0.0`
  - **Main insight**: Relation grounding improves by about 178x, but end-to-end answer quality is still blocked by hallucinated entity MIDs
  - Example failure 1: ``where does avril lavigne live now?'' grounded to a music-track relation instead of a residence relation
  - Example failure 2: ``first harry potter novel?'' matched the correct relation family, but the entity MID pointed to the wrong author
- Artifacts Produced:
  - report/slides updated to include the grounding ablation narrative
- Blockers:
  - Entity grounding is now the dominant failure mode after relation grounding is added
- Decisions Made:
  - The v1 ablation story is now generate-only vs. retrieval-assisted relation grounding, not human-only vs. human+synthetic training
- Next Actions:
  - Add entity grounding or entity whitelist constraints after relation grounding
  - Re-run the same evaluator to measure whether answerable coverage increases beyond the current 0.7%

### Session 2026-05-26 (Golden-Entity Ablation Completion)

- Goal:
  - Complete the component ablation by isolating entity grounding from relation grounding
- Completed:
  - Added a third ablation setting using golden entities together with fuzzy relation grounding
  - Confirmed that fixing entities improves answerable coverage, but does not recover F1 because relation relevance and KB coverage still limit execution quality
- Evaluation Results:
  - **Generate-only**
    - Entity: model (`0%`)
    - Relation: model (`0.5%`)
    - Answered: `0.4%`
    - F1: `0.0`
  - **+ Fuzzy relation grounding**
    - Entity: model (`5.8%`)
    - Relation: fuzzy (`88.9%`)
    - Answered: `0.7%`
    - F1: `0.0`
  - **+ Golden entity**
    - Entity: gold (`100%`)
    - Relation: fuzzy (`60.2%`)
    - Answered: `2.6%`
    - F1: `0.0`
  - **Main insight**: The pipeline needs all major components to work together: syntax generation, relation grounding, entity grounding, and enough KB coverage. Once relations and entities are improved, KB/relation coverage becomes the next limiting factor.
- Artifacts Produced:
  - report/slides updated with the full 3-stage ablation table
- Blockers:
  - The 22-relation training slice is still too narrow for broad TempQuestions coverage
  - Zenodo coverage remains incomplete for some relevant domains and facts
- Decisions Made:
  - The final v1 ablation story is now a full component decomposition rather than a single retrieval toggle
- Next Actions:
  - Add real entity linking or constrained entity grounding
  - Expand relation coverage beyond 22 relations
  - Re-run the same evaluator after both grounding stages are improved

### Session 2026-05-26 (Submission Readiness Cleanup)

- Goal:
  - Make the repo and submission documents internally consistent before final handoff
- Completed:
  - Fixed the offline temporal test suite so the current Zenodo-backed template-bank scope matches the assertions
  - Added the missing `docs/project_plan.md` linked from `README.md`
  - Updated `README.md`, `docs/data_description.md`, `docs/model_evaluation.md`, and `docs/report.md` to reflect the actual v1 setup:
    - synthetic Zenodo training data is the main supervision source
    - TempQuestions is the held-out benchmark
    - the submit baseline is the generate-only configuration
    - answer-level metrics remain blocked at `0.0 / 0.0 / 0.0`
  - Regenerated `docs/report.pdf` from the updated Markdown source
- Artifacts Produced:
  - updated submission-facing docs
  - synced `docs/report.pdf`
- Blockers:
  - The report PDF build emits a font warning for the `≥` glyph under the local LaTeX font setup, but the PDF is generated successfully
  - The project still lacks a strong answer-level TempQuestions result; this is a model limitation, not a repo inconsistency
- Decisions Made:
  - Submission materials should prioritize honest benchmark framing over optimistic claims
- Next Actions:
  - Final proofreading of report/slides
  - If time permits, export the PDF with a richer Unicode font to remove the warning
  - Continue v2 work on relation diversity, relation grounding, and entity grounding
