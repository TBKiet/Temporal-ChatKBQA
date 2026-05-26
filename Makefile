# T-ChatKBQA Pipeline Stage Cache
# ================================
# NOTE: DVC (dvc.yaml) is the primary pipeline tool — use 'dvc repro' for
# full content-addressable pipeline with data versioning.
# This Makefile is a lightweight alternative for quick local runs.
#
# Usage:
#   make all                    # Run full pipeline (zenodo variant)
#   make mine                   # Only re-mine facts if whitelist changed
#   make candidates             # Only re-generate if facts or templates changed
#   make verify                 # Only re-verify if candidates changed
#   make export                 # Only re-export if verified samples changed
#   make clean                  # Remove all pipeline artifacts
#   make clean-stamps           # Remove stamp files (force re-run)
#   make status                 # Show which stages are stale
#
# Configuration:
#   DATASET = zenodo | virtuoso  (default: zenodo — no Virtuoso needed)
#   MAX_PER_REL = 200            (max triples per relation in mining)
#   MAX_TOTAL = 10000            (absolute cap on mined facts)
#   MAX_PER_FAMILY = 500         (filter cap per relation family)

SHELL := /bin/bash
PROJECT_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SCRIPTS := $(PROJECT_ROOT)/scripts
STAMP_DIR := $(PROJECT_ROOT)/.pipeline

# ── Configuration ─────────────────────────────────────────────────────
DATASET ?= zenodo
ZENODO_DIR ?= $(PROJECT_ROOT)/data/zenodo/extracted
WHITELIST ?= $(PROJECT_ROOT)/configs/relation_whitelist.yaml

MAX_PER_REL ?= 200
MAX_TOTAL ?= 10000
MAX_PER_FAMILY ?= 500
MAX_PER_TYPE ?= 1000
MAX_PER_DATE ?= 200
SEED ?= 42

# ── Source files (changes trigger re-run) ─────────────────────────────
SRC_SPARQL_MINER := $(PROJECT_ROOT)/src/temporal_data/sparql_miner.py
SRC_TEMPLATE_BANK := $(PROJECT_ROOT)/src/temporal_data/template_bank.py
SRC_VERIFIER := $(PROJECT_ROOT)/src/temporal_data/verifier.py
SRC_DIST_FILTER := $(PROJECT_ROOT)/src/temporal_data/distribution_filter.py
SRC_TRAINING := $(PROJECT_ROOT)/src/temporal_data/training.py

# ── Output artifacts ──────────────────────────────────────────────────
ARTIFACT_FACTS := $(PROJECT_ROOT)/data/temporal/mined_facts_zenodo.json
ARTIFACT_CANDIDATES := $(PROJECT_ROOT)/data/temporal/candidate_samples.json
ARTIFACT_VERIFIED := $(PROJECT_ROOT)/data/temporal/verified_samples.json
ARTIFACT_FAILED := $(PROJECT_ROOT)/data/temporal/failed_samples.json
ARTIFACT_TRAIN := $(PROJECT_ROOT)/LLMs/data/TChatKBQA_Freebase_NQ_train/examples.json

# ── Stamp files ───────────────────────────────────────────────────────
STAMP_MINE := $(STAMP_DIR)/mine-facts.stamp
STAMP_CANDIDATES := $(STAMP_DIR)/generate-candidates.stamp
STAMP_VERIFY := $(STAMP_DIR)/verify-samples.stamp
STAMP_EXPORT := $(STAMP_DIR)/filter-and-export.stamp

# ── Source sets for each stage ────────────────────────────────────────
MINE_SRCS := $(WHITELIST) $(SRC_SPARQL_MINER)
CANDIDATE_SRCS := $(SRC_TEMPLATE_BANK) $(WHITELIST) $(ARTIFACT_FACTS)
VERIFY_SRCS := $(SRC_VERIFIER) $(ARTIFACT_CANDIDATES)
EXPORT_SRCS := $(SRC_DIST_FILTER) $(SRC_TRAINING) $(ARTIFACT_VERIFIED)

# ── Top-level targets ─────────────────────────────────────────────────

.PHONY: all
all: export

.PHONY: mine
mine: $(STAMP_MINE)

.PHONY: candidates
candidates: $(STAMP_CANDIDATES)

.PHONY: verify
verify: $(STAMP_VERIFY)

.PHONY: export
export: $(STAMP_EXPORT)

.PHONY: clean
clean:
	rm -rf $(ARTIFACT_FACTS) $(ARTIFACT_CANDIDATES) $(ARTIFACT_VERIFIED) \
	       $(ARTIFACT_FAILED) $(ARTIFACT_TRAIN) \
	       $(ARTIFACT_FACTS:.json=.summary.json) \
	       $(ARTIFACT_CANDIDATES:.json=.summary.json) \
	       $(ARTIFACT_VERIFIED:.json=)_summary.json \
	       $(STAMP_DIR)
	@echo "All pipeline artifacts and stamps removed."

.PHONY: clean-stamps
clean-stamps:
	rm -rf $(STAMP_DIR)
	@echo "Stamp files removed — next make will rebuild all stale stages."

.PHONY: status
status:
	@echo "=== Pipeline Stage Status ==="
	@echo ""
	@for stage in mine candidates verify export; do \
		stamp=$(STAMP_DIR)/$$stage-facts.stamp; \
		if [ "$$stage" = "candidates" ]; then stamp=$(STAMP_DIR)/generate-candidates.stamp; fi; \
		if [ "$$stage" = "verify" ]; then stamp=$(STAMP_DIR)/verify-samples.stamp; fi; \
		if [ "$$stage" = "export" ]; then stamp=$(STAMP_DIR)/filter-and-export.stamp; fi; \
		if [ -f "$$stamp" ]; then \
			age=$$(( $$(date +%s) - $$(stat -f %m "$$stamp" 2>/dev/null || stat -c %Y "$$stamp" 2>/dev/null) )); \
			printf "  %-25s  OK   (built %d min ago)\n" "$$stage" $$(( age / 60 )); \
		else \
			printf "  %-25s  STALE (needs rebuild)\n" "$$stage"; \
		fi; \
	done
	@echo ""
	@echo "Config: DATASET=$(DATASET)  SEED=$(SEED)"

# ── Stage 1: Mine Facts ──────────────────────────────────────────────

$(STAMP_MINE): $(MINE_SRCS)
	@mkdir -p $(STAMP_DIR) $(dir $(ARTIFACT_FACTS))
	@echo "[PIPELINE] Stage 1/4: Mining facts (--dataset $(DATASET))..."
ifeq ($(DATASET),zenodo)
	python $(SCRIPTS)/build_temporal_dataset.py mine-facts-zenodo \
		--data-dir $(ZENODO_DIR) \
		--max-per-rel $(MAX_PER_REL) \
		--max-total $(MAX_TOTAL) \
		--output $(ARTIFACT_FACTS) \
		--summary $(ARTIFACT_FACTS:.json=.summary.json)
else
	python $(SCRIPTS)/build_temporal_dataset.py mine-facts \
		--whitelist $(WHITELIST) \
		--limit $(MAX_PER_REL) \
		--output $(ARTIFACT_FACTS) \
		--summary $(ARTIFACT_FACTS:.json=.summary.json)
endif
	@touch $@

# ── Stage 2: Generate Candidates ──────────────────────────────────────

$(STAMP_CANDIDATES): $(STAMP_MINE) $(CANDIDATE_SRCS)
	@mkdir -p $(STAMP_DIR) $(dir $(ARTIFACT_CANDIDATES))
	@echo "[PIPELINE] Stage 2/4: Generating candidate samples..."
	python $(SCRIPTS)/build_temporal_dataset.py generate-candidates \
		--facts $(ARTIFACT_FACTS) \
		--whitelist $(WHITELIST) \
		--seed $(SEED) \
		--output $(ARTIFACT_CANDIDATES) \
		--summary $(ARTIFACT_CANDIDATES:.json=.summary.json)
	@touch $@

# ── Stage 3: Verify Samples ──────────────────────────────────────────

$(STAMP_VERIFY): $(STAMP_CANDIDATES) $(VERIFY_SRCS)
	@mkdir -p $(STAMP_DIR) $(dir $(ARTIFACT_VERIFIED))
	@echo "[PIPELINE] Stage 3/4: Verifying samples against KB..."
	python $(SCRIPTS)/build_temporal_dataset.py verify-samples \
		--input $(ARTIFACT_CANDIDATES) \
		--output-passed $(ARTIFACT_VERIFIED) \
		--output-failed $(ARTIFACT_FAILED) \
		--summary $(ARTIFACT_VERIFIED:.json=)_summary.json
	@touch $@

# ── Stage 4: Filter and Export ────────────────────────────────────────

$(STAMP_EXPORT): $(STAMP_VERIFY) $(EXPORT_SRCS)
	@mkdir -p $(STAMP_DIR) $(dir $(ARTIFACT_TRAIN))
	@echo "[PIPELINE] Stage 4/4: Filtering and exporting training data..."
	python $(SCRIPTS)/build_temporal_dataset.py filter-and-export \
		--input $(ARTIFACT_VERIFIED) \
		--max-per-family $(MAX_PER_FAMILY) \
		--max-per-type $(MAX_PER_TYPE) \
		--max-per-date-bucket $(MAX_PER_DATE) \
		--seed $(SEED) \
		--output $(ARTIFACT_TRAIN) \
		--summary $(ARTIFACT_TRAIN:.json=)/summary.json
	@touch $@

# ── Help ──────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "T-ChatKBQA Pipeline (stage-cached)"
	@echo ""
	@echo "Targets:"
	@echo "  make all         Run full pipeline (default)"
	@echo "  make mine        Mine temporal facts"
	@echo "  make candidates  Generate candidate samples"
	@echo "  make verify      Verify samples via KB execution"
	@echo "  make export      Filter and export training data"
	@echo "  make status      Show which stages are fresh/stale"
	@echo "  make clean       Remove all pipeline artifacts"
	@echo "  make clean-stamps  Remove stamps to force rebuild"
	@echo "  make help        Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=zenodo|virtuoso   (default: zenodo)"
	@echo "  ZENODO_DIR=/path/to/zenodo"
	@echo "  MAX_PER_REL=200"
	@echo "  MAX_TOTAL=10000"
	@echo "  MAX_PER_FAMILY=500"
	@echo "  SEED=42"
	@echo ""
	@echo "Examples:"
	@echo "  make all DATASET=virtuoso"
	@echo "  make mine MAX_PER_REL=500"
	@echo "  make clean && make all"
