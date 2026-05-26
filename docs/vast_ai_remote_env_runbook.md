# Vast.ai Remote Env Runbook

This document records the validated remote environment for running `T-ChatKBQA`
with a local Freebase Virtuoso service on a `Vast.ai` machine.

Use this runbook when you want to:

- verify that Freebase is online
- run repo tests on the remote machine
- rerun the lightweight routing + backend smoke test

## Validated Layout

- Repo path: `/workspace/ChatKBQA`
- Freebase path: `/workspace/freebase/virtuoso_db`
- Freebase archive: `/workspace/freebase/virtuoso_db.zip`
- Virtuoso config: `/workspace/freebase/virtuoso_db/virtuoso.ini`
- Virtuoso log: `/workspace/freebase/virtuoso_db/virtuoso.log`
- ODBC driver symlink expected by repo:
  - `/workspace/ChatKBQA/lib/virtodbc.so`

## Validated Ports

- SPARQL HTTP: `http://127.0.0.1:8890/sparql`
- Virtuoso ODBC/server port: `127.0.0.1:13001`

These ports match the current repo defaults in `config.py`.

## Python Environments

There are two Python environments on the current remote machine:

- Recommended for current repo code and tests: `/venv/main/bin/python`
- Legacy training-oriented env on this machine: `/venv/chatKBQA/bin/python`

Important:

- `/venv/chatKBQA` is Python `3.8` and cannot import modules that use
  annotations like `list[str]` and `dict[str, Any]`
- `/venv/main` is Python `3.12` and is the validated env for:
  - `tests.test_temporal_parser`
  - the temporal unit test suite
  - `executor/sparql_executor.py`
  - the remote smoke test below

## Required Packages In `/venv/main`

The following were installed and validated for repo imports and Freebase access:

```bash
/venv/main/bin/pip install --no-cache-dir \
  "transformers>=4.46,<5" \
  SPARQLWrapper \
  pyodbc
```

## Freebase Preflight

Verify the service is online:

```bash
ss -ltnp | grep -E ':8890|:13001'
tail -n 50 /workspace/freebase/virtuoso_db/virtuoso.log
```

Expected lines include:

- `HTTP server online at 8890`
- `Server online at 13001`

## Freebase Restart

Stop the current service:

```bash
pkill -f '/usr/bin/virtuoso-t +configfile /workspace/freebase/virtuoso_db/virtuoso.ini' || true
```

Start it again:

```bash
nohup /usr/bin/virtuoso-t \
  +configfile /workspace/freebase/virtuoso_db/virtuoso.ini \
  +wait >/workspace/freebase/virtuoso-start.log 2>&1 &
```

Notes:

- First startup after unpacking the large DB can take more than a minute
- ODBC checks can fail briefly while Virtuoso is still recovering the DB

## Lightweight Smoke Test

Run the validated smoke test from the repo root:

```bash
cd /workspace/ChatKBQA
/venv/main/bin/python scripts/smoke_test_remote_env.py
```

What it checks:

- HTTP SPARQL returns a row
- ODBC returns the same row
- `TemporalQuestionAgent` routes a temporal question and a factual question
- the stub pipeline can return a live answer from the Freebase backend

## Temporal Unit Tests

Run the routing/parser tests:

```bash
cd /workspace/ChatKBQA
/venv/main/bin/python -m unittest tests.test_temporal_parser -v
```

Run the main temporal suite:

```bash
cd /workspace/ChatKBQA
/venv/main/bin/python -m unittest \
  tests.test_temporal_dataset \
  tests.test_temporal_generator \
  tests.test_temporal_merged_audit \
  tests.test_temporal_miner \
  tests.test_temporal_quality \
  tests.test_temporal_remote_executor \
  tests.test_temporal_runbook \
  tests.test_temporal_training \
  -v
```

## Known Caveat

The current Virtuoso dump is queryable and returns consistent rows through both
HTTP and ODBC, but many sample rows resolve to strings like
`iri_id_21_with_no_name_entry` instead of clean Freebase MIDs or labels.

That means:

- the backend service is live
- the repo can connect to it
- but a deeper retrieval sanity check is still needed before trusting full
  benchmark or training runs

## Current Blocker

Deeper inspection of the loaded dump showed a stronger limitation:

- `RDF_OBJ` effectively contains no useful string/value table beyond a Virtuoso
  version string
- `RDF_IRI` exposes only a small built-in RDF vocabulary set
- the graph itself is loaded as opaque internal IRI IDs

Practical consequence:

- this dump is usable for low-level graph connectivity checks only
- it is not currently usable for:
  - English question generation from entities/relations
  - label-aware answer verification
  - meaningful Freebase relation/entity debugging

Do not treat the current dump as ready for the full synthetic-data pipeline
until a human-readable Freebase IRI/name source is available.
