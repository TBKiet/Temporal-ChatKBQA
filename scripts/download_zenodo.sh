#!/usr/bin/env bash
# Download and extract the Zenodo idirlab/freebases dataset.
#
# The dataset is ~14GB compressed, ~57GB extracted.
# It contains 4,425 relations, 122M entities, 244M triples, 47M labels.
#
# Requirements: ~60GB free disk space
#
# Usage:
#   bash scripts/download_zenodo.sh
#   bash scripts/download_zenodo.sh /path/to/data/dir

set -euo pipefail

DATA_DIR="${1:-data/zenodo}"
EXTRACT_DIR="${DATA_DIR}/extracted"
ZENODO_URL="https://zenodo.org/records/14823658/files/idirlab-freebases.zip"

echo "=== T-ChatKBQA: Zenodo Dataset Download ==="
echo "Data directory: ${DATA_DIR}"
echo "Extract target: ${EXTRACT_DIR}"
echo ""

mkdir -p "${DATA_DIR}" "${EXTRACT_DIR}"

ZIP_FILE="${DATA_DIR}/idirlab-freebases.zip"

if [ -f "${ZIP_FILE}" ]; then
    echo "[SKIP] ZIP already downloaded: ${ZIP_FILE}"
else
    echo "[1/2] Downloading idirlab-freebases.zip (~14GB)..."
    echo "  This may take 10-30 minutes depending on connection."
    echo "  URL: ${ZENODO_URL}"
    echo ""
    curl -L -C - -o "${ZIP_FILE}" "${ZENODO_URL}" || {
        echo "ERROR: Download failed. Try downloading manually from:"
        echo "  ${ZENODO_URL}"
        echo "  Then place the ZIP at: ${ZIP_FILE}"
        exit 1
    }
fi

echo ""
echo "[2/2] Extracting to ${EXTRACT_DIR}..."
echo "  This may take 5-15 minutes."
unzip -n "${ZIP_FILE}" -d "${EXTRACT_DIR}" || {
    echo "ERROR: Extraction failed."
    exit 1
}

echo ""
echo "=== Done ==="
echo "Zenodo dataset extracted to: ${EXTRACT_DIR}"
echo ""
echo "Expected contents:"
echo "  relation2id.txt       (~300KB)"
echo "  entity2id.txt         (~2.4GB)"
echo "  train.txt             (~4.5GB)"
echo "  entities_id_label.csv (~1.8GB)"
echo ""
echo "Next: dvc repro"
