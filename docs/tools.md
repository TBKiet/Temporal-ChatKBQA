# T-ChatKBQA Tooling

Bộ công cụ quản lý pipeline, config, evaluation và data cho dự án.

---

## 1. Pipeline — DVC (primary) + Makefile (lightweight)

### DVC (`dvc.yaml` + `params.yaml`)

Pipeline chính, content-addressable — stage chỉ chạy lại khi deps hoặc params thay đổi.

```bash
dvc repro                              # Chạy full pipeline
dvc repro --single-item mine-facts     # Chỉ 1 stage
dvc status                             # Stage nào stale?
dvc dag                                # Xem DAG
dvc push / dvc pull                    # Đẩy/kéo data với remote
dvc metrics show / dvc metrics diff    # Xem/so sánh metrics
```

Pipeline 4 stage:

```
mine-facts → generate-candidates → verify-samples → filter-and-export
```

- **mine-facts**: Dùng Zenodo offline (không cần Virtuoso). Deps: `sparql_miner.py`, `zenodo_loader.py`
- **generate-candidates**: Template bank sinh question + S-expression. Deps: `template_bank.py`, `relation_whitelist.yaml`
- **verify-samples**: Thực thi SPARQL để xác minh (cần Virtuoso). Có thể bỏ qua.
- **filter-and-export**: Dedup, cap, xuất ra format instruction-tuning.

Tham số tập trung trong `params.yaml`:

```yaml
mine.max_total: 10000
generate.seed: 42
verify.timeout: 30
filter.max_per_family: 500
```

### Makefile (nhẹ, thay thế)

Dùng cho quick local run, không cần DVC:

```bash
make all          # Full pipeline (zenodo)
make status       # Stage nào stale?
make clean        # Xóa artifacts
```

---

## 2. Config Validator (`configs/schema.py`)

Validate YAML configs trước khi launch job — bắt lỗi sớm, tránh crash trên remote GPU.

```bash
python configs/schema.py                     # Validate tất cả configs/
python configs/schema.py --config <file>     # Validate 1 file
python configs/schema.py --quiet             # Chỉ in lỗi
```

Các schema:

| Schema | Áp dụng cho | Checks chính |
|--------|-------------|--------------|
| `RelationWhitelist` | `relation_whitelist.yaml` | Dot-format relation, template-family tương thích temporal field type, không trùng tên family |
| `TrainConfig` | `train_*.yaml` | Learning rate range, warmup ratio [0,1], valid output path, dataset name, batch size |
| `InferenceConfig` | `inference.yaml` | Beam size [1,100], device (cuda/cpu/mps), valid eval split |

5 configs hiện tại đều pass validation.

---

## 3. Evaluation Harness (`scripts/run_eval.py`)

Gom tất cả eval modes vào 1 lệnh, output JSON + Markdown report.

```bash
# Offline — chỉ cần prediction file
python scripts/run_eval.py --pred_file preds.jsonl --mode validity

# Với Zenodo — grounding + answer-level
python scripts/run_eval.py --pred_file preds.jsonl --mode all \
    --zenodo_dir /path/to/zenodo/FB+CVT+REV \
    --label_file /path/to/entities_id_label.csv \
    --gold_file data/TempQuestions/generation/merged/TempQuestions_test.json

# Xuất báo cáo
python scripts/run_eval.py --pred_file preds.jsonl --mode all \
    --output metrics/answers.json --report metrics/report.md
```

Các mode:

| Mode | Cần gì | Output |
|------|--------|--------|
| `validity` | Không cần external service | Parse rate + operator distribution |
| `grounding` | Zenodo mappings | Entity/relation grounding rate (exact + fuzzy) |
| `answer` | Zenodo + Gold file | F1 / Hits@1 / Accuracy + temporal subset |
| `all` | Tự động detect | Tất cả những gì có thể chạy |

Kết quả lưu vào `metrics/` để tích hợp với `dvc metrics show/diff`.

---

## 4. Data Download (`scripts/download_zenodo.sh`)

Tải Zenodo idirlab/freebases dataset (14GB zip, 57GB extract).

```bash
bash scripts/download_zenodo.sh
bash scripts/download_zenodo.sh /path/to/custom/dir
```

---

## Vị trí các file

```
ChatKBQA/
  dvc.yaml                          # DVC pipeline definition
  params.yaml                       # Pipeline parameters
  .dvcignore                        # DVC ignore patterns
  .dvc/config                       # Remote storage config
  Makefile                          # Lightweight pipeline (Makefile)
  configs/
    schema.py                       # Config validator (Pydantic)
  scripts/
    run_eval.py                     # Evaluation harness
    download_zenodo.sh              # Zenodo dataset downloader
  metrics/                          # Eval outputs (for dvc metrics)
    .gitkeep
```
