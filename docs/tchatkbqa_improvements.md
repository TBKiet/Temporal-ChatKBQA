# T-ChatKBQA Improvements

## Overview

Tài liệu này tổng hợp toàn bộ các cải tiến đã được thêm vào project để nâng từ baseline `ChatKBQA` lên `T-ChatKBQA`.

Mục tiêu của nhánh cải tiến này là:

- bổ sung năng lực temporal KBQA trên nền `ChatKBQA`
- xây thêm temporal data pipeline thay vì chỉ dùng dataset có sẵn
- chuẩn bị đầy đủ đường đi từ data construction đến training, inference, evaluation trên `Vasi.ai`
- giữ local machine cho phát triển code, test nhẹ, và demo

## Baseline vs T-ChatKBQA

### Baseline

Baseline của project là `ChatKBQA`, theo hướng:

- generate logical form bằng LLM
- retrieve / ground entity-relation
- execute SPARQL trên Freebase

Baseline này đã hỗ trợ KBQA chuẩn, nhưng chưa có một pipeline temporal hoàn chỉnh cho:

- canonical temporal dataset handling
- synthetic temporal data augmentation
- temporal fact mining workflow
- temporal quality filtering
- temporal training export path
- remote execution orchestration cho `Vasi.ai`

### T-ChatKBQA

`T-ChatKBQA` mở rộng baseline theo hai trục chính:

1. **Temporal data construction**
2. **Temporal model adaptation + training workflow**

## Improvement 1: Project Memory System

Đã thêm một memory system trong `docs/project_memory/` để giữ continuity qua nhiều phiên làm việc.

### Mục đích

- giữ mục tiêu project ổn định qua nhiều session
- ghi lại plan, progress, decisions
- giúp agent hoặc người đọc mới nắm trạng thái project nhanh

### Files

- [overview.md](/Users/Kiet/Documents/Code/projects/ChatKBQA/docs/project_memory/overview.md)
- [master_plan.md](/Users/Kiet/Documents/Code/projects/ChatKBQA/docs/project_memory/master_plan.md)
- [progress.md](/Users/Kiet/Documents/Code/projects/ChatKBQA/docs/project_memory/progress.md)
- [decisions.md](/Users/Kiet/Documents/Code/projects/ChatKBQA/docs/project_memory/decisions.md)
- [session_template.md](/Users/Kiet/Documents/Code/projects/ChatKBQA/docs/project_memory/session_template.md)

### Giá trị

- giảm mất ngữ cảnh giữa các phiên
- khóa các quyết định lớn như `TempQuestions-first`, `Vasi.ai`, `Streamlit`
- theo dõi tiến độ implementation rõ hơn

## Improvement 2: Canonical Temporal Data Schema

Đã tạo một schema thống nhất cho dữ liệu temporal trong [schema.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/schema.py).

### Nội dung chính

- `TemporalSample`
- `TemporalQuestionType`
- `TemporalDataSource`
- `ValidationStatus`
- hàm `infer_temporal_question_type(...)`

### Mục đích

- thống nhất representation cho cả human data và synthetic data
- dùng chung cho preprocessing, mining, filtering, training export

### Temporal types hiện hỗ trợ

- `before`
- `after`
- `during`
- `first`
- `last`
- `explicit_date`
- `temporal_answer`

## Improvement 3: TempQuestions Standardization Pipeline

Đã thêm builder để chuẩn hóa `TempQuestions` sang canonical temporal schema trong [builder.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/builder.py).

### Chức năng

- đọc merged TempQuestions artifacts
- convert sang `TemporalSample`
- build temporal relation inventory
- summarize temporal samples

### CLI liên quan

Trong [build_temporal_dataset.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/scripts/build_temporal_dataset.py):

- `standardize-tempquestions`
- `build-relation-inventory`

### Giá trị

- tạo interface đầu vào chuẩn cho toàn bộ temporal pipeline
- giảm phụ thuộc trực tiếp vào format raw/merged cũ của ChatKBQA

## Improvement 4: Temporal Fact Mining Manifest

Đã thêm bước trích temporal mining seeds trong [miner.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/miner.py).

### Chức năng

- parse canonical temporal samples
- trích `TemporalFactSeed`
- build `fact_mining_manifest`
- summarize seeds theo type/family/relation

### Template families hiện có

- `tc_range`
- `argmax`
- `argmin`
- `fallback_temporal`

### CLI liên quan

- `build-fact-mining-manifest`

### Giá trị

- biến human temporal supervision thành mining instructions có cấu trúc
- chuẩn bị dữ liệu trung gian để chạy mining từ Freebase trên `Vasi.ai`

## Improvement 5: Remote Mining Job Specs

Đã thêm remote job batching trong [generator.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/generator.py).

### Chức năng

- gom nhiều seeds thành `RemoteMiningJob`
- gán `query_strategy`
- giữ metadata để remote execution có thể scale theo batch

### Query strategies hiện có

- `cvt_range_lookup`
- `ordering_lookup`
- `temporal_fallback_lookup`

### CLI liên quan

- `build-remote-jobs`

### Giá trị

- tách bước “seed extraction” khỏi bước “remote execution”
- giúp job chạy phù hợp với môi trường `Vasi.ai`

## Improvement 6: Remote Mining Runner

Đã thêm execution layer trung gian trong [remote_executor.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/remote_executor.py).

### Chức năng

- expand batched jobs thành `RemoteQuerySpec`
- build backend-agnostic query payloads
- chuẩn hóa raw executor rows thành `MinedTemporalFact`
- hỗ trợ `dry-run`
- hỗ trợ fixture-backed local testing

### CLI liên quan

- `run-remote-mining`

### Giá trị

- local vẫn test được mà không cần Freebase live
- `Vasi.ai` chỉ cần cắm executor thật vào runner hiện có

## Improvement 7: Synthetic Temporal Sample Generation

Đã thêm logic sinh synthetic temporal samples từ mined facts trong [generator.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/generator.py).

### Chức năng

- convert `MinedTemporalFact` sang `TemporalSample`
- sinh question templates cơ bản
- sinh temporal S-expression phù hợp
- gắn metadata như `source_seed_id`, `mined_fact_id`

### CLI liên quan

- `build-synthetic-samples`

### Giá trị

- tạo augmentation data cho training
- giữ synthetic data cùng schema với human data

## Improvement 8: Synthetic Data Quality Filtering

Đã thêm quality gate cho synthetic temporal samples trong [quality.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/quality.py).

### Chức năng

- review từng sample
- split thành `accepted` và `rejected`
- thêm `filter_issues`
- cập nhật `validation_status`
- summarize rejection reasons

### Các rule đang có

- reject nếu `temporal_type` là `unknown`
- reject nếu `s_expression` chứa `UNKNOWN`
- reject nếu question còn raw MID như `m.xxx`
- reject nếu thiếu `source_seed_id` hoặc `mined_fact_id`
- kiểm tra consistency giữa:
  - `during` và `TC`
  - `last` và `ARGMAX`
  - `first` và `ARGMIN`

### CLI liên quan

- `filter-synthetic-samples`

### Giá trị

- tránh đưa augmentation thô vào train
- giúp synthetic data sạch hơn trước khi train trên `Vasi.ai`

## Improvement 9: Temporal Training Export Path

Đã thêm training-data exporter trong [training.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/training.py).

### Chức năng

- convert `TemporalSample` sang format `examples.json`
- gộp human + synthetic data
- deduplicate theo `input/output`
- hỗ trợ giới hạn `max_synthetic`
- hỗ trợ `require_filtered_synthetic`
- có thể giữ metadata trong export để debug

### CLI liên quan

- `export-training-examples`

### Output

Dataset training mới dùng namespace:

- `TChatKBQA_Freebase_NQ_train`

### Giá trị

- nối temporal pipeline vào training flow hiện có của ChatKBQA
- không phá `process_NQ.py` cũ

## Improvement 10: Dedicated T-ChatKBQA Training Config

Đã thêm config train riêng trong [train_tchatkbqa.yaml](/Users/Kiet/Documents/Code/projects/ChatKBQA/configs/train_tchatkbqa.yaml).

### Mục đích

- tách rõ training config cho `T-ChatKBQA`
- dùng dataset mới `TChatKBQA_Freebase_NQ_train`
- giữ training path tương thích với `llmtuner` / `train_bash.py`

## Improvement 11: Dataset Registry Updates

Đã cập nhật [dataset_info.json](/Users/Kiet/Documents/Code/projects/ChatKBQA/LLMs/data/dataset_info.json) để thêm:

- `TChatKBQA_Freebase_NQ_train`
- `TChatKBQA_Freebase_NQ_test`

### Giá trị

- giúp training/inference của `LLMs/LLaMA` nhận biết dataset mới
- giảm cấu hình tay trước mỗi run

## Improvement 12: Vasi.ai Runbook Wrapper

Đã thêm runbook orchestration cho `Vasi.ai`.

### Files

- [runbook.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/src/temporal_data/runbook.py)
- [run_tchatkbqa_vasi.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/scripts/run_tchatkbqa_vasi.py)

### Chức năng

- build command sequence cho các stage:
  - `prepare`
  - `train`
  - `infer`
  - `eval`
  - `full`
- auto đảm bảo dataset registry có entry cần thiết
- hỗ trợ `dry-run`
- hỗ trợ `--execute` để chạy thật

### Giá trị

- biến toàn bộ temporal pipeline thành một workflow remote rõ ràng
- dễ rerun từng stage trên `Vasi.ai`

## Improvement 13: Test Coverage for New Temporal Pipeline

Đã thêm test cho từng tầng mới của system:

- [test_temporal_dataset.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_dataset.py)
- [test_temporal_miner.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_miner.py)
- [test_temporal_generator.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_generator.py)
- [test_temporal_remote_executor.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_remote_executor.py)
- [test_temporal_training.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_training.py)
- [test_temporal_quality.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_quality.py)
- [test_temporal_runbook.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/tests/test_temporal_runbook.py)

### Mục đích

- giữ pipeline temporal mới có thể refactor an toàn
- cho phép local verification dù chưa chạy Freebase thật

## Improvement 14: CLI Consolidation for Temporal Pipeline

Script [build_temporal_dataset.py](/Users/Kiet/Documents/Code/projects/ChatKBQA/scripts/build_temporal_dataset.py) giờ là entrypoint chính cho phần lớn temporal data pipeline.

### Các subcommands hiện có

- `standardize-tempquestions`
- `build-relation-inventory`
- `build-fact-mining-manifest`
- `build-remote-jobs`
- `run-remote-mining`
- `build-synthetic-samples`
- `filter-synthetic-samples`
- `export-training-examples`

### Giá trị

- gom toàn bộ workflow temporal data vào một CLI rõ ràng
- dễ dùng trên cả local và `Vasi.ai`

## Current End-to-End Flow

Hiện tại pipeline mới có thể đi theo chuỗi logic sau:

1. chuẩn hóa `TempQuestions`
2. build relation inventory
3. trích fact mining manifest
4. build remote mining jobs
5. chạy remote mining runner
6. build synthetic temporal samples
7. filter synthetic samples
8. export training examples
9. chạy wrapper `Vasi.ai` cho `prepare/train/infer/eval`

## What Is Already Working

### Đã code xong

- canonical temporal schema
- TempQuestions standardization
- temporal relation inventory
- temporal fact mining manifest
- batched remote mining jobs
- remote mining runner
- mined fact normalization
- synthetic temporal sample generation
- synthetic quality filtering
- train-ready temporal export
- Vasi.ai orchestration wrapper
- project memory system

### Đã test local

- local fixture-based execution cho remote mining
- local fixture-based export cho training examples
- local fixture-based synthetic filtering
- dry-run full Vasi.ai workflow

## What Is Not Finished Yet

Các phần sau vẫn là next steps:

- chạy Freebase-backed mining thật trên `Vasi.ai`
- sinh synthetic temporal data ở quy mô lớn
- train checkpoint `T-ChatKBQA` thật trên `Vasi.ai`
- chạy inference/evaluation thật
- chốt benchmark protocol và ablation protocol
- tích hợp kết quả thực nghiệm vào report/slides/demo

## Main Design Decisions Locked

- baseline lõi vẫn là `ChatKBQA`
- benchmark chính là `TempQuestions`
- synthetic temporal data dùng để augment train
- heavy jobs chạy trên `Vasi.ai`
- local machine chỉ dùng cho code, test nhẹ, và demo
- synthetic data phải qua quality gate trước khi trust cho train
- workflow remote phải stage-based để rerun từng bước độc lập

## Suggested Usage

### Nếu muốn làm data pipeline local

Ưu tiên dùng:

```bash
python scripts/build_temporal_dataset.py --help
```

### Nếu muốn chuẩn bị run trên Vasi.ai

Ưu tiên dùng:

```bash
python scripts/run_tchatkbqa_vasi.py \
  --stage full \
  --human-input data/temporal/TempQuestions/train.samples.json \
  --synthetic-input data/temporal/synthetic.filtered.samples.json
```

Mặc định script chỉ `dry-run`. Thêm `--execute` khi muốn chạy thật.

## Summary

Các cải tiến đã làm không chỉ thêm vài temporal rule nhỏ vào baseline, mà đã tạo ra một **temporal extension pipeline khá hoàn chỉnh** cho `ChatKBQA`, gồm:

- temporal schema
- temporal data construction workflow
- synthetic data augmentation path
- synthetic quality control
- temporal training export
- remote training/evaluation orchestration

Nói ngắn gọn, project hiện đã có nền kỹ thuật rõ ràng để chuyển từ `ChatKBQA` sang `T-ChatKBQA`; phần còn lại chủ yếu là chạy thực nghiệm thật trên `Vasi.ai` và khóa benchmark/ablation results.
