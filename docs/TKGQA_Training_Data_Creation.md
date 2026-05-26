# Tạo Dữ Liệu Training cho Temporal Knowledge Graph Question Answering (TKGQA)

Tổng hợp từ survey [TKGQA-Survey](https://github.com/cosmicexotic/TKGQA-Survey) và các paper liên quan.

---

## 1. Template-based Generation (CronQuestions — 410K QA pairs)

**Paper:** *Question Answering Over Temporal Knowledge Graphs* (ACL 2021)
**Code:** https://github.com/apoorvumang/CronKGQA

Đây là phương pháp quy mô lớn nhất và thực tế nhất hiện nay.

### Pipeline

| Bước | Mô tả |
|------|-------|
| **Xây dựng TKG** | Lấy facts có temporal annotation từ Wikidata (~328K facts, 125K entities, 203 relations). Chỉ giữ facts có `start time`/`end time`. Discretize về year-level. |
| **Tạo seed templates** | Dùng 5 relation phổ biến nhất (`member of sports team`, `position held`, `award received`, `spouse`, `employer`) để tạo **30 seed templates** với placeholders: `{head}`, `{tail}`, `{time}`, `{event}`, `{type}` (before/after), `{adj}` (first/last). |
| **5 kiểu reasoning** | Simple, Before/After, First/Last, Time Join. |
| **Human paraphrasing** | Người annotate paraphrase 30 templates → **246 templates** đa dạng hơn về ngôn ngữ. |
| **Automatic paraphrasing** | Dùng monolingual paraphrasing model (Hu et al., 2019) → **654 templates** sau khi human verify. |
| **Slot-filling** | Điền entities/times thật từ Wikidata aliases vào templates → **410K QA pairs**. |
| **Train/test split** | Không overlap về paraphrase template lẫn entity giữa train và test. Chỉ cho phép overlap event entity. |

### Ví dụ câu hỏi theo loại reasoning

| Loại | Ví dụ |
|------|-------|
| Simple | *"Who was the President of the USA in 2008?"* |
| Before/After | *"Who was the President before Obama?"* |
| First/Last | *"Who was the first President of the USA?"* |
| Time Join | *"Who played alongside Roberto Dinamite for Brazil?"* |

### Ưu / Nhược điểm

- **Ưu:** Quy mô lớn, có thể tái tạo, đa dạng về reasoning type.
- **Nhược:** Phân phối câu hỏi artificial, bị giới hạn bởi chất lượng templates.

---

## 2. Aggregation + Temporal Annotation (TimeQuestions — 16K questions)

**Paper:** *Complex Temporal Question Answering on Knowledge Graphs* (CIKM 2021)
**Code:** https://github.com/zhenjia1107/EXAQT

### Pipeline

| Bước | Mô tả |
|------|-------|
| **Thu thập** | Gom câu hỏi từ LC-QuAD 1.0/2.0, ComplexQuestions, WebQuestions (các benchmark KG-QA có sẵn). |
| **Temporal expression tagging** | Dùng **SUTime** và **HeidelTime** để tag temporal expressions trong câu hỏi. |
| **Signal word detection** | Dictionary-based để phát hiện từ khóa thời gian: "before", "after", "during", "first", "last". |
| **Phân loại** | 4 loại: Explicit Temporal, Implicit Temporal, Ordinal, Temporal Answer. |
| **Map sang Wikidata** | Dùng Wikipedia cross-links từ DBpedia/Freebase để map entities sang Wikidata. |

### 4 loại câu hỏi

| Loại | Mô tả | Ví dụ |
|------|-------|-------|
| Explicit Temporal | Chứa time expression rõ ràng | *"...in 2009"* |
| Implicit Temporal | Time reference ẩn | *"when Obama became president"* |
| Ordinal | Ràng buộc thứ tự | *"first president"* |
| Temporal Answer | Hỏi về thời gian | *"When did..."* |

### Ưu / Nhược điểm

- **Ưu:** Câu hỏi tự nhiên hơn template-based.
- **Nhược:** Quy mô nhỏ, phụ thuộc vào dataset có sẵn, coverage hạn chế.

---

## 3. LLM Self-Improvement Programming (Prog-TQA — 2024)

**Paper:** *Self-Improvement Programming for Temporal Knowledge Graph Question Answering* (LREC-COLING 2024)
**Code:** https://github.com/DeepLearnXMU/Prog-TQA

Hướng hiện đại nhất: dùng LLM để tự sinh và tự cải thiện dữ liệu training.

### Pipeline

| Bước | Mô tả |
|------|-------|
| **Thiết kế temporal operators** | Định nghĩa các toán tử cơ bản: `before`, `after`, `first`, `last`, `during`, `overlap`. |
| **Few-shot program generation** | Dùng LLM (GPT-4) với in-context learning để sinh program drafts từ câu hỏi tự nhiên. |
| **Program execution** | Execute program trên TKG thật để lấy answer. |
| **Lọc high-quality drafts** | Programs nào execute trả về đúng answer → giữ lại làm pseudo-labeled training data. |
| **Self-improvement loop** | Dùng các high-quality drafts làm exemplars bổ sung → cải thiện LLM generation ở vòng sau. |

### Cơ chế Self-Improvement

```
Vòng 1: Few-shot prompting → sinh program drafts
       → Execute trên TKG → lọc correct drafts
       → Thêm vào example pool

Vòng 2: Dùng example pool (đã mở rộng) làm few-shot examples
       → Sinh program drafts chất lượng cao hơn
       → Tiếp tục lọc và mở rộng pool

... lặp đến khi hội tụ
```

### Ưu / Nhược điểm

- **Ưu:** Không cần human annotation, tự động mở rộng, chất lượng cải thiện qua từng vòng.
- **Nhược:** Cần TKG có sẵn để execute program, phụ thuộc chất lượng LLM ban đầu.

---

## 4. LLM as Programming (QAaP — EMNLP 2023)

**Paper:** *Question Answering as Programming for Solving Time-Sensitive Questions* (EMNLP 2023)
**Code:** https://github.com/TianHongZXY/qaap

### Ý tưởng chính

Thay vì hỏi LLM trả lời trực tiếp, chuyển câu hỏi tự nhiên thành **code** để execute:

1. LLM sinh code (Python/SQL-like) từ câu hỏi
2. Code chứa temporal constraints được execute trên knowledge base
3. Trả về answer từ kết quả execute

### Cách dùng để tạo training data

- Dùng LLM sinh nhiều code variants cho cùng 1 câu hỏi
- Execute để verify tính đúng đắn
- Giữ lại các (question, code, answer) triplets đúng làm training data
- Có thể dùng để train model nhỏ hơn (knowledge distillation)

---

## 5. Instruction Tuning với Subgraph Retrieval (GenTKGQA — ACL 2024)

**Paper:** *Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models* (ACL 2024 Findings)

### Pipeline

| Giai đoạn | Mô tả |
|-----------|-------|
| **Subgraph Retrieval** | Dùng LLM trích xuất temporal constraints + structural links từ câu hỏi, thu hẹp không gian tìm kiếm theo cả 2 chiều thời gian và cấu trúc. |
| **Virtual Knowledge Indicators** | Fuse GNN signals từ subgraph vào LLM text representations. |
| **Instruction Tuning** | Fine-tune open-source LLM với format: `question → subgraph → answer`. |

### Cách tạo data

Dùng chính pipeline này để sinh (question, subgraph, answer) triplets từ TKG có sẵn → dùng làm training data cho instruction tuning. Đặc biệt hiệu quả khi cần fine-tune model nhỏ (7B-13B) cho TKGQA.

---

## 6. Multi-Granularity Generation (MultiTQ — ACL 2023)

**Paper:** *Multi-granularity Temporal Question Answering over Knowledge Graphs* (ACL 2023)
**Code:** https://github.com/czy1999/MultiTQ

### Điểm khác biệt

Hỗ trợ **nhiều mức độ chi tiết thời gian** (day, month, year) trong cùng một câu hỏi — realistic hơn các dataset trước đây.

### Ví dụ

| Granularity | Ví dụ |
|-------------|-------|
| Month + Year | *"Who condemned Abhisit Vejjajiva in **May 2010**?"* |
| Year only | *"Who was the first to visit the Middle East in **2008**?"* |
| Exact date | *"When did the Aam Aadmi Party first negotiated with Harish Rawat?"* → 2015-12-13 |
| Before + date | *"Who expressed intent to engage in diplomatic cooperation with Ethiopia **before Jun 25th, 2006**?"* |

### Cách tạo

- Templates được thiết kế để trộn các granularity khác nhau
- Dùng NER (`flair/ner-english-large`) để annotate entity/time expressions
- KG embedding (TComplEx) để tạo negative samples

---

## 7. Các Dataset Khác

| Dataset | Năm | Đặc điểm | Code |
|---------|-----|----------|------|
| **TempQuestions** | 2018 | Dataset temporal QA đầu tiên (1,271 câu) | TEQUILA |
| **CronQuestions** | 2021 | Lớn nhất (410K), template-based | CronKGQA |
| **Complex-CronQuestions** | 2022 | Mở rộng CronQuestions với complex reasoning | SubGTR |
| **ForecastTKGQuestions** | 2022 | Tập trung vào forecasting (dự đoán tương lai) | Có |
| **MusTQ** | 2024 | Multi-step temporal reasoning | Có |
| **TIQ** | 2024 | Faithfulness over heterogeneous sources | FAITH |

---

## Khuyến Nghị Pipeline Tạo Dữ Liệu Training

Kết hợp các phương pháp trên, đây là pipeline khả thi nhất:

```
Bước 1: Xây dựng TKG từ Wikidata
  ├── Query facts có temporal annotations (P580: start_time, P582: end_time)
  ├── Discretize timestamps về year-level (hoặc giữ granularity gốc nếu cần multi-granularity)
  └── Lọc các relation phổ biến, cân bằng phân phối

Bước 2: Tạo templates
  ├── Thiết kế 20-30 seed templates với 5 kiểu temporal reasoning
  ├── Paraphrase bằng LLM (GPT-4o) để tăng diversity (theo CronQuestions pattern)
  └── Verify thủ công một subset để đảm bảo chất lượng

Bước 3: Slot-filling tự động
  ├── Điền entities/times thật từ Wikidata vào templates
  ├── Dùng LLM để verify chất lượng QA pairs được sinh
  └── Lọc bỏ các pairs không hợp lý (inconsistent temporal logic)

Bước 4: Self-improvement với LLM (theo Prog-TQA)
  ├── LLM sinh program/query từ mỗi câu hỏi
  ├── Execute trên TKG để verify answer
  └── Giữ lại high-quality pairs, thêm vào training set

Bước 5: Train/Test split
  ├── Không overlap entity giữa train và test
  ├── Không overlap template paraphrase
  └── Stratify theo reasoning type để đảm bảo cân bằng

Bước 6 (Optional): Instruction tuning data
  ├── Format: {question, temporal_constraints, subgraph, answer}
  ├── Dùng cho fine-tune open-source LLM (Qwen, LLaMA, Phi)
  └── Có thể kết hợp negative sampling từ TKG embeddings
```

---

## Tài Liệu Tham Khảo

| Paper | Link | Code |
|-------|------|------|
| CronKGQA (ACL 2021) | https://arxiv.org/abs/2106.01515 | https://github.com/apoorvumang/CronKGQA |
| EXAQT (CIKM 2021) | https://arxiv.org/abs/2109.08935 | https://github.com/zhenjia1107/EXAQT |
| QAaP (EMNLP 2023) | https://arxiv.org/abs/2305.14221 | https://github.com/TianHongZXY/qaap |
| MultiTQ (ACL 2023) | https://aclanthology.org/2023.acl-long.637/ | https://github.com/czy1999/MultiTQ |
| Prog-TQA (LREC-COLING 2024) | https://arxiv.org/abs/2404.01720 | https://github.com/DeepLearnXMU/Prog-TQA |
| GenTKGQA (ACL 2024) | https://arxiv.org/abs/2402.16568 | — |
| TimeR4 (2024) | — | https://github.com/Nankai-Know/TimeR4 |
| Survey TKGQA | https://arxiv.org/abs/2406.14191 | https://github.com/cosmicexotic/TKGQA-Survey |
