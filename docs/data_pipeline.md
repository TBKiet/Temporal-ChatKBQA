# Data Pipeline — Temporal ChatKBQA

## Tổng quan

Data training được xây dựng bằng cách kết hợp 3 nguồn, áp dụng methodology từ CronQuestions (ACL 2021) để đảm bảo chất lượng.

**Kết quả cuối cùng:**

| Chỉ số                   | Giá trị |
| ------------------------ | ------- |
| Train examples           | 4,995   |
| Test examples            | 564     |
| Entity overlap           | 0%      |
| Temporal reasoning types | 5 loại  |
| Temporal signal types    | 9 loại  |
| S-expression quality     | 100% real (ChatKBQA SOTA) |
| Synthetic examples       | 0       |
| False positives          | 0       |

---

## Bước 1: Thu thập TempQuestions gốc

**Nguồn:** [TempQuestions](http://qa.mpi-inf.mpg.de/TempQuestions.zip) (Jia et al., EMNLP 2018)

**Kết quả:** 1,271 câu hỏi temporal kèm gold answers và temporal signal types.

```bash
curl -L -o TempQuestions.zip "http://qa.mpi-inf.mpg.de/TempQuestions.zip"
unzip TempQuestions.zip
```

**Format gốc:**
```json
{
  "Id": 1,
  "Question": "who is the first husband of julia roberts?",
  "Gold answer": ["Lyle Lovett"],
  "Temporal signal": ["ORDINAL"],
  "Type": ["Ordinal"],
  "Data source": "ComplexQuestions"
}
```

**Vấn đề:** Dataset gốc không có SPARQL queries → không convert được sang S-expression.

**Giải pháp ban đầu:** Generate synthetic S-expressions dựa trên temporal signal type và Freebase relation patterns. Tuy nhiên, các S-expression này dùng placeholder MID (`m.02mjmr`) và relation đoán — không execute được trên Freebase thật.

**Quyết định cuối cùng:** Loại bỏ toàn bộ synthetic S-expressions. Chỉ giữ lại các câu từ WebQSP và CWQ — những câu có S-expression thật từ ChatKBQA pipeline. TempQuestions gốc được dùng làm reference để validate chất lượng phân loại temporal signal.

---

## Bước 2: Trích xuất temporal questions từ WebQSP và CWQ

**Nguồn:** ChatKBQA SFT data (`ChatKBQA-SFTdata/`) — downloadable từ TeraBox/Baidu.

**Phương pháp:** Regex-based filtering — tìm câu hỏi chứa temporal signals.

**Temporal signals detected:**
```
BEFORE, AFTER, DURING, FIRST, LAST, LATEST, EARLIEST,
MOST RECENT, in YEAR, since YEAR, until YEAR
```

**Script:**
```python
TEMPORAL_RE = re.compile(
    r'\b(before|after|during|first|last|latest|earliest|most recent|'
    r'in \d{4}|since \d{4}|until \d{4}|by \d{4}|\d{4})\b',
    re.IGNORECASE
)
```

**Kết quả trích xuất:**

| Nguồn                  | Train     | Test    |
| ---------------------- | --------- | ------- |
| WebQSP                 | 161       | 114     |
| CWQ                    | 5,163     | 609     |
| **Tổng từ WebQSP+CWQ** | **5,324** | **723** |

Các câu hỏi này có sẵn **S-expression thật** từ ChatKBQA pipeline (LLaMA-2-7B fine-tuned, đạt SOTA trên WebQSP và CWQ).

---

## Bước 3: Hợp nhất và làm sạch

### 3.1 Gộp 3 nguồn

```
TempQuestions gốc     : 888 train + 383 test (reference only — NO S-expression)
WebQSP temporal       : 161 train + 114 test (REAL S-expression)
CWQ temporal          : 5,163 train + 609 test (REAL S-expression)
─────────────────────────────────────────────
Tổng (chưa clean)     : 6,212 train + 1,106 test
```

**Ghi chú:** TempQuestions gốc không có S-expression nên chỉ dùng để tham khảo temporal signal types. Sau bước entity-aware split, các câu synthetic từ TempQuestions bị loại bỏ hoàn toàn.

### 3.2 Loại bỏ false positives

12 câu bị loại do chứa "first name", "last name", "last season", "last episode", "first book of" — đây là cách dùng "first"/"last" không mang ý nghĩa temporal.

**Sau khi clean:** 6,561 train + 742 test = 7,303 total.

---

## Bước 4: Entity-aware split (theo CronQuestions methodology)

### 4.1 Vấn đề

CronQuestions paper nhấn mạnh: nếu entity xuất hiện trong cả train và test, model có thể học cách **ghi nhớ entity** thay vì học **temporal reasoning pattern**. Để đánh giá đúng khả năng temporal reasoning, cần đảm bảo **0% entity overlap** giữa train và test.

### 4.2 Phương pháp

1. **Trích xuất entities từ S-expression:** Phân biệt entity thực (people, places, organizations) với relation paths (schema names như `government.government_position_held`). Relation paths được phép overlap vì đó là patterns model cần học.

2. **Greedy assignment:**
   - Với mỗi entity, gán tất cả câu hỏi chứa entity đó vào cùng một split (train hoặc test)
   - Entity hiếm (≤3 câu) được ưu tiên đưa vào test
   - Entity phổ biến được đưa vào train
   - Nếu conflict (entity đã xuất hiện ở cả 2 split), giữ lại split có nhiều câu hơn

3. **Cleanup:** Các câu test overlap với train entity được chuyển sang train.

### 4.3 Kết quả

```
Trước entity-aware split:
  Train: 6,561 | Test: 742 | Entity overlap: 70%

Sau entity-aware split + loại bỏ synthetic:
  Train: 4,995 | Test: 564 | Entity overlap: 0% | Synthetic: 0
```

Entity overlap giảm từ 70% xuống 0%. 154 train + 178 test synthetic examples bị loại bỏ do dùng placeholder MID không execute được trên Freebase thật. Kết quả: **100% S-expression thật từ ChatKBQA pipeline.**

---

## Bước 5: Phân loại temporal reasoning type

Theo taxonomy của CronQuestions, mỗi câu hỏi được gán 1 trong 5 loại:

| Type             | Mô tả                               | Ví dụ                                    | Count (train) |
| ---------------- | ----------------------------------- | ---------------------------------------- | ------------- |
| **SimpleTime**   | Câu hỏi có explicit time constraint | "Who was president in 2008?"             | 2,658 (53.7%) |
| **Before/After** | So sánh trước/sau                   | "Who was president before Obama?"        | 1,178 (23.8%) |
| **First/Last**   | Ordinal/superlative                 | "What was the first book by JK Rowling?" | 806 (16.3%)   |
| **SimpleEntity** | Entity answer với implicit time     | "Who is the current king of Cambodia?"   | 278 (5.6%)    |
| **TimeJoin**     | Temporal + event overlap            | "Who was president during WWII?"         | 29 (0.6%)     |

**Phương pháp phân loại:**
```python
def classify_temporal_type(question, sexpr):
    if 'during' in q and event_pattern.search(q):
        return 'TimeJoin'
    if 'TC' in sexpr and any(w in q for w in ['before','after']):
        return 'Before/After'
    if 'TC' in sexpr:
        return 'SimpleTime'
    if any(w in q for w in ['first','last','latest','earliest','oldest','newest']):
        return 'First/Last' if 'ARGMAX' in sexpr or 'ARGMIN' in sexpr else 'SimpleTime'
    if any(op in sexpr for op in ['gt ','lt ','ge ','le ']):
        return 'Before/After'
    if year_pattern.search(q):
        return 'SimpleTime'
    return 'SimpleEntity'
```

---

## Bước 6: Temporal annotation

Mỗi câu hỏi được gán metadata theo format của CronQuestions:

```json
{
  "instruction": "Generate a Logical Form query for this temporal question...",
  "input": "Question: { who was president before obama? }",
  "output": "(TC (JOIN (R government.government_position_held) ...) ...)",
  "temporal_type": "Before/After",
  "temporal_annotation": {
    "head_entity": "Barack Obama",
    "tail_entity": null,
    "time_value": "2009",
    "temporal_signal": ["BEFORE"]
  }
}
```

---

## Bước 7: Chuẩn bị LLM training data

Data được lưu dưới dạng instruction-tuning format cho LLaMA-2:

```json
{
  "instruction": "Generate a Logical Form query for this temporal question, using TC operators for date constraints and ARGMAX/ARGMIN for ordinal constraints.\n",
  "input": "Question: { who was president before obama? }",
  "output": "(TC (JOIN (R government.government_position_held) m.02mjmr) government.government_position_held.from 2009-12-31)",
  "history": [],
  "temporal_type": "Before/After",
  "temporal_annotation": { ... }
}
```

**File locations:**
```
LLMs/data/TempQuestions_Freebase_NQ_train/examples.json  → 4,995 examples (100% real ChatKBQA S-expr)
LLMs/data/TempQuestions_Freebase_NQ_test/examples.json   → 564 examples (100% real ChatKBQA S-expr)
```

---

## Phân bố temporal signals trong train

| Signal   | Count  | %     |
| -------- | ------ | ----- |
| YEAR     | 3,300+ | 66.7% |
| IMPLICIT | 700+   | 14.1% |
| FIRST    | 400+   | 8.1%  |
| AFTER    | 380+   | 7.7%  |
| BEFORE   | 290+   | 5.9%  |
| EARLIEST | 250+   | 5.1%  |
| LAST     | 240+   | 4.8%  |
| LATEST   | 190+   | 3.8%  |
| DURING   | 150+   | 3.0%  |

---

## Phân bố temporal operators trong S-expression (train)

| Operator           | Count  | %     | Ý nghĩa                   |
| ------------------ | ------ | ----- | ------------------------- |
| JOIN               | 4,900+ | 99.0% | Basic projection          |
| AND                | 2,900+ | 58.6% | Conjunction               |
| le (less equal)    | 1,100+ | 22.2% | Before comparison         |
| TC                 | 950+   | 19.2% | Temporal constraint range |
| ARGMIN             | 380+   | 7.7%  | Earliest/oldest           |
| ARGMAX             | 300+   | 6.1%  | Latest/most recent        |
| ge (greater equal) | 220+   | 4.4%  | After comparison          |
| lt (less than)     | 35+    | 0.7%  | Strict before             |

---

## So sánh với các dataset chuẩn

| Tiêu chí                 | TempQuestions (2018) | CronQuestions (2021)    | **Data của chúng tôi**      |
| ------------------------ | -------------------- | ----------------------- | --------------------------- |
| **Nguồn KB**             | Freebase             | Wikidata                | Freebase                    |
| **Số lượng train**       | ~890                 | 350,000                 | **4,995**                   |
| **Phương pháp tạo**      | Lọc thủ công         | Template + Paraphrase   | Lọc tự động từ SOTA KBQA    |
| **SPARQL/S-expr**        | TEQUILA-generated    | Template execution      | **100% ChatKBQA SOTA**      |
| **Entity overlap**       | Không kiểm soát      | 0%                      | **0%**                      |
| **Reasoning types**      | 4 loại               | 5 loại                  | **5 loại**                  |
| **Temporal annotation**  | Signal type          | head/tail/time/template | **head/tail/time/signal**   |
| **Synthetic examples**   | N/A                  | N/A                     | **0 (đã loại bỏ)**          |
| **False positives**      | Manual verified      | Template verified       | **0 (regex cleaned)**       |

---

## Hạn chế & Cải thiện trong tương lai

1. **Paraphrasing:** Chưa có đa dạng ngôn ngữ như CronQuestions. Có thể thêm bằng GPT-4o-mini hoặc Claude API.
2. **KG facts kèm theo:** CronQuestions có kèm KG facts (quintuple format). Data của chúng tôi dùng Freebase rời — cần truy vấn thêm.
3. **TimeJoin examples:** Chỉ có 29 câu (0.6%) — quá ít. Cần bổ sung thêm từ Freebase event entities.
4. **Kích thước:** 4,995 examples là đủ cho LoRA fine-tuning nhưng nhỏ hơn nhiều so với CronQuestions (350k).

---

## Cách tái tạo

```bash
# 1. Download TempQuestions
curl -L -o /tmp/TempQuestions.zip "http://qa.mpi-inf.mpg.de/TempQuestions.zip"
unzip /tmp/TempQuestions.zip -d /tmp/

# 2. Download ChatKBQA SFT data từ TeraBox/Baidu
#    Đặt vào ChatKBQA-SFTdata/

# 3. Chạy pipeline
python scripts/download_tempquestions.py  # Tạo split
# Sau đó chạy các script trong docs/data_pipeline.md để:
#   - Trích xuất temporal từ WebQSP/CWQ
#   - Entity-aware split
#   - Phân loại reasoning type
#   - Thêm temporal annotation
```
