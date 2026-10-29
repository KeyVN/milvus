# Hướng dẫn Tìm kiếm và Đánh giá Đa Bộ Dữ liệu với Milvus Vector Database

Dự án này tích hợp và vận hành đồng thời **2 bộ dữ liệu** trên **Milvus Vector Database**:
1. **Zalo AI Legal**: 61,425 điều khoản văn bản pháp luật Việt Nam (`zalo_legal`).
2. **Natural Questions VN (NQ-VN)**: 2.68 triệu bài viết/đoạn trích tri thức bách khoa mở kèm bản dịch và gốc tiếng Anh (`nq_vn`).

Toàn bộ dữ liệu vector của cả 2 bộ được quản lý trong cơ sở dữ liệu thống nhất: **`milvus_data.db`**.

> 💡 **Xem hướng dẫn cài đặt và chạy chi tiết từ A-Z:** 👉 **[HUONG_DAN_CHAY.md](file:///d:/New/milvus/HUONG_DAN_CHAY.md)**

---

## 📁 Cấu trúc Thư mục

```text
milvus/
├── requirements.txt          # Danh sách thư viện Python cần cài đặt
├── device_utils.py           # Module quản lý thiết bị GPU/CUDA, auto-detect, FP16, fallback CPU
├── benchmark_all.py          # Script benchmark lần lượt TOÀN BỘ 2 bộ dữ liệu (All-in-One)
├── benchmark_full.py         # Script benchmark thực nghiệm 100% Full Corpus, Queries & Qrels
├── index_milvus.py           # Script trích xuất vector embedding và nạp vào Milvus (hỗ trợ GPU & FP16)
├── search_milvus.py          # Script tìm kiếm (CLI một câu hỏi, REPL tương tác)
├── evaluate_milvus.py        # Script đo lường Recall@K, Latency và QPS trên dữ liệu hiện có
├── benchmark_full_corpus.py  # Script tính toán & dự báo hiệu năng khi nạp 100% TOÀN BỘ dữ liệu
├── benchmark_smart_pool.py   # Script benchmark siêu tốc (Ground-Truth Pool: 100% queries chỉ 3-5 phút)
├── app.py                    # Ứng dụng Web UI tìm kiếm song song bằng Flask
├── templates/
│   └── index.html            # Giao diện Web trực quan (Tailwind CSS, hỗ trợ tab chọn bộ dữ liệu)
├── docker-compose.yml        # File khởi chạy Milvus Standalone Server + Attu Web GUI qua Docker
├── HUONG_DAN_CHAY.md         # HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN TRÊN MÁY MỚI (A - Z)
├── README.md                 # Tài liệu tổng quan dự án
└── milvus_data.db/           # Thư mục lưu trữ database vector cục bộ của Milvus Lite
```

---

## 🚀 1. Benchmark 100% Full Corpus, Queries & Qrels (`benchmark_full.py`)

Script [benchmark_full.py](file:///d:/New/milvus/benchmark_full.py) thực thi đo lường thực nghiệm chính xác tuyệt đối trên **100% dữ liệu thật** (không dùng sample, không dùng pool rút gọn) và **tự động lưu kết quả ra file JSON**:

```bash
# 1. Benchmark 100% Corpus Zalo Legal (61.425 docs, 818 queries, 793 qrels) - Chạy trong ~38 giây:
python benchmark_full.py --dataset zalo --device cuda --fp16

# 2. Benchmark 100% Full Corpus NQ-VN (Tự động nạp nốt 2.68 triệu docs bằng GPU rồi đo 1.900 queries):
python benchmark_full.py --dataset nq --device cuda --fp16

# 3. Chạy lần lượt cả 2 bộ dữ liệu:
python benchmark_full.py --dataset both --device cuda --fp16 --output-json full_results.json
```

---

## ⚡ 2. Benchmark Siêu Tốc Bằng Smart Pool (`benchmark_smart_pool.py`) - KHUYÊN DÙNG

Phương pháp tối ưu cho kho dữ liệu khổng lồ (như NQ-VN 2.68 triệu bài):
- Tự động gom **100% tài liệu đáp án trong `qrels`** (~2.283 docs cho NQ, ~793 docs cho Zalo).
- Bổ sung tài liệu gây nhiễu (Distractors) ngẫu nhiên (ví dụ 5.000 - 15.000 docs).
- Đánh giá **100% câu hỏi** (1.900 queries NQ, 818 queries Zalo) đối chiếu với đáp án chuẩn.
- Thời gian chạy: **chỉ mất 2 ~ 4 phút** thay vì chờ 12 tiếng!

```bash
# 1. Benchmark NQ-VN với 100% câu hỏi (Pool 5.000 docs - Chạy cực nhanh ~2 phút):
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --recreate

# 2. Benchmark NQ-VN với Pool 15.000 docs (Tăng độ khó / nhiều nhiễu hơn):
python benchmark_smart_pool.py --dataset nq --pool-size 15000 --recreate

# 3. Benchmark Zalo Legal bằng Smart Pool:
python benchmark_smart_pool.py --dataset zalo --pool-size 5000 --recreate

# 4. Xuất kết quả ra file JSON:
python benchmark_smart_pool.py --dataset nq --pool-size 5000 --output-json nq_smart_report.json
```

---

## 📊 3. Đánh giá Hiệu năng Hiện tại từ Qrels (`evaluate_milvus.py`)

Script [evaluate_milvus.py](file:///d:/New/milvus/evaluate_milvus.py) tự động đối sánh câu hỏi với nhãn chuẩn **`qrels`** trên số tài liệu hiện có trong Milvus:

```bash
# Đánh giá trên tập Pháp luật (Zalo Legal)
python evaluate_milvus.py --dataset zalo --in-corpus-only

# Đánh giá trên tập Tri thức mở (NQ-VN)
python evaluate_milvus.py --dataset nq --in-corpus-only
```

---

## 🚀 3. Lập chỉ mục dữ liệu vào Milvus (`index_milvus.py`)

Mô hình embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, Cosine metric).

```bash
# Nạp cả 2 bộ dữ liệu cùng lúc
python index_milvus.py --dataset both --limit-zalo 5000 --limit-nq 5000

# Chỉ nạp văn bản Pháp luật (Zalo Legal)
python index_milvus.py --dataset zalo --limit-zalo 5000

# Chỉ nạp Tri thức mở (NQ-VN)
python index_milvus.py --dataset nq --limit-nq 10000

# Nạp lại từ đầu (xóa collection cũ và tạo lại)
python index_milvus.py --dataset nq --limit-nq 5000 --recreate
```

---

## 🔍 4. Tìm kiếm qua dòng lệnh (`search_milvus.py`)

```bash
# Tìm kiếm đồng thời trên cả 2 bộ dữ liệu (Mặc định)
python search_milvus.py --query "Bài hát Không thể không yêu" --dataset both --top-k 3

# Chỉ tìm trong văn bản Pháp luật
python search_milvus.py --query "Quy định xử phạt vi phạm nồng độ cồn" --dataset zalo --top-k 5

# Chế độ tương tác liên tục (REPL)
python search_milvus.py --interactive
```

---

## 🌐 5. Giao diện Web Trực quan (`app.py`)

Khởi động giao diện Web tra cứu song song:

```bash
python app.py
```

Truy cập: **[http://localhost:5000](http://localhost:5000)** trên trình duyệt.
