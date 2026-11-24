# Real-time-Sentiment-Analysis-of-Youtube-Live-Chat
IE212 Final Project

## Yêu cầu môi trường (Prerequisites)
- **Python**: Khuyến nghị phiên bản **Python 3.11** (hỗ trợ tốt nhất cho PySpark 3.5 và NumPy 1.26).
- **Java**: Java 8/11/17 (yêu cầu để chạy Apache Spark / PySpark).

## Cài đặt môi trường

```powershell
# 1. Tạo môi trường ảo với Python 3.11
py -3.11 -m venv .venv

# 2. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 3. Nâng cấp pip và cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Huấn luyện mô hình cảm xúc

Ba CSV mặc định ở `data/processed/train_{esport,football,music}.csv`, mã hóa UTF-8 với hai cột `text,label`. Nhãn `0` là positive, `1` là negative và `2` là neutral. Dùng `--data-dir` hoặc `--esport-csv`, `--football-csv`, `--music-csv` để đổi nguồn.

### CPU · TF-IDF + Logistic Regression

Từ thư mục gốc, có thể cài riêng các thư viện dùng cho baseline.

```bash
python -m pip install numpy pandas scikit-learn matplotlib joblib regex emoji
python train_tfidf_lr.py
python evaluate.py
```

Dùng `--search none --scenario in-domain` để chạy nhanh, hoặc `--search grid` để thử toàn bộ lưới tham số. Randomized search mặc định thử 12 tổ hợp với CV 5 fold. Dùng `--n-jobs` để điều chỉnh số tiến trình CPU.

### Google Colab · Transformer

Chọn runtime GPU T4 và chuyển thư mục làm việc đến thư mục gốc của repo, có đủ ba CSV và `data/dictionaries`. Colab dùng PyTorch có sẵn nếu phiên bản từ 2.6 trở lên. Trong một ô notebook, cài các phụ thuộc NLP.

```python
%pip install "torch>=2.6,<3" "transformers>=4.57.1,<6" "datasets>=3,<6" "accelerate>=1.1,<2" "sentencepiece>=0.2,<1" "protobuf>=4,<8" pyvi emoji regex scikit-learn pandas matplotlib joblib
```

Chạy ViSoBERT với ba seed mặc định.

```bash
python train_bert.py --model-name uitnlp/visobert
python evaluate.py
```

Chọn `vinai/phobert-base-v2` để dùng PhoBERT với tách từ PyVi tự động, hoặc `xlm-roberta-base` để dùng đối chứng đa ngữ. Dùng `--learning-rates 2e-5 1e-5 3e-5 5e-5` để chọn learning rate theo macro-F1 của validation cho từng seed và tập chia. Nếu thiếu VRAM, thử `--batch-size 16 --eval-batch-size 16 --gradient-accumulation-steps 2`. Trong ô Colab, thêm `!` trước lệnh shell. [ViSoBERT](https://huggingface.co/uitnlp/visobert) · [PhoBERT](https://huggingface.co/vinai/phobert-base-v2) · [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base)

### Toàn bộ thí nghiệm và ablation

```bash
bash run_all.sh all
bash run_all.sh baseline
ABLATIONS=0 BERT_MODELS=uitnlp/visobert bash run_all.sh bert --scenario in-domain
```

Lệnh `all` chạy baseline và ba Transformer trên bốn tập chia, với ba seed cho mỗi Transformer và ba ablation. Các checkpoint có thể chiếm hàng chục GiB. Dùng `--output-dir` để lưu vào ổ đĩa lâu dài, và dùng thư mục kết quả riêng khi đổi dữ liệu, seed hoặc cách chia. Có thể giới hạn một miền test bằng `--scenario cross-domain --held-out-domain music`.

- Cấu hình gốc giữ emoji Unicode và emote, giữ tên riêng, dùng trọng số lớp. Mỗi ablation đổi một yếu tố. Có thể chạy riêng bằng `--variant` cùng `--emoji-mode remove`, `--mask-entities` hoặc `--no-class-weight`.
- Các cờ bổ sung gồm `--emoji-mode demojize`, `--teencode`, `--mask-numbers` và `--no-normalize`. Che tên dùng alias từ từ điển repo. Thêm từ teencode vào `TEENCODE` trong `data_utils.py`.
- In-domain chia xấp xỉ 70/15/15 theo nhãn sau khi gộp và lọc trùng. Một nội dung trùng và cùng nhãn ở nhiều miền được gán miền của dòng hợp lệ đầu tiên theo thứ tự esport, football, music. Các dòng cùng nội dung nhưng khác nhãn nằm chung một tập chia. Nếu dự kiến dưới 5 nhóm của một lớp trong val hoặc test, dùng 5 fold ngoài, mỗi fold giữ riêng validation trong phần còn lại. Cũng có thể bật `--force-kfold`.
- Cross-domain giữ miền test và bỏ các nội dung trùng với test khỏi hai miền nguồn, rồi chia 85/15 cho train và validation. Khóa lọc trùng cố định dùng NFC, chữ thường, khoảng trắng, URL, mention, ký tự lặp và giữ emoji. Các phép che tùy chọn dùng chung tập chia này.

### Đọc kết quả

Mặc định kết quả nằm trong `output/sentiment/`. Chạy `python evaluate.py --output-dir <thư-mục>` sau khi huấn luyện.

| File | Nội dung |
| --- | --- |
| `comparison.csv` | Macro-F1, weighted-F1, accuracy, F1 từng lớp, thời gian và kích thước. Có dòng toàn tập và theo miền. |
| `run_metrics.csv` | Precision, recall, F1, support từng lớp và confusion matrix cho từng seed/fold. |
| `ablation.csv` | Δ macro-F1 so với cấu hình gốc trên cùng seed và tập chia. |
| `analysis.md` | Chênh lệch BERT so với baseline và lưu ý khi lớp hiếm có ít mẫu. |
| `runs/…` | Model, cấu hình tiền xử lý, dự đoán CSV, lỗi liên quan lớp negative ở đầu file, confusion matrix PNG và kết quả tinh chỉnh. Baseline có thêm `top_features.csv`. |

`seed_std` và `fold_std` là độ lệch chuẩn mẫu, để trống khi chỉ có một lần đo tương ứng. Khi có nhiều seed và fold, độ lệch chuẩn qua seed lấy điểm đã trung bình trên các fold, và ngược lại. Thời gian huấn luyện gồm tiền xử lý tập train và tìm siêu tham số, loại trừ thời gian tải checkpoint tiền huấn luyện. Thời gian suy luận là trung vị của ba lần đo trên cùng 1.000 bình luận, gồm tiền xử lý, trên thiết bị ghi trong bảng. Nếu test có dưới 1.000 dòng, phép đo lấy mẫu có hoàn lại.

Để nạp lại baseline, dùng `joblib.load` với `model.joblib` và khởi tạo `Preprocessor(**config)` từ `preprocessing.json` trước khi gọi `predict_proba`. Với Transformer, nạp model và tokenizer từ `best_model`, áp dụng cấu hình tiền xử lý lưu cùng thư mục và dùng `max_length` trong `config.json` của lần chạy.
