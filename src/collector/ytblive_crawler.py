import logging
import json
from pathlib import Path
from datetime import datetime, timezone
import pytchat

# Thiết lập đường dẫn dựa trên cấu trúc thư mục (từ src/ingestion/ lùi về root)
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
# Đặt file log trong thư mục monitoring phù hợp với ngữ cảnh theo dõi hệ thống
LOG_DIR = ROOT_DIR / "monitoring"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình logging: Chỉ dùng FileHandler để ghi vào file, không in ra terminal
log_file = LOG_DIR / "data_collector.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def collect_youtube_chat(video_id: str):
    logger.info(f"Bắt đầu thu thập dữ liệu chat cho video: {video_id}")
    
    # Định dạng JSON Lines (.jsonl) hỗ trợ tốt cho dữ liệu linh động và lưu trữ emoji
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DATA_RAW_DIR / f"chat_{video_id}_{timestamp_str}.jsonl"
    
    try:
        chat = pytchat.create(video_id=video_id)
        
        # Mở file với utf-8 để giữ nguyên emoji và icon
        with open(output_file, mode="a", encoding="utf-8") as f:
            while chat.is_alive():
                for c in chat.get().sync_items():
                    record = {
                        "message_id": c.id,
                        "video_id": video_id,
                        "timestamp": c.datetime,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "author": c.author.name,
                        "text": c.message,
                        "message_type": c.type if hasattr(c, 'type') else "textMessage",
                        "source": "youtube_live",
                        "schema_version": "1.0"
                    }
                    
                    # ensure_ascii=False đảm bảo emoji không bị ép kiểu sang mã unicode dạng chuỗi
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    logger.info(f"Đã lưu tin nhắn ID: {c.id}")
                    
    except Exception as e:
        logger.error(f"Lỗi khi thu thập dữ liệu video {video_id}: {e}")
    finally:
        logger.info("Kết thúc quá trình thu thập.")

if __name__ == "__main__":
    # Thay thế bằng ID video YouTube thực tế khi chạy (ví dụ đoạn mã sau v=...)
    TARGET_VIDEO_ID = "V13LZc75GUo"
    collect_youtube_chat(TARGET_VIDEO_ID)