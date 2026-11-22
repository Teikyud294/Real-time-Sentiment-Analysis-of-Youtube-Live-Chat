import json
import time
import logging
import pytchat
from pathlib import Path
from dotenv import load_dotenv
from src.schemas.schema import LiveChatMessage


load_dotenv()
VIDEO_ID = "m0PfPvJaxgU"
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "monitoring"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "collector-09.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8")],
)
logger = logging.getLogger(__name__)


def collect_YtbChat(video_id: str):
    logger.info(f"START COLLECTING DATA: {video_id}")
    output_file = DATA_RAW_DIR / "ytb_live_chat_Esport3.jsonl"
    chat = None

    try:
        chat = pytchat.create(video_id=video_id)

        with open(output_file, mode="a", encoding="utf-8") as file:
            while chat.is_alive():
                for item in chat.get().sync_items():
                    record = LiveChatMessage.format_Message(item, video_id)
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file.flush()
                    logger.info(f"SAVED MESSAGE - Id: {item.id} | Text: {record['message']}")

                time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("STOP CRAWL BY USER!")

    except Exception as exc:
        logger.exception(f"ERROR WHILE CRAWL DATA - {video_id}: {exc}")

    finally:
        if chat is not None:
            try:
                chat.terminate()
            except Exception:
                pass
        logger.info("FINISH CRAWL DATA")


if __name__ == "__main__":
    collect_YtbChat(VIDEO_ID)