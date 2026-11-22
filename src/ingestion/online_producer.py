import os
import json
import time
import logging
import threading
import pytchat
from pathlib import Path
from dotenv import load_dotenv
from kafka import KafkaProducer
from src.schemas.schema import LiveChatMessage

load_dotenv()

VIDEO_ID = os.getenv("VIDEO_ID", "m0PfPvJaxgU")
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "monitoring"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "online-producer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8")],
)
logger = logging.getLogger(__name__)


def create_producer() -> KafkaProducer:
    producer = KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        max_request_size=10485760,
    )
    return producer


def send_message(producer, topic, key, value, on_success=None, on_error=None):
    try:
        future = producer.send(topic, key=key, value=value)
        if on_success:
            future.add_callback(on_success)
        if on_error:
            future.add_errback(on_error)
        return future

    except Exception as error:
        if on_error:
            on_error(error)
        else:
            logger.error(f"[ERROR] Message failed: {error}")
        return None


def handle_online_stream(producer: KafkaProducer, video_id: str, topic: str):
    logger.info(f"START ONLINE STREAMING - Video ID: {video_id} | Topic: {topic}")

    total_crawled = 0
    total_sent = 0
    total_failed = 0
    stats_lock = threading.Lock()

    def on_success(metadata):
        nonlocal total_sent
        with stats_lock:
            total_sent += 1

    def on_error(error):
        nonlocal total_failed
        with stats_lock:
            total_failed += 1
        logger.error(f"Send failed: {error}")

    chat = None
    start_time = time.perf_counter()

    try:
        chat = pytchat.create(video_id=video_id)

        while chat.is_alive():
            for item in chat.get().sync_items():
                record = LiveChatMessage.format_Message(item, video_id)
                total_crawled += 1

                send_message(
                    producer,
                    topic,
                    b"online",
                    record,
                    on_success,
                    on_error,
                )
                logger.info(f"PUSHED MESSAGE - Id: {item.id} | Text: {record['message']}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("STOP ONLINE PRODUCER BY USER!")

    except Exception as exc:
        logger.exception(f"ERROR WHILE STREAMING DATA - {video_id}: {exc}")

    finally:
        if chat is not None:
            try:
                chat.terminate()
            except Exception:
                pass

        # Tín hiệu kết thúc luồng dữ liệu (tương tự cơ chế Poison Pill ở replay_producer)
        end_message = {"_stream_control": "END_OF_ONLINE"}
        end_future = send_message(producer, topic, b"online", end_message)
        if end_future:
            try:
                end_future.get(timeout=30)
            except Exception as exc:
                logger.error(f"Failed to send end control message: {exc}")

        producer.flush()
        producer.close()
        logger.info("FINISH ONLINE STREAMING")

    elapsed = time.perf_counter() - start_time
    actual_rate = total_sent / elapsed if elapsed > 0 else 0.0

    print("\n========== ONLINE PRODUCER SUMMARY ==========")
    print(f"Video ID     : {video_id}")
    print(f"Topic        : {topic}")
    print(f"Total crawled: {total_crawled}")
    print(f"Total sent   : {total_sent}")
    print(f"Total failed : {total_failed}")
    print(f"Elapsed time : {elapsed:.2f} seconds")
    print(f"Actual rate  : {actual_rate:.2f} msg/s")
    print("=============================================")


if __name__ == "__main__":
    topic_name = os.getenv("TOPIC_NAME")
    if not topic_name:
        raise ValueError("Missing TOPIC_NAME in environment variables (.env)")

    kafka_producer = create_producer()
    handle_online_stream(kafka_producer, VIDEO_ID, topic_name)