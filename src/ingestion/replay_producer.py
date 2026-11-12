import os
import json
import time
import threading
from dotenv import load_dotenv
from kafka import KafkaProducer
TARGET_RATE = 100


load_dotenv()


def read_jsonl(file_path):
    
    messages = []
    with open(file_path, "r", encoding="utf-8") as file:     
        
        for idx, line in enumerate(file):
            line = line.strip()

            try:
                message = json.loads(line)
                messages.append(message)

            except json.JSONDecodeError as error:
                print(f"Invalid JSON at line {idx}: {error}")

    return messages


def create_producer():

    producer = KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        max_request_size=10485760
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
            print(f"[ERROR] Message failed: {error}")
        return None



def handle_process(producer, data):
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
        print(f"Send failed: {error}")

    interval = 1 / TARGET_RATE
    start_time = time.perf_counter()
    next_send_time = start_time

    try:
        for msg in data:
            wait_time = next_send_time - time.perf_counter()
            if wait_time > 0:
                time.sleep(wait_time)

            send_message(
                producer, 
                f"{os.getenv('TOPIC_NAME')}-benchmark-{TARGET_RATE}msg-s", 
                b"replay", 
                msg,
                on_success,
                on_error
            )

            next_send_time += interval

        # Tín hiệu kết thúc benchmark (Poison Pill)
        end_message = {"_benchmark_control": "END_OF_REPLAY"}
        end_future = send_message(
            producer,
            f"{os.getenv('TOPIC_NAME')}-benchmark-{TARGET_RATE}msg-s", 
            b"replay",
            end_message
        )
        if end_future:
            end_future.get(timeout=30)

        producer.flush()

    finally:
        producer.close()

    elapsed = time.perf_counter() - start_time
    actual_rate = total_sent / elapsed

    print("\n========== PRODUCER SUMMARY ==========")
    print(f"Total loaded : {len(data)}")
    print(f"Total sent   : {total_sent}")
    print(f"Total failed : {total_failed}")
    print(f"Elapsed time : {elapsed:.2f} seconds")
    print(f"Target rate  : {TARGET_RATE} msg/s")
    print(f"Actual rate  : {actual_rate:.2f} msg/s")
    print("======================================")


if __name__ == "__main__":
    data = read_jsonl(os.getenv("REPLAY_FILE"))
    producer = create_producer()

    handle_process(producer, data)
