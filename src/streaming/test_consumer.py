import os
import json
import time
import uuid
from dotenv import load_dotenv
from kafka import KafkaConsumer
TARGET_RATE = 50


load_dotenv()


def create_consumer(): 
    consumer = KafkaConsumer(
        f"{os.getenv('TOPIC_NAME')}-benchmark-{TARGET_RATE}msg-s",
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=f"benchmark-test-{uuid.uuid4().hex}",
        value_deserializer=lambda data: json.loads(data.decode("utf-8")),
    )
    return consumer


def handle_process(consumer):
    total_receive = 0
    first_msg_time = None
    end_time = None

    try:
        while True:
            records = consumer.poll(timeout_ms=1000)

            for partition, messages in records.items():
                for msg in messages:
                    data = msg.value

                    total_receive += 1

                    if first_msg_time is None:
                        first_msg_time = time.perf_counter()

                    if data.get("_benchmark_control") == "END_OF_REPLAY":
                        end_time = time.perf_counter()
                        return

    finally:
        consumer.close()

        print("\n========== CONSUMER SUMMARY ==========")
        print(f"Total received: {total_receive}")

        if first_msg_time is not None and end_time is not None:
            elapsed = end_time - first_msg_time
            throughput = total_receive / elapsed

            print(f"Elapsed time : {elapsed:.2f} seconds")
            print(f"Throughput   : {throughput:.2f} msg/s")

        print("=======================================")



if __name__ == "__main__":
    consumer = create_consumer()

    handle_process(consumer)