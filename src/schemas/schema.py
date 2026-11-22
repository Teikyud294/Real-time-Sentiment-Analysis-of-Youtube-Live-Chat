import emoji
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict
from pyspark.sql.types import StructType, StructField, StringType


@dataclass(frozen=True)
class LiveChatMessage:
    """Data contract chuẩn cho mỗi bản ghi chat đẩy vào Kafka."""
    message_id: str
    video_id: str
    author: str
    message: str
    timestamp: str
    collected_at: str


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


    @staticmethod
    def extract_rawText(item: Any) -> str:
        if not hasattr(item, "messageEx") or not item.messageEx:
            raw_text = str(getattr(item, "message", "") or "")
            return emoji.emojize(raw_text, language="alias")

        parts = []
        for part in item.messageEx:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                emoji_txt = part.get("txt") or ""
                parts.append(emoji_txt)
        
        raw_text = "".join(parts)
        return emoji.emojize(raw_text, language="alias")


    @classmethod
    def format_Message(cls, item: Any, video_id: str) -> Dict[str, Any]:
        record = cls(
            message_id=str(item.id),
            video_id=str(video_id),
            author=str(getattr(getattr(item, "author", None), "name", "Unknown") or "Unknown"),
            message=cls.extract_rawText(item),
            timestamp=str(item.datetime),
            collected_at=datetime.now(timezone.utc).isoformat(),
        )
        return record.to_dict()



SPARK_SCHEMA = StructType([
    StructField("message_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("author", StringType(), True),
    StructField("message", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("collected_at", StringType(), True),
])