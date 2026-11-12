from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

SCHEMA = StructType([
    StructField("message_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("collected_at", StringType(), True),
    StructField("author", StringType(), True),
    StructField("text", StringType(), True),
    StructField("message_type", StringType(), True),
    StructField("source", StringType(), True),
    StructField("schema_version", StringType(), True)
])