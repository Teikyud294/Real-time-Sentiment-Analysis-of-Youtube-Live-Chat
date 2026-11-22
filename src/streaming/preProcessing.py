from pyspark.sql import DataFrame
from pyspark.sql.functions import col, trim, to_timestamp

from src.streaming.preprocessing_adapter import normalize_spark_dataframe


# KIỂM TRA DỮ LIỆU HỢP LỆ?
def validate_Messages(DF: DataFrame) -> DataFrame:
    DF = DF.filter(col("message").isNotNull() & (trim(col("message")) != ""))
    DF = DF.filter(col("message_id").isNotNull())

    return DF


# LÀM SẠCH VĂN BẢN
def clean_Messages(DF: DataFrame, domain: str = "ESPORTS") -> DataFrame:
    DF = DF.withColumn("message", trim(col("message")))
    DF = normalize_spark_dataframe(DF, domain=domain, message_column="message")

    return DF


# LOẠI BỎ DỮ LIỆU TRÙNG LẶP
def remove_Duplicates(DF):
    return DF.dropDuplicates(["message_id"])


# CHUẨN HÓA TIMESTAMP
def convert_Timestamps(DF):
    DF = DF.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
    DF = DF.withColumn("collected_at", to_timestamp(col("collected_at"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"))

    return DF


# HÀM TỔNG HỢP ĐỂ IMPORT
def preProcess_Messages(DF: DataFrame, domain: str = "ESPORTS") -> DataFrame:
    DF = validate_Messages(DF)
    DF = clean_Messages(DF, domain=domain)
    DF = convert_Timestamps(DF)

    return DF