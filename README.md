# 🚀 PySpark & Databricks Customer Segmentation Pipeline

An end-to-end data pipeline built on Databricks using PySpark and Delta Lake. This project processes raw e-commerce clickstream and transaction logs using a **Medallion Architecture** (Bronze → Silver → Gold) and applies K-Means Clustering to segment users based on purchasing behavior.

---

## 🏗️ Architecture & Workflow

1. **Bronze Layer (Raw Ingestion):** Ingests raw JSON clickstream and transaction event logs directly into Delta Lake format.
2. **Silver Layer (Cleaning & Sessionization):** Filters out invalid sessions, deduplicates data, and creates standardized session metrics.
3. **Gold Layer (Feature Store & ML):** Aggregates customer behavior (`Total Spend`, `Visit Frequency`, `Avg Cart Size`) and runs a PySpark MLlib `KMeans` pipeline to cluster customers into 4 distinct profiles.

---

## 📊 Customer Segments

* **High-Value Loyal:** High purchase frequency & high overall spend.
* **Cart Abandoners:** High average cart sizes with low transaction completion.
* **At-Risk Spenders:** Moderate spend with declining visit frequency over time.
* **Occasional Shoppers:** Low spend and low interaction frequency.

---

## 🛠️ Tech Stack & Prerequisites

* **Platform:** Databricks (Community or Enterprise Edition)
* **Processing:** PySpark DataFrame API, Spark SQL
* **Storage Format:** Delta Lake (ACID transactions, Time Travel)
* **Machine Learning:** PySpark MLlib (`VectorAssembler`, `StandardScaler`, `KMeans`)

---

## 🚀 How to Run

1. Clone or import the provided `pyspark_pipeline.py` / `.ipynb` notebook into your Databricks workspace.
2. Attach the notebook to an active Spark Cluster (Runtime 11.3 LTS or higher).
3. Run all cells sequentially to generate synthetic data, build Delta tables, and execute the clustering model.

# Databricks PySpark Customer Segmentation & Analytics Pipeline
from datetime import datetime, timedelta
import random
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# 1. Ingestion & Bronze Layer (Raw Logs)
schema = StructType([
    StructField("session_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("event_time", TimestampType(), True),
    StructField("event_type", StringType(), True),
    StructField("items_in_cart", IntegerType(), True),
    StructField("purchase_amount", DoubleType(), True),
])

raw_data = []
customers = [f"cust_{i:03d}" for i in range(1, 101)]
event_types = ["view", "cart", "purchase"]

for _ in range(5000):
  session_id = f"sess_{random.randint(10000, 99999)}"
  customer_id = random.choice(customers)
  event_time = datetime.now() - timedelta(
      days=random.randint(0, 30), hours=random.randint(0, 23)
  )
  event_type = random.choices(event_types, weights=[0.6, 0.3, 0.1], k=1)[0]
  items_in_cart = random.randint(1, 5)
  purchase_amount = (
      round(random.uniform(15.0, 300.0), 2) if event_type == "purchase" else 0.0
  )
  raw_data.append((
      session_id,
      customer_id,
      event_time,
      event_type,
      items_in_cart,
      purchase_amount,
  ))

df_raw = spark.createDataFrame(raw_data, schema)
df_raw.write.format("delta").mode("overwrite").save(
    "/tmp/delta/bronze_customer_logs"
)

# 2. Transformation & Silver Layer
df_silver = df_raw.filter(F.col("customer_id").isNotNull()).dropDuplicates()
df_silver.write.format("delta").mode("overwrite").save(
    "/tmp/delta/silver_customer_transactions"
)

# 3. Gold Layer & ML K-Means Clustering
customer_features = df_silver.groupBy("customer_id").agg(
    F.sum("purchase_amount").alias("total_spend"),
    F.count("session_id").alias("visit_count"),
    F.avg("items_in_cart").alias("avg_cart_size"),
)

assembler = VectorAssembler(
    inputCols=["total_spend", "visit_count", "avg_cart_size"],
    outputCol="raw_features",
)
assembled_df = assembler.transform(customer_features)

scaler = StandardScaler(
    inputCol="raw_features", outputCol="features", withStd=True, withMean=True
)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

kmeans = KMeans(k=4, seed=42).setFeaturesCol("features").setPredictionCol("segment")
model = kmeans.fit(scaled_df)
segmented_df = model.transform(scaled_df)

display(
    segmented_df.groupBy("segment").agg(
        F.count("customer_id").alias("customer_count"),
        F.avg("total_spend").alias("avg_spend"),
        F.avg("visit_count").alias("avg_visits"),
    )
)
