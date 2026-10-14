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

