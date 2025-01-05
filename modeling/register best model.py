# Databricks notebook source
# MAGIC %run "../constants/common_variables"

# COMMAND ----------

# MAGIC %run "../constants/common_functions"

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel
import mlflow

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

sa_access_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='storage-account-access-key')
spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

best_model_pipeline = PipelineModel.load(f"{presentation_folder_path}/best_model_pipeline")
best_model_pipeline

# COMMAND ----------

mlflow.spark.log_model(best_model_pipeline, "best_model_pipeline", registered_model_name="tmdb_movie_vote_predictor")