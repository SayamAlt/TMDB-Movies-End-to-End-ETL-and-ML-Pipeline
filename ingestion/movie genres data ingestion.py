# Databricks notebook source
# MAGIC %run "../constants/common_variables"

# COMMAND ----------

import requests, json
import pandas as pd
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, ArrayType, DoubleType, BooleanType

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

bearer_token = dbutils.secrets.get(scope='tmdb-secret-scope',key='api-read-access-token')
bearer_token

# COMMAND ----------

sa_access_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='storage-account-access-key')
sa_access_key

# COMMAND ----------

spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

def get_movie_genres():
    url = "https://api.themoviedb.org/3/genre/movie/list?language=en"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    response = requests.get(url, headers=headers)

    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"Error: {e}")

# COMMAND ----------

genres = get_movie_genres()
genres_df = pd.DataFrame(genres['genres'])
genres_df.head()

# COMMAND ----------

genres_df.shape

# COMMAND ----------

genres_df.info()

# COMMAND ----------

genres_df.describe()

# COMMAND ----------

genres_df.isna().sum()

# COMMAND ----------

genres_schema = StructType([
    StructField('id',IntegerType(),True),
    StructField('name',StringType(),True)
])
genres_schema

# COMMAND ----------

genres_data = spark.createDataFrame(genres_df,schema=genres_schema)
genres_data

# COMMAND ----------

genres_data.printSchema()

# COMMAND ----------

genres_data.count()

# COMMAND ----------

genres_data.show(n=5, truncate=False)

# COMMAND ----------

genres_data.write.format("parquet").mode("overwrite").save(f"{raw_folder_path}/movie_genres")