# Databricks notebook source
# MAGIC %run "../constants/common_variables"

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

sa_access_key = dbutils.secrets.get(scope="tmdb-secret-scope", key="storage-account-access-key")
sa_access_key

# COMMAND ----------

spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

present_movies = spark.read.format("parquet").load(f"{raw_folder_path}/present_movies")
display(present_movies)

# COMMAND ----------

upcoming_movies = spark.read.format("parquet").load(f"{raw_folder_path}/upcoming_movies")
display(upcoming_movies)

# COMMAND ----------

top_rated_movies = spark.read.format("parquet").load(f"{raw_folder_path}/top_rated_movies")
display(top_rated_movies)

# COMMAND ----------

popular_movies = spark.read.format("parquet").load(f"{raw_folder_path}/popular_movies")
display(popular_movies)

# COMMAND ----------

popular_movies.count(), top_rated_movies.count(), upcoming_movies.count(), present_movies.count()

# COMMAND ----------

combined_movies = popular_movies.union(top_rated_movies).union(upcoming_movies).union(present_movies).dropDuplicates()
display(combined_movies)

# COMMAND ----------

combined_movies.count()

# COMMAND ----------

combined_movies.printSchema()

# COMMAND ----------

combined_movies = combined_movies.drop('adult','video') # Since adult and video columns contain only one value i.e. False, so we can drop them.

# COMMAND ----------

combined_movies.printSchema()

# COMMAND ----------

combined_movies.write.format("parquet").mode("overwrite").save(f"{processed_folder_path}/all_movies")