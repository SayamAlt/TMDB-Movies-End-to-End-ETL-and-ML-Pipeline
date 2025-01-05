# Databricks notebook source
# MAGIC %run "../constants/common_variables"

# COMMAND ----------

import requests, json
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, ArrayType

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

api_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='api-key')
bearer_token = dbutils.secrets.get(scope='tmdb-secret-scope',key='api-read-access-token')
sa_access_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='storage-account-access-key')

# COMMAND ----------

spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

def get_top_rated_movies(page):
    url = f"https://api.themoviedb.org/3/movie/top_rated?language=en-US&page={page}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return json.loads(response.text)

# COMMAND ----------

data = get_top_rated_movies(1)
num_pages = data["total_pages"]

if num_pages > 500:
    num_pages = 500 # As the API allows a maximum of 500 pages
    
movies = []

for movie in data["results"]:
    movies.append(movie)

print("Page 1 processed.")

for page in range(2, num_pages + 1):
    print(f"Processing page {page}...")
    top_rated_movies = get_top_rated_movies(page)
    res = top_rated_movies["results"]

    for movie in res:
        movies.append(movie)
    
    print(f"Page {page} processed.")

print(f"Total number of movies: {len(movies)}")

# COMMAND ----------

movies_df = pd.DataFrame(movies)
movies_df.head()

# COMMAND ----------

movies_df.shape

# COMMAND ----------

movies_df.info()

# COMMAND ----------

movies_df.describe()

# COMMAND ----------

movies_df.isnull().sum()

# COMMAND ----------

movies_df.backdrop_path.fillna('/no_image.jpg', inplace=True)
movies_df.poster_path.fillna('/no_image.jpg', inplace=True)

# COMMAND ----------

movies_df.isna().sum()

# COMMAND ----------

movies_schema = StructType([
    StructField('adult', BooleanType(), True),
    StructField('backdrop_path', StringType(), True),
    StructField('genre_ids', ArrayType(IntegerType()), True),
    StructField('id', IntegerType(), True),
    StructField('original_language', StringType(), True),
    StructField('original_title', StringType(), True),
    StructField('overview', StringType(), True),
    StructField('popularity', DoubleType(), True),
    StructField('poster_path', StringType(), True),
    StructField('release_date', StringType(), True),
    StructField('title', StringType(), True),
    StructField('video', BooleanType(), True),
    StructField('vote_average', DoubleType(), True),
    StructField('vote_count', IntegerType(), True)
])

movies_data = spark.createDataFrame(movies_df,schema=movies_schema)
movies_data.printSchema()

# COMMAND ----------

movies_data.show(5)

# COMMAND ----------

movies_data.count()

# COMMAND ----------

movies_data.write.format("parquet").mode("overwrite").save(f"{raw_folder_path}/top_rated_movies")