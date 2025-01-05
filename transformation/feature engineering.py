# Databricks notebook source
# MAGIC %run "../constants/common_variables"

# COMMAND ----------

from pyspark.sql.functions import col, expr, year, month, dayofmonth, when, round, length, lit, concat, datediff, current_date
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

sa_access_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='storage-account-access-key')
spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

movies_df = spark.read.format("parquet").load(f"{processed_folder_path}/all_movies")
movies_df.printSchema()

# COMMAND ----------

movies_df.count()

# COMMAND ----------

movies_df.show(10)

# COMMAND ----------

genres_df = spark.read.format("parquet").load(f"{raw_folder_path}/movie_genres")
genres_df.printSchema()

# COMMAND ----------

genre_id_name_map = {row.id: row.name for row in genres_df.collect()}
genre_id_name_map

# COMMAND ----------

def get_genre_names(genre_ids):
    return [genre_id_name_map.get(genre_id) for genre_id in genre_ids]

# COMMAND ----------

get_genre_names_udf = udf(get_genre_names, ArrayType(StringType()))

# COMMAND ----------

movies_df = movies_df.withColumn('genre_names', get_genre_names_udf(movies_df['genre_ids']))
display(movies_df.select('genre_ids', 'genre_names'))

# COMMAND ----------

movies_df = movies_df.withColumn(
    'genre_names',
    expr(
        "CASE WHEN size(genre_names) > 1 THEN concat_ws(', ', slice(genre_names, 1, size(genre_names)-1)) || ' and ' || element_at(genre_names, -1) ELSE element_at(genre_names, 1) END"
    )
)
movies_df.select('genre_names').show(10)

# COMMAND ----------

movies_df.printSchema()

# COMMAND ----------

movies_df = movies_df.withColumn('release_year',year(col('release_date'))) \
                     .withColumn('release_month',month(col('release_date'))) \
                     .withColumn('release_day',dayofmonth(col('release_date')))

movies_df.show(10)

# COMMAND ----------

movies_df.select('vote_average').describe().show()

# COMMAND ----------

movies_df = movies_df.withColumn('rating_category',when(col('vote_average') >= 8, 'High').when(col('vote_average') >= 5, 'Medium').otherwise('Low'))
movies_df.select('rating_category').show(10)

# COMMAND ----------

max_popularity = movies_df.agg({'popularity': 'max'}).collect()[0][0]
max_popularity
movies_df = movies_df.withColumn('normalized_popularity', col('popularity') / max_popularity)
movies_df.show(10)

# COMMAND ----------

movies_df = movies_df.withColumn('normalized_popularity',round(col('normalized_popularity'),3))
movies_df.select('normalized_popularity').show(10)

# COMMAND ----------

movies_df = movies_df.withColumn('overview_length', length(col('overview')))
movies_df.select('overview_length').show(10)

# COMMAND ----------

base_url = "https://image.tmdb.org/t/p/w500"
movies_df = movies_df.withColumn("poster_url", concat(lit(base_url), col("poster_path"))) \
                     .withColumn("backdrop_url", concat(lit(base_url), col("backdrop_path")))

# COMMAND ----------

movies_df = movies_df.withColumn("is_remake", when(col("title") != col("original_title"), 1).otherwise(0))
movies_df.select('is_remake').show(10)

# COMMAND ----------

movies_df = movies_df.withColumn('is_remake', (col('is_remake').cast('boolean')))
movies_df.printSchema()

# COMMAND ----------

movies_df = movies_df.withColumn('movie_age', datediff(current_date(), col('release_date')) / 365)
movies_df = movies_df.withColumn('movie_age',round(col('movie_age'),2))
movies_df.select('movie_age').show(10)

# COMMAND ----------

movies_df.write.format("parquet").mode("overwrite").save(f"{processed_folder_path}/movies_processed")