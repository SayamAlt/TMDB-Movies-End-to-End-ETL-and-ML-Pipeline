# Databricks notebook source
# MAGIC %run "../constants/common_variables"

# COMMAND ----------

from pyspark.sql.functions import col, round, desc, when, avg, count

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

sa_access_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='storage-account-access-key')
spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

movies_df = spark.read.format("parquet").load(f"{processed_folder_path}/movies_processed")
movies_df.printSchema()

# COMMAND ----------

movies_df.count()

# COMMAND ----------

movies_df.display(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Descriptive and Exploratory Analysis

# COMMAND ----------

movies_df.select('popularity','vote_average','vote_count','movie_age').describe().show()

# COMMAND ----------

print("Number of unique languages:",movies_df.select('original_language').distinct().count())

# COMMAND ----------

movies_df.groupBy('rating_category').count().sort('count',ascending=False).show()

# COMMAND ----------

movies_df.groupBy('original_language').count().sort('count',ascending=False).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC English is the most spoken language which is not at all surprising.

# COMMAND ----------

movies_df.groupBy('is_remake').count().sort('count',ascending=False).show()

# COMMAND ----------

movies_df.groupBy('title').agg({'normalized_popularity': 'avg'}).withColumnRenamed('avg(normalized_popularity)', 'avg_normalized_popularity').sort('avg_normalized_popularity', ascending=False).show(10)

# COMMAND ----------

movies_df.groupBy('title').agg({'normalized_popularity': 'avg'}).withColumnRenamed('avg(normalized_popularity)', 'avg_normalized_popularity').sort('avg_normalized_popularity', ascending=True).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Gladiator II is the most popular movie while Bootay: Untold is the least popular movie.

# COMMAND ----------

movies_df.createOrReplaceTempView('movies')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from movies;

# COMMAND ----------

movies_df.groupBy('genre_names').count().sort('count', ascending=False).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Drama is the most frequent genre while Thriller is the least frequent genre.

# COMMAND ----------

movies_df.groupBy('original_language').agg({'vote_average': 'avg'}).withColumnRenamed('avg(vote_average)','avg_votes').withColumn('avg_votes',round(col('avg_votes'),3)).sort('avg_votes',ascending=False).show(10)

# COMMAND ----------

# Seasonal trends in releases
movies_df.groupBy('release_month').count().orderBy('count',ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC December has the highest number of movie releases, indicating a possible trend of studios aiming for holiday season audiences.

# COMMAND ----------

# Avg ratings by release year
movies_df.groupBy('release_year').agg({'vote_average': 'avg'}).withColumnRenamed('avg(vote_average)','avg_votes').withColumn('avg_votes',round(col('avg_votes'),3)).sort('avg_votes',ascending=False).show(100)

# COMMAND ----------

# MAGIC %md
# MAGIC The general trend indicates that the average rating of movies has decreased over the years.

# COMMAND ----------

# Overview length trends
movies_df.groupBy('rating_category').agg({'overview_length': 'avg'}).withColumnRenamed('avg(overview_length)','avg_length').sort('avg_length',ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Medium category movies tend to have a longer overview as compared to low and high category movies.

# COMMAND ----------

movies_df.stat.corr('vote_average','normalized_popularity')

# COMMAND ----------

movies_df.stat.corr('vote_count','normalized_popularity')

# COMMAND ----------

# High rating, low popularity movies
movies_df.filter(movies_df.vote_average > 8.0).filter(movies_df.normalized_popularity < 0.5).select('title','vote_average','normalized_popularity').show(10)

# COMMAND ----------

# Movies with most counts
movies_df.groupBy('title').agg({'vote_count':'sum'}).sort(desc('sum(vote_count)')).withColumnRenamed('sum(vote_count)', 'total_votes').show(10)

# COMMAND ----------

movies_df.groupBy('title').agg({'vote_count':'sum'}).sort('sum(vote_count)').withColumnRenamed('sum(vote_count)', 'total_votes').show(10,truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Inception and Interstellar have received the highest votes whereas Archives du changement, Scream Before You Die, and so on have received no votes. 

# COMMAND ----------

movies_df.groupBy('is_remake').agg({'vote_average': 'avg'}).withColumnRenamed('avg(vote_average)', 'avg_rating').show()

# COMMAND ----------

# MAGIC %md
# MAGIC There is a negligible difference in average ratings between remakes and non-remakes.

# COMMAND ----------

# Correlation of movie age with popularity
movies_df.select("movie_age", "normalized_popularity").stat.corr("movie_age", "normalized_popularity")

# COMMAND ----------

# Average rating of movies by age groups
movies_df.withColumn('age_group',when(col('movie_age') < 10, 'New').when(col('movie_age') < 30, 'Modern').otherwise('Classic')).groupBy('age_group').agg(avg(col('vote_average'))).withColumnRenamed('avg(vote_average)', 'avg_votes').sort('avg_votes',ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Classic movies have received the highest ratings followed by modern movies while new movies have received the lowest ratings.

# COMMAND ----------

# Popular movies by age groups
movies_df.withColumn('age_group',when(col('movie_age') < 10, 'New').when(col('movie_age') < 30, 'Modern').otherwise('Classic')).groupBy('age_group').agg(avg(col('popularity'))).withColumnRenamed('avg(popularity)', 'avg_popularity').sort('avg_popularity',ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC On average, the latest movies receive the most popularity while classic movies receive the least popularity.

# COMMAND ----------

movies_df.filter(col('poster_url').isNull()).count()

# COMMAND ----------

movies_df.filter(col('backdrop_url').isNull()).count()

# COMMAND ----------

# Movies released per year
movies_df.groupBy('release_year').count().withColumnRenamed('count','num_movies').orderBy('num_movies',ascending=False).show(50)

# COMMAND ----------

# MAGIC %md
# MAGIC As per the above table, the number of movies released per year has surged over the years.

# COMMAND ----------

# Compare rating categories
movies_df.groupBy('rating_category').agg(avg('vote_average').alias('avg_votes'),avg('vote_count').alias('avg_votes_count'),avg('popularity').alias('avg_popularity')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC - **High-rated movies** have the highest average votes, vote count, and popularity.
# MAGIC - **Low-rated movies** have significantly lower average votes, vote count, and popularity compared to other categories.
# MAGIC - **Medium-rated movies** have moderate average votes, vote count, and popularity, but still higher than low-rated movies.

# COMMAND ----------

# Top performers over decades
movies_df.filter(col('vote_average') > 0.8).groupBy('release_year').count().orderBy('count',ascending=False).show()

# COMMAND ----------

# Check vote_average distribution
movies_df.select('vote_average').summary().show()

# COMMAND ----------

# Null values count
movies_df.select([count(when(col(c).isNull(),c)).alias(c) for c in movies_df.columns]).display()