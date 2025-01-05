# Databricks notebook source
# MAGIC %md
# MAGIC ## Executing dependent notebooks

# COMMAND ----------

# MAGIC %run "../constants/common_variables"

# COMMAND ----------

# MAGIC %run "../constants/common_functions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing the required libraries

# COMMAND ----------

from pyspark.sql.functions import col, when, expr, count
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GBTRegressor, RandomForestRegressor, FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

# MAGIC %md
# MAGIC ## Authenticating ADLS Storage Account

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

sa_access_key = dbutils.secrets.get(scope='tmdb-secret-scope',key='storage-account-access-key')
spark.conf.set("fs.azure.account.key.tmdbmoviesdl.dfs.core.windows.net", sa_access_key)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the processed movies data

# COMMAND ----------

movies_df = spark.read.format("parquet").load(f"{processed_folder_path}/movies_processed")
movies_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration

# COMMAND ----------

movies_df.count()

# COMMAND ----------

movies_df.show(10,truncate=False)

# COMMAND ----------

display(movies_df)

# COMMAND ----------

# Cast 'is_remake' column to integer type
movies_df = movies_df.withColumn('is_remake',when(col('is_remake') == False,0).otherwise(1))
movies_df = movies_df.withColumn('is_remake',col('is_remake').cast('integer'))
movies_df.select('is_remake').show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### One-Hot Encoding

# COMMAND ----------

movies_df.groupBy('rating_category').count().show()

# COMMAND ----------

movies_df = one_hot_encode(movies_df,'genre_names')
movies_df = one_hot_encode(movies_df,'original_language')
display(movies_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Removing Missing Values

# COMMAND ----------

movies_df.select([count(when(col(c).isNull(),c)).alias(c) for c in movies_df.columns]).display()

# COMMAND ----------

movies_df = movies_df.dropna(subset=['release_year', 'release_month', 'release_day', 'movie_age'])

# COMMAND ----------

movies_df.select([count(when(col(c).isNull(),c)).alias(c) for c in movies_df.columns]).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split

# COMMAND ----------

train_df, test_df = movies_df.randomSplit([0.7,0.3],seed=42)

# COMMAND ----------

train_df.count(), test_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Evaluation

# COMMAND ----------

# Define distinct stages of the pipeline

indexer = StringIndexer(inputCol='rating_category',outputCol='rating_category_index',handleInvalid='keep')

assembler = VectorAssembler(inputCols=[
    'vote_count',
    'release_year',
    'release_month',
    'release_day',
    'rating_category_index',
    'normalized_popularity',
    'overview_length',
    'is_remake',
    'movie_age',
    'Drama',
    'Comedy',
    'Documentary',
    'Drama and Romance',
    'other_genre_names',
    'en',
    'fr',
    'ja',
    'es',
    'it',
    'other_original_language'
],outputCol='features')

scaler = StandardScaler(inputCol='features',outputCol='scaled_features')

# COMMAND ----------

lr_model = LinearRegression(featuresCol='scaled_features',labelCol='vote_average')
lr_pipeline = create_pipeline(indexer,assembler,scaler,lr_model)
lr_pipeline

# COMMAND ----------

trained_lr_pipeline = train_and_evaluate_model(lr_pipeline,train_df,test_df)
trained_lr_pipeline

# COMMAND ----------

dt_model = DecisionTreeRegressor(featuresCol='scaled_features',labelCol='vote_average')
dt_pipeline = create_pipeline(indexer,assembler,scaler,dt_model)
dt_pipeline

# COMMAND ----------

trained_dt_pipeline = train_and_evaluate_model(dt_pipeline,train_df,test_df)
trained_dt_pipeline

# COMMAND ----------

gbt_model = GBTRegressor(featuresCol='scaled_features',labelCol='vote_average')
gbt_pipeline = create_pipeline(indexer,assembler,scaler,gbt_model)
gbt_pipeline

# COMMAND ----------

trained_gbt_pipeline = train_and_evaluate_model(gbt_pipeline,train_df,test_df)
trained_gbt_pipeline

# COMMAND ----------

rf_model = RandomForestRegressor(featuresCol='scaled_features',labelCol='vote_average')
rf_pipeline = create_pipeline(indexer,assembler,scaler,rf_model)
rf_pipeline

# COMMAND ----------

trained_rf_pipeline = train_and_evaluate_model(rf_pipeline,train_df,test_df)
trained_rf_pipeline

# COMMAND ----------

fm_model = FMRegressor(featuresCol='scaled_features',labelCol='vote_average')
fm_pipeline = create_pipeline(indexer,assembler,scaler,fm_model)
fm_pipeline

# COMMAND ----------

trained_fm_pipeline = train_and_evaluate_model(fm_pipeline,train_df,test_df)
trained_fm_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC The GBT Regressor model has achieved the best results with an outstanding R2 score of more than 96%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ### FM Regressor Pipeline

# COMMAND ----------

paramGrid = (ParamGridBuilder()
             .addGrid(fm_model.stepSize, [0.001, 0.01, 0.1])
             .addGrid(fm_model.factorSize, [4, 8, 12])
             .build())
             
tuned_fm_pipeline = optimize_model(pipeline=fm_pipeline,param_grid=paramGrid,train_df=train_df,test_df=test_df,labelCol='vote_average')
tuned_fm_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear Regression Pipeline

# COMMAND ----------

paramGrid_lr = (ParamGridBuilder()
                .addGrid(lr_model.regParam, [0.01, 0.1, 1.0])
                .addGrid(lr_model.elasticNetParam, [0.0, 0.5, 1.0])
                .build())

tuned_lr_pipeline = optimize_model(pipeline=lr_pipeline,param_grid=paramGrid_lr,train_df=train_df,test_df=test_df,labelCol='vote_average')
tuned_lr_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree Pipeline

# COMMAND ----------

paramGrid_dt = (ParamGridBuilder()
                .addGrid(dt_model.maxDepth, [5, 10, 15])
                .addGrid(dt_model.minInstancesPerNode, [1, 2, 4])
                .build())

tuned_dt_pipeline = optimize_model(pipeline=dt_pipeline,param_grid=paramGrid_dt,train_df=train_df,test_df=test_df,labelCol='vote_average')
tuned_dt_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Pipeline

# COMMAND ----------

paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf_model.numTrees, [5,10])
                .build())

tuned_rf_pipeline = optimize_model(pipeline=rf_pipeline, param_grid=paramGrid_rf, train_df=train_df, test_df=test_df, labelCol='vote_average', cv_steps=2)
tuned_rf_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gradient Boosted Tree Pipeline

# COMMAND ----------

paramGrid_gbt = (ParamGridBuilder()
                 .addGrid(gbt_model.stepSize, [0.001, 0.01, 0.1])
                 .build())

tuned_gbt_pipeline = optimize_model(pipeline=gbt_pipeline, param_grid=paramGrid_gbt, train_df=train_df, test_df=test_df, labelCol='vote_average', cv_steps=2)
tuned_gbt_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC After performing hyperparameter tuning, the GBT Regressor model has achieved the best results with an astonishing R2 score of more than 97%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving the best model

# COMMAND ----------

tuned_gbt_pipeline.stages[-1]

# COMMAND ----------

tuned_gbt_pipeline.write().overwrite().save(f"{presentation_folder_path}/best_model_pipeline")