# Databricks notebook source
from pyspark.sql.functions import expr
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark

# COMMAND ----------

def one_hot_encode(df,column):
    top_categories = df.groupBy(column).count() \
        .orderBy('count', ascending=False) \
        .limit(5).select(column).rdd.flatMap(lambda x: x).collect()
    if None in top_categories:
        top_categories.remove(None)
    
    for category in top_categories:
        df = df.withColumn(category, expr(f"CASE WHEN {column} == '{category}' THEN 1 ELSE 0 END"))
    
    cat_conditions = ",".join([f"'{category}'" for category in top_categories])

    df = df.withColumn(f'other_{column}', expr(f"CASE WHEN {column} NOT IN ({cat_conditions}) THEN 1 ELSE 0 END"))

    return df

# COMMAND ----------

def create_pipeline(indexer,assembler,scaler,model):
    pipeline = Pipeline(stages=[indexer,assembler,scaler,model])
    return pipeline

# COMMAND ----------

def train_and_evaluate_model(pipeline,train_df,test_df):
    with mlflow.start_run(run_name='training'):
        pipeline_model = pipeline.fit(train_df)
        predictions = pipeline_model.transform(test_df)
        evaluator = RegressionEvaluator(labelCol='vote_average')
        rmse = evaluator.setMetricName('rmse').evaluate(predictions)
        mae = evaluator.setMetricName('mae').evaluate(predictions)
        mse = evaluator.setMetricName('mse').evaluate(predictions)
        r2 = evaluator.setMetricName('r2').evaluate(predictions)
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.2f}")
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('R2', r2)
        
        # Log model parameters
        for stage in pipeline.getStages():
            params = stage.extractParamMap()
            for param, value in params.items():
                mlflow.log_param(f"{stage.__class__.__name__}_{param.name}", value)

        model_name = pipeline.getStages()[-1].__class__.__name__
        mlflow.log_param('Model Name', model_name)
        
        mlflow.spark.log_model(pipeline_model, f"trained_{model_name}")

    return pipeline_model

# COMMAND ----------

def optimize_model(pipeline,param_grid,train_df,test_df,labelCol='vote_average',cv_steps=3): 
    evaluator = RegressionEvaluator(labelCol=labelCol)
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=cv_steps)

    with mlflow.start_run(run_name='cross_validation'):
        cv_model = crossval.fit(train_df)
        best_model = cv_model.bestModel
        test_predictions = best_model.transform(test_df)
        mae = evaluator.setMetricName('mae').evaluate(test_predictions)
        mse = evaluator.setMetricName('mse').evaluate(test_predictions)
        rmse = evaluator.setMetricName('rmse').evaluate(test_predictions)
        r2 = evaluator.setMetricName('r2').evaluate(test_predictions)
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_param('Model Name', best_model.__class__.__name__)

        # Log model parameters
        for stage in best_model.stages:
            params = stage.extractParamMap()
            for param, value in params.items():
                mlflow.log_param(f"{stage.__class__.__name__}_{param.name}", value)
                
        mlflow.spark.log_model(best_model, "best_model")
        
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.2f}")
        
    return best_model