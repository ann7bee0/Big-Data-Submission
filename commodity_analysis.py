import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Set environment variables for Spark and Java
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Commodity Trade Analysis and Model Training") \
    .getOrCreate()

# Load dataset from HDFS
data_path = "hdfs://namenode:8020/user/data2.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Validate data loading
print("Schema of loaded data:")
df.printSchema()

# Data Cleaning
if "weight_kg" in df.columns:
    df = df.withColumn("weight_kg", col("weight_kg").cast("double"))
    mean_weight = df.select(avg("weight_kg")).first()[0] or 0
    df = df.fillna({"weight_kg": mean_weight})
    df = df.withColumn("weight_ton", round(col("weight_kg") / 1000, 2))
else:
    raise ValueError("Column 'weight_kg' is missing.")

if "trade_usd" in df.columns:
    df = df.withColumn("trade_usd", col("trade_usd").cast("double"))
    mean_trade = df.select(avg("trade_usd")).first()[0] or 0
    df = df.fillna({"trade_usd": mean_trade})
else:
    raise ValueError("Column 'trade_usd' is missing.")

# Feature Engineering
assembler = VectorAssembler(inputCols=["weight_ton"], outputCol="features")
df = assembler.transform(df)

# Prepare data for training
df = df.select("features", col("trade_usd").alias("label"))
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Drop nulls
train_df = train_df.na.drop(subset=["features", "label"])
test_df = test_df.na.drop(subset=["features", "label"])

# Train Linear Regression Model
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1, elasticNetParam=0.8)
try:
    model = lr.fit(train_df)
    print("Model trained successfully!")
except Exception as e:
    print(f"Error during model training: {e}")
    spark.stop()
    exit(1)

# Evaluate the Model
predictions = model.transform(test_df)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
mse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse").evaluate(predictions)

print(f"Model R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Save the trained model
model_path = "hdfs://namenode:8020/user/linear_regression_model"
model.write().overwrite().save(model_path)
print(f"Trained model saved to {model_path}")

# Stop SparkSession
spark.stop()
