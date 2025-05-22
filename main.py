# main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, to_date, avg, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark
spark = SparkSession.builder.appName("FinalAirPollutionProject").getOrCreate()
print("Spark Session Started")

# Load data
df_pd = pd.read_csv(r"C:\Users\Win11\Downloads\AirPollution-Analysis\Dataset\pollution_2000_2023.csv")
df = spark.createDataFrame(df_pd).drop("Unnamed: 0")

# Preprocessing
df = df.withColumn("Date", to_date("Date", "yyyy-MM-dd"))
df = df.withColumn("Year", year("Date")).withColumn("Month", month("Date"))
df = df.dropna().dropDuplicates().repartition(8)
df.cache()
print(f"Rows after cleaning: {df.count()}")

# Average Ozone AQI Over Years
ozone_yearly = df.groupBy("Year").agg(avg("O3 AQI").alias("Avg_O3_AQI")).orderBy("Year")
ozone_pd = ozone_yearly.toPandas()

plt.figure(figsize=(10, 5))
sns.lineplot(data=ozone_pd, x="Year", y="Avg_O3_AQI", marker="o")
plt.title("Average Ozone AQI Over Years")
plt.xlabel("Year")
plt.ylabel("Avg O3 AQI")
plt.grid(True)
plt.tight_layout()
plt.show()

# Average AQI by State
state_avg = df.groupBy("State").agg(
    avg("O3 AQI").alias("Avg_O3_AQI"),
    avg("CO AQI").alias("Avg_CO_AQI"),
    avg("SO2 AQI").alias("Avg_SO2_AQI"),
    avg("NO2 AQI").alias("Avg_NO2_AQI")
).orderBy("Avg_O3_AQI", ascending=False)

state_avg_pd = state_avg.toPandas()
state_avg_pd.plot(kind="bar", x="State", stacked=True, figsize=(14,6))
plt.title("Average AQI by State")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Latest records per state
windowSpec = Window.partitionBy("State").orderBy(col("Date").desc())
latest_per_state = df.withColumn("row_num", row_number().over(windowSpec)).filter("row_num = 1")
latest_per_state.select("State", "Date", "O3 AQI", "CO AQI", "SO2 AQI", "NO2 AQI").show(5)

# Simulated Population Data
population_data = pd.DataFrame({
    "State": ["California", "Texas", "New York"],
    "Population": [39500000, 29000000, 19500000]
})
pop_spark = spark.createDataFrame(population_data)
df_joined = df.join(pop_spark, on="State", how="left")
df_joined.select("State", "Population", "O3 AQI", "Date").show(5)

# ML Model
features = ["O3 Mean", "O3 1st Max Value", "O3 1st Max Hour"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_ml = assembler.transform(df)
ml_data = df_ml.select("features", "O3 AQI")
train, test = ml_data.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="O3 AQI")
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
evaluator = RegressionEvaluator(labelCol="O3 AQI", predictionCol="prediction", metricName="rmse")
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

cv_model = cv.fit(train)
results = cv_model.transform(test)

rmse = evaluator.evaluate(results)
r2 = RegressionEvaluator(labelCol="O3 AQI", predictionCol="prediction", metricName="r2").evaluate(results)
print(f"Final Model Performance:\n - RMSE: {rmse:.2f}\n - R²: {r2:.2f}")

# Visualization
results_pd = results.select("O3 AQI", "prediction").toPandas()
plt.figure(figsize=(8, 6))
sns.scatterplot(x="O3 AQI", y="prediction", data=results_pd, color='blue', edgecolor='w', s=60)
plt.plot([results_pd["O3 AQI"].min(), results_pd["O3 AQI"].max()],
         [results_pd["O3 AQI"].min(), results_pd["O3 AQI"].max()],
         'r--', linewidth=2)
plt.xlabel("Actual O3 AQI")
plt.ylabel("Predicted O3 AQI")
plt.title("Scatter Plot of Actual vs Predicted O3 AQI")
plt.grid(True)
plt.tight_layout()
plt.show()
