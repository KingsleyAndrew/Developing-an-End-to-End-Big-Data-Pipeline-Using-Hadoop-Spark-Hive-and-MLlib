from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

# Create Spark session
spark = SparkSession.builder.appName("TPSAcleaned").getOrCreate()

# Load data from GCS
input_path = "gs://drwbucket1/TPSAfrica.csv"
df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

# Basic feature engineering
df = df.withColumn("balanceDiffOrig", col("oldbalanceOrg") - col("newbalanceOrig"))
df = df.withColumn("balanceDiffDest", col("newbalanceDest") - col("oldbalanceDest"))

# Label encode 'type' (handleInvalid added for safety)
indexer = StringIndexer(inputCol="type", outputCol="type_encoded", handleInvalid="keep")
df = indexer.fit(df).transform(df)

# Drop duplicates
df = df.dropDuplicates()

# Coalesce to 1 partition (single CSV)
df = df.coalesce(1)

# Save cleaned data back to GCS
output_path = "gs://drwbucket1/cleaned/PSAcleaned"
df.write.option("header", "true").mode("overwrite").csv(output_path)

print(f" Done! Cleaned data saved to: {output_path}")
