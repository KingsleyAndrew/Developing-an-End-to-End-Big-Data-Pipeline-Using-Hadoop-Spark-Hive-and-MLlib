from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import subprocess

#  Create Spark session
spark = SparkSession.builder.appName("PaySimAfrica_ML_Evaluation_Visualization").getOrCreate()

#  Load cleaned data in the Google Cloud bucket
df = spark.read.csv("gs://drwbucket1/cleaned_paysim.csv", header=True, inferSchema=True)

#  Assemble features
feature_cols = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'balanceDiffOrig',
    'balanceDiffDest',
    'type_encoded'
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", "isFraud")

#  Train and Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

#  Random Forest Classifier
rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=50)
rf_model = rf.fit(train_data)
rf_preds = rf_model.transform(test_data)

#  Logistic Regression
lr = LogisticRegression(labelCol="isFraud", featuresCol="features", maxIter=10)
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

#  Define evaluators
binary_evaluator = BinaryClassificationEvaluator(labelCol="isFraud")
multi_evaluator_acc = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="accuracy")
multi_evaluator_precision = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="weightedPrecision")
multi_evaluator_recall = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="weightedRecall")
multi_evaluator_f1 = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")

#  Evaluate Random Forest
print("\n=== Random Forest Metrics ===")
rf_auc = binary_evaluator.evaluate(rf_preds)
print(f"AUC: {rf_auc:.4f}")
print(f"Accuracy: {multi_evaluator_acc.evaluate(rf_preds):.4f}")
print(f"Precision: {multi_evaluator_precision.evaluate(rf_preds):.4f}")
print(f"Recall: {multi_evaluator_recall.evaluate(rf_preds):.4f}")
print(f"F1-Score: {multi_evaluator_f1.evaluate(rf_preds):.4f}")

#  Evaluate Logistic Regression
print("\n=== Logistic Regression Metrics ===")
lr_auc = binary_evaluator.evaluate(lr_preds)
print(f"AUC: {lr_auc:.4f}")
print(f"Accuracy: {multi_evaluator_acc.evaluate(lr_preds):.4f}")
print(f"Precision: {multi_evaluator_precision.evaluate(lr_preds):.4f}")
print(f"Recall: {multi_evaluator_recall.evaluate(lr_preds):.4f}")
print(f"F1-Score: {multi_evaluator_f1.evaluate(lr_preds):.4f}")

# Generate ROC Curve using sklearn for both models

def get_roc_points(preds):
    """Extract true labels and probability scores to compute ROC curve"""
    y_true = np.array(preds.select("isFraud").collect()).flatten()
    y_score = np.array(preds.select("probability").rdd.map(lambda x: float(x[0][1])).collect())
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr

rf_fpr, rf_tpr = get_roc_points(rf_preds)
lr_fpr, lr_tpr = get_roc_points(lr_preds)

plt.figure(figsize=(8,6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.4f})")
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.4f})")
plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)

# Save and upload ROC curve plot
plt.savefig("roc_curve.png")
print("ROC curve saved as roc_curve.png")

# Upload to GCS
subprocess.run(["gsutil", "cp", "roc_curve.png", "gs://drwbucket1/output/roc_curve.png"], check=True)
print("ROC curve uploaded to gs://drwbucket1/output/roc_curve.png")

# Done
spark.stop()
