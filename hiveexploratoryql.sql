

-- Creates an external table for exploratory data analysis

CREATE EXTERNAL TABLE paysim_cleaned_external (
  step INT,
  type STRING,
  amount DOUBLE,
  oldbalanceOrg DOUBLE,
  newbalanceOrig DOUBLE,
  oldbalanceDest DOUBLE,
  newbalanceDest DOUBLE,
  isFraud INT,
  isFlaggedFraud INT,
  balanceDiffOrig DOUBLE,
  balanceDiffDest DOUBLE,
  type_encoded INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'gs://drwbucket1/cleaned/';


-- Find the total records in the dataset

SELECT COUNT(*) AS total_records
FROM paysim_cleaned_external;


-- Preview of the dataset (First 8 rows)
SELECT 
        * 
    FROM paysim_cleaned_external
LIMIT 8;


-- Top 10 largest transactions that are fraud
SELECT *
FROM 
  paysim_cleaned_external
WHERE isFraud = 1
  ORDER BY amount DESC
LIMIT 10;


-- Total count of fraduluent transactions

SELECT 
COUNT(*) AS 
  total_count_is_fraud
FROM paysim_cleaned_external
WHERE isFraud = 1;

 
-- Total the count for each transaction type
SELECT 
  type, 
  COUNT(*) AS transaction_count
FROM 
  paysim_cleaned_external
GROUP BY 
  type
ORDER BY 
  transaction_count DESC;