import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
from functools import reduce

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loans_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # calculate mob (months on book)
    df = df.withColumn("mob", 
            F.months_between(col("snapshot_date"), col("loan_start_date")).cast("int"))
    # get customer at mob
    df = df.filter(col("mob") == mob)

    df = df.withColumn("dpd", 
        F.when(col("overdue_amt") > 0,
            F.datediff(col("snapshot_date"), col("loan_start_date")) - (col("installment_num") * 30)
        ).otherwise(0)
    )

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table 
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df

###############################################################################################################

def process_features_gold_table(snapshot_date_str, silver_dirs, gold_dirs, spark):
    """
    Process features for gold layer - only include customers with active loans
    """
    
    # Essential features (5 base + 3 derived)
    essential_features = [
        'Annual_Income',                    # Financial capacity
        'Outstanding_Debt',                 # Current liabilities
        'Credit_Utilization_Ratio',         # Credit usage pattern
        'Total_EMI_per_month',             # Monthly obligations
        'Num_of_Delayed_Payment',           # Payment behavior
        'Debt_to_Income_Ratio',             # Derived: Debt burden
        'EMI_Burden_Ratio',                 # Derived: Payment pressure
        'Credit_Mix_encoded'                # Derived: Credit quality
    ]
    
    # Step 1: Load ACTIVE customers from loans data
    loans_path = os.path.join(silver_dirs["loans"], 
                            f"silver_loans_{snapshot_date_str.replace('-','_')}.parquet")
    
    if not os.path.exists(loans_path):
        print(f"No loans data found for {snapshot_date_str}, skipping...")
        return None
    
    active_customers = spark.read.parquet(loans_path).select("Customer_ID", "snapshot_date").distinct()
    print(f"Active customers for {snapshot_date_str}: {active_customers.count()}")
    
    # Step 2: Load and merge feature tables
    feature_dfs = []
    
    # Load financial features
    financial_path = os.path.join(silver_dirs["financials"], 
                                f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(financial_path):
        financial_df = spark.read.parquet(financial_path)
        feature_dfs.append(financial_df)
    
    # Load attribute features  
    attribute_path = os.path.join(silver_dirs["attributes"], 
                                f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(attribute_path):
        attribute_df = spark.read.parquet(attribute_path)
        feature_dfs.append(attribute_df)
    
    # Load clickstream features
    clickstream_path = os.path.join(silver_dirs["clickstream"], 
                                  f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(clickstream_path):
        clickstream_df = spark.read.parquet(clickstream_path)
        feature_dfs.append(clickstream_df)
    
    if not feature_dfs:
        print("Error: No feature files found")
        return None
    
    # Merge all features
    features_df = reduce(lambda a, b: a.join(b, ["Customer_ID", "snapshot_date"], "left"), feature_dfs)
    
    # Step 3: FILTER - Only include customers with active loans
    features_df = features_df.join(active_customers, ["Customer_ID", "snapshot_date"], "inner")
    print(f"Features after filtering active customers: {features_df.count()}")
    
    # Step 4: Feature Engineering
    
    # Encode Credit Mix
    if "Credit_Mix" in features_df.columns:
        features_df = features_df.withColumn("Credit_Mix_encoded",
            F.when(col("Credit_Mix") == "Good", 2)
             .when(col("Credit_Mix") == "Standard", 1)
             .otherwise(0)
        )
    
    # Create derived features
    features_df = features_df.withColumn(
        "Debt_to_Income_Ratio",
        F.when(col("Annual_Income") > 0, 
              col("Outstanding_Debt") / col("Annual_Income"))
         .otherwise(None)
    )
    
    features_df = features_df.withColumn(
        "EMI_Burden_Ratio",
        F.when(col("Monthly_Inhand_Salary") > 0,
              col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
         .otherwise(None)
    )
    
    # Step 5: Select only essential features
    available_features = [col for col in essential_features if col in features_df.columns]
    final_cols = ['Customer_ID', 'snapshot_date'] + available_features
    
    features_df = features_df.select(final_cols)
    
    # Step 6: Save results
    gold_path = gold_dirs["feature_store"] + f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    features_df.write.mode("overwrite").parquet(gold_path)
    print(f"Gold feature store saved to: {gold_path}, row count: {features_df.count()}")
    
    return features_df