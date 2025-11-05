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
            F.datediff(col("snapshot_date"), col("loan_start_date")) - col(("installment_num") * 30)
            .otherwise(0)
        ) 

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(snapshot_date_str, silver_directory, gold_directory, spark):
    
    # columns to remove. eg. 'fe_1' to 'fe_20'
    cols_to_remove = [
        'label', 'target', 'Name', 'SSN', 'Occupation',
        'Payment_of_Min_Amount', 'Payment_Behaviour', 'fe_sum',
        'fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5', 'fe_6', 'fe_7', 'fe_8', 'fe_9',
        'fe_10', 'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15', 'fe_16', 'fe_17',
        'fe_18', 'fe_19', 'fe_20', 'Changed_Credit_Limit', 'Credit_History_Age',
        'Amount_invested_monthly', 'Type_of_Loan'
    ]
    
    # features selected
    feature_cols = [
        'Annual_Income', 
        'Monthly_Inhand_Salary', 
        'Num_Bank_Accounts',
        'Interest_Rate',
        'Num_of_Loan',
        'Outstanding_Debt',
        'Credit_Utilization_Ratio',
        'Total_EMI_per_month',
        'Credit_Mix_encoded', 
    ]

    numeric_cols = [
        'Monthly_Balance', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Outstanding_Debt', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
        'Total_EMI_per_month', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment'
    ]
    
    # initialize feature dataframe
    feature_dataframe = []
    
    # LOAD SILVER TABLES
    
    # clickstream
    clickstream_path = os.path.join(silver_directory["clickstream"], 
                                     f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(clickstream_path):
        clickstream_df = spark.read.parquet(clickstream_path)
        feature_dataframe.append(clickstream_df)

    # attributes
    attributes_path = os.path.join(silver_directory["attributes"], 
                                   f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(attributes_path):
        attributes_df = spark.read.parquet(attributes_path)
        feature_dataframe.append(attributes_df)
    
    # financials
    financials_path = os.path.join(silver_directory["financials"], 
                                   f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(financials_path):
        financials_df = spark.read.parquet(financials_path)
        feature_dataframe.append(financials_df)

    if not feature_dataframe:
        print("No feature files found.")
        return None 

    # MERGE DATAFRAMES
    feature_df = reduce(lambda df1, df2: df1.join(df2, ["Customer_ID", "snapshot_date"], "outer"), feature_dataframe)

    if feature_df.count() == 0:
        print("Merged feature dataframe is empty.")
        return None
    
    # encode "Occupation" as integer
    if "Occupation" in feature_df.columns:
        occupation_dict = feature_df.select("Occupation").distinct().na.drop().rdd.map(lambda row: row[0]).zipWithIndex().collectAsMap()

        bc_occupation_dict = spark.sparkContext.broadcast(occupation_dict)
        def encode_occupation(occupation):
            return bc_occupation_dict.value.get(occupation)
        
        encode_occupation_udf = F.udf(encode_occupation, IntegerType())
        feature_df = feature_df.withColumn("Occupation_encoded", encode_occupation_udf(col("Occupation")))
        
    # save gold feature store table
    gold_path = gold_directory["feature_store"] + f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    feature_df.write.mode("overwrite").parquet(gold_path)
    print(f'saved to: {gold_path}, row count: {feature_df.count()}')

    # final check to ensure no label and unwanted columns left
    final_cols = [col for col in feature_df.columns if not col.lower().startswith(('label', 'target'))
                  and col not in cols_to_remove]
    feature_df = feature_df.select(*final_cols)
    
    for col_name in numeric_cols:
        if col_name in feature_df.columns:
            feature_df = feature_df.withColumn(col_name, 
                        F.when(col(col_name).isNotNull(), col(col_name))
                        .otherwise(F.lit(None).cast(FloatType())))
    
    # Credit_Mix to encoded
    if 'Credit_Mix' in feature_df.columns:
        credit_mix_dict = {'Bad':0,'Standard': 1, 'Good': 2}
        feature_df = feature_df.withColumn('Credit_Mix_encoded',
                        F.when(col('Credit_Mix') == 'Bad', credit_mix_dict['Bad'])
                        .when(col('Credit_Mix') == 'Standard', credit_mix_dict['Standard'])
                        .when(col('Credit_Mix') == 'Good', credit_mix_dict['Good'])
                    )
    final_cols = [col for col in feature_df.columns if col in feature_cols or col in ['Customer_ID', 'snapshot_date']]
        
    ## Feature Engineering - creating derived features that can be more predictive
    feature_df = feature_df.withColumn('Debt_to_Income_Ratio',
                    F.when(col('Outstanding_Debt').isNotNull() & col('Annual_Income').isNotNull() & (col('Annual_Income') > 0),
                            col('Outstanding_Debt') / col('Annual_Income'))
                    .otherwise(None)
                )
    
    feature_df = feature_df.withColumn('EMI_to_Income_Ratio',
                    F.when(col('Monthly_Inhand_Salary') > 0,
                            col('Total_EMI_per_month') / col('Monthly_Inhand_Salary'))
                    .otherwise(None)
                )
    
    # To maintain a healthy credit score, lenders prefer a ratio of 30% or less, 
    # Normalized credit usage pattern
    feature_df = feature_df.withColumn('Credit_Utilization_Trend',
                    (col('Credit_Utilization_Ratio') - F.lit(0.3)) / F.lit(0.7)
                )   
    
    feature_df = feature_df.withColumn('Deliquency_Score',
                    (col('Num_of_Delayed_Payment') * 0.4) + (col('Delay_from_due_date') * 0.6)
                    )
    
    feature_df = feature_df.select(*final_cols, 'Debt_to_Income_Ratio', 'EMI_to_Income_Ratio', 'Credit_Utilization_Trend', 'Deliquency_Score')
    
    # important features to consider; derived features will also be added
    important_features = [
        'Annual_Income', 
        'Outstanding_Debt',
        'Credit_Utilization_Ratio',
        'Total_EMI_per_month',
        'Num_of_Delayed_Payment',
        'Debt_to_Income_Ratio',      # derived
        'EMI_to_Income_Ratio',    # derived 
        'Credit_Utilization_Trend',     # derived
        'Deliquency_Score'          # derived
    ]
    
    # join with other silver tables 
    feature_dfs = []
    for each in ["financials", "attributes", "clickstream"]:
        path = os.path.join(silver_directory[each], 
                            f"silver_{each}_{snapshot_date_str.replace('-','_')}.parquet")
        if os.path.exists(path):
            feature_dfs.append(spark.read.parquet(path))
            
    if not feature_dfs:
        print("No feature files found.")
        return None
    
    features_df = reduce(lambda df1, df2: df1.join(df2, ["Customer_ID", "snapshot_date"], "outer"), feature_dfs)
    
    ######################################################################################################################
    # Add this debug code before the problematic line (~217)
    print("=== DEBUG: Available columns in feature_df ===")
    print(feature_df.columns)
    print("=============================================")

    # Check if specific columns exist
    important_columns = ['Outstanding_Debt', 'Annual_Income', 'Monthly_Inhand_Salary']
    for col_name in important_columns:
        if col_name in feature_df.columns:
            print(f"✓ Column '{col_name}' exists")
        else:
            print(f"✗ Column '{col_name}' NOT FOUND")
    ######################################################################################################################
    
    features_df = features_df.withColumn('Debt_to_Income_Ratio',
                    F.when(col('Outstanding_Debt').isNotNull() & col('Annual_Income').isNotNull() & (col('Annual_Income') > 0),
                            col('Outstanding_Debt') / col('Annual_Income'))
                    .otherwise(None)
                )
    
    features_df = features_df.withColumn('EMI_to_Income_Ratio',
                    F.when(col('Monthly_Inhand_Salary') > 0,
                            col('Total_EMI_per_month') / col('Monthly_Inhand_Salary'))
                    .otherwise(None)
                )

    features_df = features_df.withColumn('Credit_Mix_encoded',
                        F.when(col('Credit_Mix') == 'Bad', credit_mix_dict['Bad'])
                        .when(col('Credit_Mix') == 'Standard', credit_mix_dict['Standard'])
                        .when(col('Credit_Mix') == 'Good', credit_mix_dict['Good'])
                    )
    
    
    
    # select only important features
    features_df = features_df.select("Customer_ID", "snapshot_date", *important_features)
    
    gold_path = gold_directory["feature_store"] + f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    features_df.write.mode("overwrite").parquet(gold_path)
    print(f'saved to: {gold_path}, row count: {features_df.count()}')
        
    return features_df