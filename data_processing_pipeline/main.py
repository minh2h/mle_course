import pandas as pd
import os
import pyspark
from utils.data_processing_bronze import process_bronze_table
from utils.data_processing_silver import process_silver_loans, process_silver_clickstream, process_silver_attributes, process_silver_financials 
from utils.data_processing_gold import process_features_gold_table, process_labels_gold_table
from pyspark.sql import SparkSession
from utils.date_utils import generate_first_of_month_dates
import glob
import shutil

# load raw datasets
loans = pd.read_csv("data/lms_loan_daily.csv")
clickstream = pd.read_csv("data/feature_clickstream.csv")
attributes = pd.read_csv("data/features_attributes.csv")
financials = pd.read_csv("data/features_financials.csv")

os.makedirs("datamart/bronze", exist_ok=True)

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# set up config
snapshot_date_str = "2023-01-01"
start_date_str = "2023-01-01"
end_date_str = "2024-12-01" # MAY NEED TO CHANGE

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

# # bronze layer processing
# os.makedirs("datamart/bronze/loans", exist_ok=True)
# os.makedirs("datamart/bronze/clickstream", exist_ok=True)
# os.makedirs("datamart/bronze/attributes", exist_ok=True)
# os.makedirs("datamart/bronze/financials", exist_ok=True)

# create bronze datalake
bronze_directory = {
    "loans": "datamart/bronze/loans/",
    "clickstream": "datamart/bronze/clickstream/",
    "attributes": "datamart/bronze/attributes/",
    "financials": "datamart/bronze/financials/"
}

for each in bronze_directory.values():
    os.makedirs(each, exist_ok=True)

silver_directory = {
    "loans": "datamart/silver/loans/",
    "clickstream": "datamart/silver/clickstream/",
    "attributes": "datamart/silver/attributes/",
    "financials": "datamart/silver/financials/"
}

for each in silver_directory.values():
    os.makedirs(each, exist_ok=True)
    
gold_directory = {
    "feature_store": "datamart/gold/feature_store/",
    "label_store": "datamart/gold/label_store/"
}

for each in gold_directory.values():
    os.makedirs(each, exist_ok=True)

# bronze layer processing
for date_str in dates_str_lst:
    process_bronze_table(date_str, bronze_directory, spark)

def save_single_parquet(df, tmp_directory, final_path):
    df.repartition(1).write.mode("overwrite").parquet(tmp_directory)
    part_file = glob.glob(os.path.join(tmp_directory, "part-*.parquet"))[0]
    shutil.move(part_file, final_path)
    shutil.rmtree(tmp_directory)

# silver layer processing
for date_str in dates_str_lst:
    from utils.data_processing_silver import process_silver_loans, process_silver_clickstream, process_silver_attributes, process_silver_financials
    process_silver_loans(date_str, bronze_directory["loans"], silver_directory["loans"], spark)
    process_silver_clickstream(date_str, bronze_directory["clickstream"], silver_directory["clickstream"], spark)
    process_silver_attributes(date_str, bronze_directory["attributes"], silver_directory["attributes"], spark)
    process_silver_financials(date_str, bronze_directory["financials"], silver_directory["financials"], spark)

# gold layer processing - feature store
for date_str in dates_str_lst:
    feature_df = process_features_gold_table(date_str, silver_directory, gold_directory, spark)
    
    if feature_df is not None:
        tmp_directory = gold_directory["feature_store"] + f"gold_feature_store_{date_str.replace('-','_')}"
        feature_file = gold_directory["feature_store"] + f"gold_feature_store_{date_str.replace('-','_')}.parquet"
        save_single_parquet(feature_df, tmp_directory, feature_file)
        print(f"Saved single parquet file to {feature_file}")
    
# gold layer processing - label store
for date_str in dates_str_lst:
    label_df = process_labels_gold_table(date_str, 
                                         silver_directory["loans"], 
                                         gold_directory["label_store"], 
                                         spark, 
                                         dpd=30, 
                                         mob=6)
    if label_df is not None:
        tmp_directory = gold_directory["label_store"] + f"gold_label_store_{date_str.replace('-','_')}"
        label_file = gold_directory["label_store"] + f"gold_label_store_{date_str.replace('-','_')}.parquet"
        save_single_parquet(label_df, tmp_directory, label_file)
        print(f"Saved single parquet file to {label_file}")
        
# final check
folder_path = gold_directory["feature_store"]
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print(f"Final feature store row count: {df.count()}")
df.show()