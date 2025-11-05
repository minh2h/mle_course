import os
import pandas as pd
from typing import List, Dict, Any
import logging 
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
import argparse
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_bronze_table(snapshot_date_str, bronze_directory, spark):
    """
    Process raw data and store it in the bronze layer.

    Args:
        snapshot_date_str (str): The snapshot date in "YYYY-MM-DD" format.
        bronze_directory (str): The directory to store the bronze data.
        spark (SparkSession): The Spark session object.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    raw_files = {
        # "loan_daily" : "data/lms_loan_daily.csv",
        "loans" : "data/lms_loan_daily.csv",
        "clickstream" : "data/feature_clickstream.csv",
        "attributes" : "data/features_attributes.csv",
        "financials" : "data/features_financials.csv"
    }
    
    for name, path in raw_files.items():
        df = spark.read.csv(path, header=True, inferSchema=True)
        if "snapshot_date" in df.columns:
            df = df.filter(col('snapshot_date') == snapshot_date_str)
        partition_name = f"bronze_{name}_" + snapshot_date_str.replace("-", "_") + ".csv"
        subdir = bronze_directory["loans"] if name == "loan_daily" else bronze_directory[name]
        os.makedirs(subdir, exist_ok=True)
        filepath = subdir + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print(f"Saved {name} data to {filepath}")
        
    return

