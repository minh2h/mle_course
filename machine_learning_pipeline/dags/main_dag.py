from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
project_root = '/app'
if project_root not in sys.path:
    sys.path.append(project_root)

# Default arguments for the DAG
default_args = {
    'owner': 'ml-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'end_to_end_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for model training using existing datamart data',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['ml', 'model-training'],
)

# Task 1: Train Model
train_model_task = BashOperator(
    task_id='model_train',
    bash_command='cd /app && python utils/model_train.py',
    dag=dag,
)

# Task 2: Make Predictions
make_predictions_task = BashOperator(
    task_id='make_predictions',
    bash_command='cd /app && python utils/predict.py',
    dag=dag,
)

# Task 3: Monitor Model
monitor_model_task = BashOperator(
    task_id='monitor_model',
    # bash_command='cd /app && python utils/monitor.py --current-data datamart/predictions/predictions_training.csv --reference-data datamart/gold/training/feature_store.parquet --output-dir datamart/gold/monitoring_report --enable-oot-analysis --test-data datamart/gold/training/label_store.parquet --oot1-data datamart/gold/OOT1/label_store.parquet --oot2-data datamart/gold/OOT2/label_store.parquet --predictions-dir datamart/predictions',
    bash_command=(
        'cd /app && '
        'python utils/monitor.py '
        '--current-data datamart/predictions/predictions_training.csv '
        '--reference-data datamart/gold/training/feature_store.parquet '
        '--output-dir datamart/gold/monitoring_report '
        '--enable-oot-analysis '
        '--test-data datamart/gold/training/label_store.parquet '
        '--oot1-data datamart/gold/OOT1/label_store.parquet '
        '--oot2-data datamart/gold/OOT2/label_store.parquet '
        '--predictions-dir datamart/predictions'
    ),
    dag=dag,
)

# Define task dependencies
# Direct pipeline: Train -> Predict -> Monitor using existing datamart data
train_model_task >> make_predictions_task >> monitor_model_task