"""
Weather ETL and Model Training DAG.

This DAG orchestrates the complete ML pipeline:
1. Extract weather data from OpenWeatherMap API
2. Validate data quality (mandatory quality gate)
3. Transform and engineer features
4. Train model with MLflow tracking
5. Register model in MLflow Model Registry
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default arguments for the DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_weather_data(**context):
    """Extract weather data from OpenWeatherMap API."""
    from src.data.extract import WeatherDataExtractor
    
    extractor = WeatherDataExtractor()
    current_path, forecast_path = extractor.extract_and_save(include_forecast=True)
    
    # Push paths to XCom for downstream tasks
    context["ti"].xcom_push(key="current_weather_path", value=str(current_path))
    context["ti"].xcom_push(key="forecast_path", value=str(forecast_path))
    
    return {"current_path": str(current_path), "forecast_path": str(forecast_path)}


def validate_data_quality(**context):
    """
    Mandatory data quality gate.
    
    This task validates the extracted data and fails the DAG if quality checks don't pass.
    """
    from src.data.extract import WeatherDataExtractor
    from src.data.quality_checks import DataQualityChecker, DataQualityError
    from src.config import REPORTS_DIR
    
    # Load historical data
    extractor = WeatherDataExtractor()
    df = extractor.load_historical_data()
    
    if df.empty:
        raise ValueError("No weather data found for validation")
    
    # Run quality checks
    checker = DataQualityChecker()
    
    try:
        report = checker.run_all_checks(df, fail_on_error=True)
        
        # Save quality report
        report_path = REPORTS_DIR / "quality_metrics.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        context["ti"].xcom_push(key="quality_report_path", value=str(report_path))
        context["ti"].xcom_push(key="quality_passed", value=True)
        
        return "quality_passed"
        
    except DataQualityError as e:
        # Save failed report
        report_path = REPORTS_DIR / "quality_metrics.json"
        with open(report_path, "w") as f:
            json.dump(e.report.to_dict(), f, indent=2)
        
        context["ti"].xcom_push(key="quality_report_path", value=str(report_path))
        context["ti"].xcom_push(key="quality_passed", value=False)
        
        # Re-raise to fail the DAG
        raise


def transform_data(**context):
    """Transform raw data and engineer features."""
    from src.data.extract import WeatherDataExtractor
    from src.data.transform import WeatherDataTransformer
    
    # Load historical data
    extractor = WeatherDataExtractor()
    df = extractor.load_historical_data()
    
    # Transform data
    transformer = WeatherDataTransformer()
    df_transformed = transformer.transform(df, create_target=True)
    
    # Save processed data
    data_path = transformer.save_processed_data(df_transformed)
    
    # Generate profile report
    try:
        report_path = transformer.generate_profile_report(df_transformed)
        context["ti"].xcom_push(key="profile_report_path", value=str(report_path))
    except Exception as e:
        print(f"Warning: Could not generate profile report: {e}")
    
    context["ti"].xcom_push(key="processed_data_path", value=str(data_path))
    
    return {"processed_data_path": str(data_path), "n_samples": len(df_transformed)}


def train_model(**context):
    """Train model with MLflow tracking."""
    from src.models.train import WeatherModelTrainer
    from src.config import PROCESSED_DATA_DIR
    import pandas as pd
    
    # Get processed data path from XCom
    ti = context["ti"]
    data_path = ti.xcom_pull(task_ids="transform_data", key="processed_data_path")
    
    if data_path is None:
        # Find latest processed data
        processed_files = list(PROCESSED_DATA_DIR.glob("*.parquet"))
        if not processed_files:
            raise FileNotFoundError("No processed data found")
        data_path = max(processed_files, key=lambda p: p.stat().st_mtime)
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Train model
    trainer = WeatherModelTrainer(experiment_name="weather-prediction-airflow")
    X, y, feature_columns = trainer.prepare_data(df)
    metrics = trainer.train(X, y)
    
    # Save model locally
    model_path = trainer.save_model()
    
    # Push metrics to XCom
    ti.xcom_push(key="model_metrics", value=metrics)
    ti.xcom_push(key="model_path", value=str(model_path))
    
    return metrics


def log_artifacts_to_mlflow(**context):
    """Log additional artifacts to MLflow."""
    import mlflow
    from src.config import settings, REPORTS_DIR
    
    ti = context["ti"]
    
    # Get artifact paths from XCom
    quality_report = ti.xcom_pull(task_ids="validate_data_quality", key="quality_report_path")
    profile_report = ti.xcom_pull(task_ids="transform_data", key="profile_report_path")
    
    # Setup MLflow
    dagshub_config = settings.dagshub
    if dagshub_config.username and dagshub_config.token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_config.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_config.token
        mlflow.set_tracking_uri(dagshub_config.mlflow_tracking_uri)
    
    mlflow.set_experiment("weather-prediction-airflow")
    
    # Log artifacts in a new run
    with mlflow.start_run(run_name="pipeline-artifacts"):
        if quality_report and Path(quality_report).exists():
            mlflow.log_artifact(quality_report, "quality_reports")
        
        if profile_report and Path(profile_report).exists():
            mlflow.log_artifact(profile_report, "data_profiles")
        
        # Log pipeline metadata
        mlflow.log_params({
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "execution_date": str(context["execution_date"]),
        })


def notify_success(**context):
    """Notify on successful pipeline completion."""
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="train_model", key="model_metrics")
    
    print("=" * 50)
    print("🎉 Weather ETL Pipeline Completed Successfully!")
    print("=" * 50)
    print(f"Model Metrics:")
    for key, value in (metrics or {}).items():
        print(f"  {key}: {value:.4f}")
    print("=" * 50)


def notify_failure(context):
    """Notify on pipeline failure."""
    print("=" * 50)
    print("❌ Weather ETL Pipeline Failed!")
    print(f"Task: {context['task_instance'].task_id}")
    print(f"Error: {context.get('exception', 'Unknown error')}")
    print("=" * 50)


# Define the DAG
with DAG(
    dag_id="weather_etl_training_pipeline",
    default_args=default_args,
    description="Weather data ETL and model training pipeline",
    schedule_interval="0 */6 * * *",  # Run every 6 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["weather", "ml", "etl"],
    on_failure_callback=notify_failure,
) as dag:
    
    # Task: Start pipeline
    start = EmptyOperator(task_id="start")
    
    # Task: Extract weather data
    extract = PythonOperator(
        task_id="extract_weather_data",
        python_callable=extract_weather_data,
        provide_context=True,
    )
    
    # Task: Validate data quality (MANDATORY GATE)
    validate = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data_quality,
        provide_context=True,
    )
    
    # Task: Transform data
    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
        provide_context=True,
    )
    
    # Task: Train model
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )
    
    # Task: Log artifacts
    log_artifacts = PythonOperator(
        task_id="log_artifacts_to_mlflow",
        python_callable=log_artifacts_to_mlflow,
        provide_context=True,
    )
    
    # Task: Notify success
    notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    # Task: End pipeline
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )
    
    # Define task dependencies
    # Extract -> Validate (Quality Gate) -> Transform -> Train -> Log -> Notify
    start >> extract >> validate >> transform >> train >> log_artifacts >> notify >> end

