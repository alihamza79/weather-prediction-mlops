"""Model training module with MLflow experiment tracking."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from xgboost import XGBRegressor

from src.config import settings, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, PROJECT_ROOT


class WeatherModelTrainer:
    """Train weather prediction models with MLflow tracking."""

    def __init__(self, experiment_name: str = "weather-prediction"):
        self.experiment_name = experiment_name
        self.model = None
        self.feature_columns: list[str] = []
        self.target_column = settings.model.target_column
        self.params = self._load_params()
        
        # Setup MLflow
        self._setup_mlflow()

    def _load_params(self) -> dict:
        """Load parameters from params.yaml."""
        params_path = PROJECT_ROOT / "params.yaml"
        if params_path.exists():
            with open(params_path) as f:
                return yaml.safe_load(f)
        return {}

    def _setup_mlflow(self):
        """Configure MLflow tracking with Dagshub."""
        dagshub_config = settings.dagshub
        
        if dagshub_config.username and dagshub_config.token:
            # Set Dagshub credentials
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_config.username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_config.token
            
            tracking_uri = dagshub_config.mlflow_tracking_uri
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
        else:
            # Use local MLflow
            mlflow.set_tracking_uri("mlruns")
            logger.warning("Dagshub not configured, using local MLflow tracking")
        
        # Set or create experiment
        mlflow.set_experiment(self.experiment_name)

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Prepare data for training.
        
        Args:
            df: Transformed DataFrame with features
            feature_columns: List of feature column names (auto-detected if None)
            
        Returns:
            X, y, feature_columns
        """
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            exclude_cols = [
                "timestamp", "dt", "city", "country", "latitude", "longitude",
                "weather_main", "weather_description", "sunrise", "sunset",
                "forecast_dt", "forecast_datetime", self.target_column,
            ]
            feature_columns = [
                col for col in df.columns
                if col not in exclude_cols
                and not col.startswith("weather_")  # Exclude raw weather strings
                and df[col].dtype in ["float64", "int64", "float32", "int32"]
            ]
        
        self.feature_columns = feature_columns
        
        # Prepare X and y
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        logger.info(f"Prepared data: {len(X)} samples, {len(feature_columns)} features")
        return X, y, feature_columns

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Optional[dict] = None,
    ) -> dict[str, float]:
        """
        Train the model with MLflow tracking.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_params: Model hyperparameters (uses params.yaml if None)
            
        Returns:
            Dictionary of metrics
        """
        # Get parameters
        train_params = self.params.get("train", {})
        if model_params:
            train_params.update(model_params)
        
        # Split data (time-series aware)
        test_size = train_params.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=False,  # Don't shuffle for time series
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            
            # Log parameters
            mlflow.log_params({
                "model_type": train_params.get("model_type", "xgboost"),
                "n_estimators": train_params.get("n_estimators", 100),
                "max_depth": train_params.get("max_depth", 6),
                "learning_rate": train_params.get("learning_rate", 0.1),
                "subsample": train_params.get("subsample", 0.8),
                "colsample_bytree": train_params.get("colsample_bytree", 0.8),
                "test_size": test_size,
                "n_features": len(self.feature_columns),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "forecast_horizon_hours": settings.model.forecast_horizon_hours,
            })
            
            # Initialize model
            self.model = XGBRegressor(
                n_estimators=train_params.get("n_estimators", 100),
                max_depth=train_params.get("max_depth", 6),
                learning_rate=train_params.get("learning_rate", 0.1),
                subsample=train_params.get("subsample", 0.8),
                colsample_bytree=train_params.get("colsample_bytree", 0.8),
                random_state=train_params.get("random_state", 42),
                n_jobs=-1,
            )
            
            # Train model
            logger.info("Training XGBoost model...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "test_r2": r2_score(y_test, y_pred_test),
            }
            
            # Cross-validation
            cv_params = self.params.get("cv", {})
            tscv = TimeSeriesSplit(n_splits=cv_params.get("n_splits", 5))
            cv_scores = cross_val_score(
                self.model, X, y,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
            )
            metrics["cv_rmse_mean"] = -cv_scores.mean()
            metrics["cv_rmse_std"] = cv_scores.std()
            
            # Log metrics
            mlflow.log_metrics(metrics)
            logger.info(f"Metrics: {metrics}")
            
            # Log model
            mlflow.xgboost.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=settings.model.model_name,
            )
            
            # Log feature importance
            self._log_feature_importance(run_id)
            
            # Log feature columns
            mlflow.log_dict(
                {"feature_columns": self.feature_columns},
                "feature_columns.json",
            )
            
            # Save metrics locally
            self._save_metrics(metrics)
            
            return metrics

    def _log_feature_importance(self, run_id: str):
        """Log feature importance plot and data."""
        import matplotlib.pyplot as plt
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        
        # Log as artifact
        mlflow.log_dict(
            feature_importance.to_dict(orient="records"),
            "feature_importance.json",
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        ax.barh(range(top_n), top_features["importance"].values)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        
        # Save plot
        plot_path = REPORTS_DIR / "feature_importance.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        mlflow.log_artifact(str(plot_path))
        logger.info(f"Logged feature importance plot to {plot_path}")

    def _save_metrics(self, metrics: dict):
        """Save metrics to local file."""
        metrics_path = REPORTS_DIR / "metrics.json"
        
        # Add metadata
        metrics_with_meta = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": settings.model.model_name,
            **metrics,
        }
        
        with open(metrics_path, "w") as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")

    def save_model(self, output_path: Optional[Path] = None) -> Path:
        """Save model locally."""
        output_path = output_path or MODELS_DIR / "model.joblib"
        
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Saved model to {output_path}")
        
        return output_path

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load model from file."""
        model_path = model_path or MODELS_DIR / "model.joblib"
        
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.target_column = model_data["target_column"]
        
        logger.info(f"Loaded model from {model_path}")


def train_model(
    data_path: Optional[Path] = None,
    save_model: bool = True,
) -> dict[str, float]:
    """
    Main training function for Airflow DAG.
    
    Args:
        data_path: Path to processed data (uses latest if None)
        save_model: Whether to save model locally
        
    Returns:
        Dictionary of metrics
    """
    # Find latest processed data
    if data_path is None:
        processed_files = list(PROCESSED_DATA_DIR.glob("*.parquet"))
        if not processed_files:
            raise FileNotFoundError("No processed data found")
        data_path = max(processed_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Initialize trainer
    trainer = WeatherModelTrainer()
    
    # Prepare data
    X, y, feature_columns = trainer.prepare_data(df)
    
    # Train model
    metrics = trainer.train(X, y)
    
    # Save model locally
    if save_model:
        trainer.save_model()
    
    return metrics


if __name__ == "__main__":
    # Test training with sample data
    import numpy as np
    
    # Create sample processed data
    n_samples = 500
    sample_data = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * np.arange(n_samples) / 24),
        "hour_cos": np.cos(2 * np.pi * np.arange(n_samples) / 24),
        "dow_sin": np.sin(2 * np.pi * (np.arange(n_samples) // 24) / 7),
        "dow_cos": np.cos(2 * np.pi * (np.arange(n_samples) // 24) / 7),
        "month_sin": np.zeros(n_samples),
        "month_cos": np.ones(n_samples),
        "is_weekend": np.zeros(n_samples),
        "temperature": np.random.uniform(10, 30, n_samples),
        "humidity": np.random.uniform(40, 80, n_samples),
        "pressure": np.random.uniform(1000, 1020, n_samples),
        "wind_speed": np.random.uniform(0, 20, n_samples),
        "clouds": np.random.uniform(0, 100, n_samples),
        "visibility": np.random.uniform(5000, 10000, n_samples),
        "temperature_lag_1h": np.random.uniform(10, 30, n_samples),
        "temperature_lag_3h": np.random.uniform(10, 30, n_samples),
        "temperature_rolling_mean_6h": np.random.uniform(10, 30, n_samples),
    })
    
    # Create target (temperature 6h ahead with some noise)
    sample_data["temperature_target"] = (
        sample_data["temperature"] 
        + np.random.normal(0, 2, n_samples)
    )
    
    # Save sample data
    sample_path = PROCESSED_DATA_DIR / "sample_processed.parquet"
    sample_data.to_parquet(sample_path)
    
    # Train
    metrics = train_model(data_path=sample_path)
    print(f"\nTraining metrics: {metrics}")

