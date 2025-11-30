"""Model prediction module."""

import os
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings, MODELS_DIR


class WeatherPredictor:
    """Make predictions with trained weather model."""

    def __init__(self, model_path: Optional[Path] = None, use_mlflow: bool = False):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to local model file
            use_mlflow: Whether to load model from MLflow registry
        """
        self.model = None
        self.feature_columns: list[str] = []
        self.target_column = settings.model.target_column
        
        if use_mlflow:
            self._load_from_mlflow()
        elif model_path:
            self._load_from_file(model_path)
        else:
            # Try default path
            default_path = MODELS_DIR / "model.joblib"
            if default_path.exists():
                self._load_from_file(default_path)

    def _load_from_file(self, model_path: Path):
        """Load model from local file."""
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.target_column = model_data.get("target_column", "temperature_target")
        logger.info(f"Loaded model from {model_path}")

    def _load_from_mlflow(self):
        """Load model from MLflow registry."""
        dagshub_config = settings.dagshub
        
        if dagshub_config.username and dagshub_config.token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_config.username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_config.token
            mlflow.set_tracking_uri(dagshub_config.mlflow_tracking_uri)
        
        model_name = settings.model.model_name
        model_uri = f"models:/{model_name}/latest"
        
        try:
            self.model = mlflow.xgboost.load_model(model_uri)
            
            # Try to load feature columns
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(model_name)
            if latest_versions:
                run_id = latest_versions[0].run_id
                artifact_path = client.download_artifacts(run_id, "feature_columns.json")
                import json
                with open(artifact_path) as f:
                    data = json.load(f)
                    self.feature_columns = data.get("feature_columns", [])
            
            logger.info(f"Loaded model '{model_name}' from MLflow registry")
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise

    def predict(self, features: Union[pd.DataFrame, dict, list[dict]]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            features: Input features as DataFrame, dict, or list of dicts
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, list):
            features = pd.DataFrame(features)
        
        # Ensure correct columns
        if self.feature_columns:
            # Add missing columns with default values
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            
            # Select only required columns in correct order
            features = features[self.feature_columns]
        
        # Handle NaN values
        features = features.fillna(features.median())
        
        # Make prediction
        predictions = self.model.predict(features)
        
        return predictions

    def predict_single(self, **kwargs) -> float:
        """
        Make a single prediction.
        
        Args:
            **kwargs: Feature values as keyword arguments
            
        Returns:
            Single prediction value
        """
        predictions = self.predict(kwargs)
        return float(predictions[0])

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


def load_predictor(
    model_path: Optional[Path] = None,
    use_mlflow: bool = False,
) -> WeatherPredictor:
    """
    Factory function to create predictor.
    
    Args:
        model_path: Path to local model file
        use_mlflow: Whether to load from MLflow registry
        
    Returns:
        Initialized WeatherPredictor
    """
    return WeatherPredictor(model_path=model_path, use_mlflow=use_mlflow)


if __name__ == "__main__":
    # Test prediction
    predictor = WeatherPredictor()
    
    if predictor.is_loaded():
        # Sample prediction
        sample_features = {
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "dow_sin": 0.0,
            "dow_cos": 1.0,
            "month_sin": 0.0,
            "month_cos": 1.0,
            "is_weekend": 0,
            "temperature": 20.0,
            "humidity": 60.0,
            "pressure": 1013.0,
            "wind_speed": 5.0,
            "clouds": 50.0,
            "visibility": 10000.0,
            "temperature_lag_1h": 19.5,
            "temperature_lag_3h": 18.0,
            "temperature_rolling_mean_6h": 19.0,
        }
        
        prediction = predictor.predict_single(**sample_features)
        print(f"Predicted temperature (6h ahead): {prediction:.2f}°C")
    else:
        print("No model loaded. Train a model first.")

