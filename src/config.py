"""Configuration management for the Weather Prediction MLOps project."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class OpenWeatherMapConfig(BaseModel):
    """OpenWeatherMap API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENWEATHERMAP_API_KEY", ""))
    city_name: str = Field(default_factory=lambda: os.getenv("CITY_NAME", "London"))
    country_code: str = Field(default_factory=lambda: os.getenv("COUNTRY_CODE", "GB"))
    base_url: str = "https://api.openweathermap.org/data/2.5"
    forecast_url: str = "https://api.openweathermap.org/data/2.5/forecast"
    history_url: str = "https://history.openweathermap.org/data/2.5/history/city"


class DagshubConfig(BaseModel):
    """Dagshub and MLflow configuration."""

    username: str = Field(default_factory=lambda: os.getenv("DAGSHUB_USERNAME", ""))
    repo_name: str = Field(
        default_factory=lambda: os.getenv("DAGSHUB_REPO_NAME", "weather-prediction-mlops")
    )
    token: str = Field(default_factory=lambda: os.getenv("DAGSHUB_TOKEN", ""))

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"https://dagshub.com/{self.username}/{self.repo_name}.mlflow"

    @property
    def dvc_remote_url(self) -> str:
        return f"https://dagshub.com/{self.username}/{self.repo_name}.dvc"


class MinIOConfig(BaseModel):
    """MinIO (S3-compatible) storage configuration."""

    endpoint: str = Field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    access_key: str = Field(default_factory=lambda: os.getenv("MINIO_ROOT_USER", "minioadmin"))
    secret_key: str = Field(default_factory=lambda: os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"))
    bucket_name: str = Field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "weather-data"))
    secure: bool = False


class ModelConfig(BaseModel):
    """Model training configuration."""

    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "weather_predictor"))
    target_column: str = "temperature_target"
    forecast_horizon_hours: int = 6  # Predict 6 hours ahead
    train_test_split: float = 0.2
    random_state: int = 42

    # Feature engineering
    lag_features: list[int] = [1, 2, 3, 6, 12, 24]  # Hours of lag
    rolling_windows: list[int] = [3, 6, 12, 24]  # Rolling window sizes


class APIConfig(BaseModel):
    """FastAPI configuration."""

    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")


class DataQualityConfig(BaseModel):
    """Data quality thresholds."""

    max_null_ratio: float = 0.01  # Max 1% null values
    min_rows: int = 100  # Minimum rows required
    max_temperature: float = 60.0  # Celsius
    min_temperature: float = -60.0  # Celsius
    max_humidity: float = 100.0
    min_humidity: float = 0.0


class MonitoringConfig(BaseModel):
    """Prometheus and Grafana configuration."""

    prometheus_port: int = Field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    grafana_port: int = Field(default_factory=lambda: int(os.getenv("GRAFANA_PORT", "3000")))
    latency_threshold_ms: float = 500.0  # Alert if latency exceeds this
    drift_threshold: float = 0.1  # Alert if drift ratio exceeds 10%


class Settings(BaseModel):
    """Main settings container."""

    openweathermap: OpenWeatherMapConfig = Field(default_factory=OpenWeatherMapConfig)
    dagshub: DagshubConfig = Field(default_factory=DagshubConfig)
    minio: MinIOConfig = Field(default_factory=MinIOConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
