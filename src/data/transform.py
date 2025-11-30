"""Data transformation and feature engineering module."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, settings


class WeatherDataTransformer:
    """Transform raw weather data and engineer features for ML."""

    def __init__(self, config: Any | None = None):
        self.config = config or settings.model
        self.feature_columns: list[str] = []

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp."""
        df = df.copy()

        if "timestamp" not in df.columns:
            logger.warning("No timestamp column found, skipping time features")
            return df

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract time components
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding for hour (captures 23:00 being close to 00:00)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        logger.info("Created time-based features")
        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
        lag_hours: list[int] | None = None,
    ) -> pd.DataFrame:
        """Create lag features for specified columns."""
        if columns is None:
            columns = ["temperature", "humidity", "pressure", "wind_speed"]
        df = df.copy()
        lag_hours = lag_hours or self.config.lag_features

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping lag features")
                continue

            for lag in lag_hours:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

        logger.info(f"Created lag features for {columns} with lags {lag_hours}")
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Create rolling window statistics."""
        if columns is None:
            columns = ["temperature", "humidity", "pressure", "wind_speed"]
        df = df.copy()
        windows = windows or self.config.rolling_windows

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping rolling features")
                continue

            for window in windows:
                # Rolling mean
                df[f"{col}_rolling_mean_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                # Rolling std
                df[f"{col}_rolling_std_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                # Rolling min/max
                df[f"{col}_rolling_min_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                df[f"{col}_rolling_max_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )

        logger.info(f"Created rolling features for {columns} with windows {windows}")
        return df

    def create_diff_features(
        self,
        df: pd.DataFrame,
        columns: list[str] = None,
    ) -> pd.DataFrame:
        """Create difference features (rate of change)."""
        if columns is None:
            columns = ["temperature", "humidity", "pressure"]
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            # First difference (change from previous hour)
            df[f"{col}_diff_1h"] = df[col].diff(1)

            # 3-hour difference
            df[f"{col}_diff_3h"] = df[col].diff(3)

            # 6-hour difference
            df[f"{col}_diff_6h"] = df[col].diff(6)

        logger.info(f"Created difference features for {columns}")
        return df

    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_column: str = "temperature",
        horizon_hours: int | None = None,
    ) -> pd.DataFrame:
        """Create target variable (future value to predict)."""
        df = df.copy()
        horizon = horizon_hours or self.config.forecast_horizon_hours

        # Target is the temperature `horizon` hours in the future
        df["temperature_target"] = df[target_column].shift(-horizon)

        logger.info(f"Created target variable: {target_column} {horizon}h ahead")
        return df

    def encode_weather_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode weather condition."""
        df = df.copy()

        if "weather_main" not in df.columns:
            return df

        # Get dummies for weather condition
        weather_dummies = pd.get_dummies(
            df["weather_main"],
            prefix="weather",
            dummy_na=False,
        )

        df = pd.concat([df, weather_dummies], axis=1)
        logger.info(f"Encoded weather conditions: {list(weather_dummies.columns)}")
        return df

    def select_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Select features for model training."""
        # Define feature groups
        time_features = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
        ]

        base_features = [
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "clouds",
            "visibility",
        ]

        # Get all lag features
        lag_features = [col for col in df.columns if "_lag_" in col]

        # Get all rolling features
        rolling_features = [col for col in df.columns if "_rolling_" in col]

        # Get all diff features
        diff_features = [col for col in df.columns if "_diff_" in col]

        # Get weather condition features
        weather_features = [col for col in df.columns if col.startswith("weather_")]

        # Combine all features
        all_features = (
            time_features
            + base_features
            + lag_features
            + rolling_features
            + diff_features
            + weather_features
        )

        # Filter to only existing columns
        feature_columns = [col for col in all_features if col in df.columns]

        self.feature_columns = feature_columns
        logger.info(f"Selected {len(feature_columns)} features for training")

        return df, feature_columns

    def transform(
        self,
        df: pd.DataFrame,
        create_target: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all transformations to create ML-ready dataset.

        Args:
            df: Raw weather DataFrame
            create_target: Whether to create target variable

        Returns:
            Transformed DataFrame with features
        """
        logger.info(f"Transforming {len(df)} rows of weather data")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Apply transformations
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_diff_features(df)
        df = self.encode_weather_condition(df)

        if create_target:
            df = self.create_target_variable(df)

        # Select features
        df, feature_columns = self.select_features(df)

        # Drop rows with NaN in target (due to shifting)
        if create_target and "temperature_target" in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=["temperature_target"])
            logger.info(f"Dropped {initial_len - len(df)} rows with NaN target")

        logger.info(f"Transformation complete: {len(df)} rows, {len(feature_columns)} features")
        return df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_dir: Path | None = None,
        filename: str | None = None,
    ) -> Path:
        """Save processed data to parquet format."""
        output_dir = output_dir or PROCESSED_DATA_DIR
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"processed_weather_{timestamp}.parquet"

        output_path = output_dir / filename
        df.to_parquet(output_path, index=False)

        logger.info(f"Saved processed data to {output_path}")
        return output_path

    def generate_profile_report(
        self,
        df: pd.DataFrame,
        output_dir: Path | None = None,
        title: str = "Weather Data Quality Report",
    ) -> Path:
        """Generate data profile report as JSON summary."""
        output_dir = output_dir or REPORTS_DIR
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"data_profile_{timestamp}.json"

        logger.info("Generating data profile report...")

        # Generate a simple profile report
        profile = {
            "title": title,
            "timestamp": timestamp,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": {},
            "missing_values": {},
            "dtypes": {},
        }

        for col in df.columns:
            profile["dtypes"][col] = str(df[col].dtype)
            profile["missing_values"][col] = int(df[col].isna().sum())

            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                profile["columns"][col] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                }

        import json

        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Saved profile report to {output_path}")

        return output_path


def transform_weather_data(
    df: pd.DataFrame,
    save: bool = True,
    generate_report: bool = True,
) -> tuple[pd.DataFrame, Path | None, Path | None]:
    """
    Convenience function to transform weather data.

    This is the main entry point for the Airflow DAG.

    Returns:
        Tuple of (transformed_df, data_path, report_path)
    """
    transformer = WeatherDataTransformer()

    # Transform data
    df_transformed = transformer.transform(df)

    data_path = None
    report_path = None

    if save:
        data_path = transformer.save_processed_data(df_transformed)

    if generate_report:
        report_path = transformer.generate_profile_report(df_transformed)

    return df_transformed, data_path, report_path


if __name__ == "__main__":
    # Test transformation with sample data
    import numpy as np

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
            "temperature": np.random.uniform(10, 30, 200)
            + np.sin(np.arange(200) * 2 * np.pi / 24) * 5,
            "humidity": np.random.uniform(40, 80, 200),
            "pressure": np.random.uniform(1000, 1020, 200),
            "wind_speed": np.random.uniform(0, 20, 200),
            "clouds": np.random.uniform(0, 100, 200),
            "visibility": np.random.uniform(5000, 10000, 200),
            "weather_main": np.random.choice(["Clear", "Clouds", "Rain"], 200),
        }
    )

    df_transformed, data_path, report_path = transform_weather_data(
        sample_data,
        save=True,
        generate_report=False,  # Skip report for quick test
    )

    print(f"\nTransformed data shape: {df_transformed.shape}")
    print(f"Features: {df_transformed.columns.tolist()[:20]}...")
    print(f"Data saved to: {data_path}")
