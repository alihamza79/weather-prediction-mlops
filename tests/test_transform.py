"""Tests for data transformation module."""

import numpy as np
import pandas as pd
import pytest

from src.data.transform import WeatherDataTransformer


@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": np.random.uniform(10, 30, 100),
        "humidity": np.random.uniform(40, 80, 100),
        "pressure": np.random.uniform(1000, 1020, 100),
        "wind_speed": np.random.uniform(0, 20, 100),
        "clouds": np.random.uniform(0, 100, 100),
        "visibility": np.random.uniform(5000, 10000, 100),
        "weather_main": np.random.choice(["Clear", "Clouds", "Rain"], 100),
    })


class TestWeatherDataTransformer:
    """Test WeatherDataTransformer class."""

    def test_create_time_features(self, sample_weather_data):
        """Test time feature creation."""
        transformer = WeatherDataTransformer()
        df = transformer.create_time_features(sample_weather_data)
        
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "is_weekend" in df.columns

    def test_create_lag_features(self, sample_weather_data):
        """Test lag feature creation."""
        transformer = WeatherDataTransformer()
        df = transformer.create_lag_features(
            sample_weather_data,
            columns=["temperature"],
            lag_hours=[1, 3],
        )
        
        assert "temperature_lag_1h" in df.columns
        assert "temperature_lag_3h" in df.columns

    def test_create_rolling_features(self, sample_weather_data):
        """Test rolling feature creation."""
        transformer = WeatherDataTransformer()
        df = transformer.create_rolling_features(
            sample_weather_data,
            columns=["temperature"],
            windows=[3, 6],
        )
        
        assert "temperature_rolling_mean_3h" in df.columns
        assert "temperature_rolling_std_3h" in df.columns
        assert "temperature_rolling_mean_6h" in df.columns

    def test_create_target_variable(self, sample_weather_data):
        """Test target variable creation."""
        transformer = WeatherDataTransformer()
        df = transformer.create_target_variable(
            sample_weather_data,
            target_column="temperature",
            horizon_hours=6,
        )
        
        assert "temperature_target" in df.columns
        # Target should be shifted, so last 6 values should be NaN
        assert df["temperature_target"].isna().sum() == 6

    def test_full_transform(self, sample_weather_data):
        """Test full transformation pipeline."""
        transformer = WeatherDataTransformer()
        df = transformer.transform(sample_weather_data, create_target=True)
        
        # Should have time features
        assert "hour_sin" in df.columns
        
        # Should have target
        assert "temperature_target" in df.columns
        
        # Should not have NaN in target (dropped)
        assert df["temperature_target"].isna().sum() == 0

    def test_encode_weather_condition(self, sample_weather_data):
        """Test weather condition encoding."""
        transformer = WeatherDataTransformer()
        df = transformer.encode_weather_condition(sample_weather_data)
        
        # Should have one-hot encoded columns
        weather_cols = [c for c in df.columns if c.startswith("weather_")]
        assert len(weather_cols) > 0

