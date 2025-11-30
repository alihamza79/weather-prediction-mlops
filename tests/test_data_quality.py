"""Tests for data quality checks."""

import numpy as np
import pandas as pd
import pytest

from src.data.quality_checks import DataQualityChecker, DataQualityError


@pytest.fixture
def valid_weather_data():
    """Create valid weather data for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
        "temperature": np.random.uniform(10, 30, 200),
        "humidity": np.random.uniform(40, 80, 200),
        "pressure": np.random.uniform(1000, 1020, 200),
        "wind_speed": np.random.uniform(0, 20, 200),
    })


@pytest.fixture
def invalid_weather_data():
    """Create invalid weather data with nulls."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": [None] * 50 + list(np.random.uniform(10, 30, 50)),
        "humidity": np.random.uniform(40, 80, 100),
        "pressure": np.random.uniform(1000, 1020, 100),
        "wind_speed": np.random.uniform(0, 20, 100),
    })
    return df


class TestDataQualityChecker:
    """Test DataQualityChecker class."""

    def test_valid_data_passes(self, valid_weather_data):
        """Test that valid data passes all checks."""
        checker = DataQualityChecker()
        report = checker.run_all_checks(valid_weather_data, fail_on_error=False)
        
        assert report.passed is True
        assert report.total_rows == 200

    def test_null_check_fails(self, invalid_weather_data):
        """Test that data with too many nulls fails."""
        checker = DataQualityChecker()
        
        with pytest.raises(DataQualityError):
            checker.run_all_checks(invalid_weather_data, fail_on_error=True)

    def test_schema_validation(self, valid_weather_data):
        """Test schema validation."""
        checker = DataQualityChecker()
        result = checker.check_schema(valid_weather_data)
        
        assert result.passed is True

    def test_missing_columns_fails(self):
        """Test that missing required columns fails."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "temperature": np.random.uniform(10, 30, 100),
        })
        
        checker = DataQualityChecker()
        result = checker.check_schema(df)
        
        assert result.passed is False
        assert "humidity" in result.details["missing_columns"]

    def test_temperature_range_check(self, valid_weather_data):
        """Test temperature range validation."""
        checker = DataQualityChecker()
        result = checker.check_temperature_range(valid_weather_data)
        
        assert result.passed == True

    def test_extreme_temperature_fails(self):
        """Test that extreme temperatures fail."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "temperature": [100.0] * 100,  # Invalid temperature
            "humidity": np.random.uniform(40, 80, 100),
            "pressure": np.random.uniform(1000, 1020, 100),
            "wind_speed": np.random.uniform(0, 20, 100),
        })
        
        checker = DataQualityChecker()
        result = checker.check_temperature_range(df)
        
        assert result.passed == False

    def test_minimum_rows_check(self):
        """Test minimum rows validation."""
        small_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "temperature": np.random.uniform(10, 30, 10),
            "humidity": np.random.uniform(40, 80, 10),
            "pressure": np.random.uniform(1000, 1020, 10),
            "wind_speed": np.random.uniform(0, 20, 10),
        })
        
        checker = DataQualityChecker()
        result = checker.check_minimum_rows(small_df)
        
        assert result.passed is False  # Default minimum is 100

