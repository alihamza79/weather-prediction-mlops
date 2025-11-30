"""Data quality validation module."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.config import settings


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Complete data quality report."""

    timestamp: str
    total_rows: int
    total_columns: int
    checks: list[QualityCheckResult]
    passed: bool
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "passed": self.passed,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class DataQualityChecker:
    """Validate data quality with configurable checks."""

    # Required columns for weather data
    REQUIRED_COLUMNS = [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]

    # Numeric columns that should not have nulls
    CRITICAL_NUMERIC_COLUMNS = [
        "temperature",
        "humidity",
        "pressure",
    ]

    def __init__(self, config: Optional[Any] = None):
        self.config = config or settings.data_quality

    def check_null_ratio(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if null ratio is within acceptable limits."""
        null_ratios = df[self.CRITICAL_NUMERIC_COLUMNS].isnull().mean()
        max_null_ratio = null_ratios.max()
        
        passed = max_null_ratio <= self.config.max_null_ratio
        
        return QualityCheckResult(
            check_name="null_ratio_check",
            passed=passed,
            message=f"Max null ratio: {max_null_ratio:.4f} (threshold: {self.config.max_null_ratio})",
            details={
                "null_ratios": null_ratios.to_dict(),
                "max_null_ratio": max_null_ratio,
                "threshold": self.config.max_null_ratio,
            },
        )

    def check_minimum_rows(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if dataset has minimum required rows."""
        row_count = len(df)
        passed = row_count >= self.config.min_rows
        
        return QualityCheckResult(
            check_name="minimum_rows_check",
            passed=passed,
            message=f"Row count: {row_count} (minimum: {self.config.min_rows})",
            details={
                "row_count": row_count,
                "minimum_required": self.config.min_rows,
            },
        )

    def check_schema(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if required columns are present."""
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        passed = len(missing_columns) == 0
        
        return QualityCheckResult(
            check_name="schema_validation",
            passed=passed,
            message=f"Missing columns: {missing_columns}" if missing_columns else "All required columns present",
            details={
                "required_columns": self.REQUIRED_COLUMNS,
                "present_columns": list(df.columns),
                "missing_columns": missing_columns,
            },
        )

    def check_temperature_range(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if temperature values are within valid range."""
        if "temperature" not in df.columns:
            return QualityCheckResult(
                check_name="temperature_range_check",
                passed=False,
                message="Temperature column not found",
                details={},
            )
        
        temp = df["temperature"].dropna()
        min_temp = temp.min()
        max_temp = temp.max()
        
        passed = (
            min_temp >= self.config.min_temperature
            and max_temp <= self.config.max_temperature
        )
        
        return QualityCheckResult(
            check_name="temperature_range_check",
            passed=passed,
            message=f"Temperature range: [{min_temp:.2f}, {max_temp:.2f}] (valid: [{self.config.min_temperature}, {self.config.max_temperature}])",
            details={
                "min_value": min_temp,
                "max_value": max_temp,
                "valid_min": self.config.min_temperature,
                "valid_max": self.config.max_temperature,
            },
        )

    def check_humidity_range(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if humidity values are within valid range (0-100%)."""
        if "humidity" not in df.columns:
            return QualityCheckResult(
                check_name="humidity_range_check",
                passed=False,
                message="Humidity column not found",
                details={},
            )
        
        humidity = df["humidity"].dropna()
        min_humidity = humidity.min()
        max_humidity = humidity.max()
        
        passed = (
            min_humidity >= self.config.min_humidity
            and max_humidity <= self.config.max_humidity
        )
        
        return QualityCheckResult(
            check_name="humidity_range_check",
            passed=passed,
            message=f"Humidity range: [{min_humidity:.2f}, {max_humidity:.2f}] (valid: [{self.config.min_humidity}, {self.config.max_humidity}])",
            details={
                "min_value": min_humidity,
                "max_value": max_humidity,
                "valid_min": self.config.min_humidity,
                "valid_max": self.config.max_humidity,
            },
        )

    def check_duplicates(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for duplicate timestamps."""
        if "timestamp" not in df.columns:
            return QualityCheckResult(
                check_name="duplicate_check",
                passed=True,
                message="No timestamp column to check duplicates",
                details={},
            )
        
        duplicate_count = df["timestamp"].duplicated().sum()
        duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0
        
        # Allow some duplicates (up to 5%)
        passed = duplicate_ratio <= 0.05
        
        return QualityCheckResult(
            check_name="duplicate_check",
            passed=passed,
            message=f"Duplicate timestamps: {duplicate_count} ({duplicate_ratio:.2%})",
            details={
                "duplicate_count": int(duplicate_count),
                "duplicate_ratio": duplicate_ratio,
            },
        )

    def check_timestamp_continuity(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for large gaps in timestamp data."""
        if "timestamp" not in df.columns or len(df) < 2:
            return QualityCheckResult(
                check_name="timestamp_continuity_check",
                passed=True,
                message="Not enough data to check continuity",
                details={},
            )
        
        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff().dropna()
        
        # Convert to hours
        max_gap_hours = time_diffs.max().total_seconds() / 3600 if len(time_diffs) > 0 else 0
        median_gap_hours = time_diffs.median().total_seconds() / 3600 if len(time_diffs) > 0 else 0
        
        # Flag if max gap is more than 24 hours
        passed = max_gap_hours <= 24
        
        return QualityCheckResult(
            check_name="timestamp_continuity_check",
            passed=passed,
            message=f"Max gap: {max_gap_hours:.2f}h, Median gap: {median_gap_hours:.2f}h",
            details={
                "max_gap_hours": max_gap_hours,
                "median_gap_hours": median_gap_hours,
            },
        )

    def run_all_checks(
        self,
        df: pd.DataFrame,
        fail_on_error: bool = True,
    ) -> DataQualityReport:
        """
        Run all quality checks on the dataframe.
        
        Args:
            df: DataFrame to validate
            fail_on_error: If True, raise exception on failed checks
            
        Returns:
            DataQualityReport with all check results
        """
        logger.info(f"Running data quality checks on {len(df)} rows")
        
        checks = [
            self.check_schema(df),
            self.check_null_ratio(df),
            self.check_minimum_rows(df),
            self.check_temperature_range(df),
            self.check_humidity_range(df),
            self.check_duplicates(df),
            self.check_timestamp_continuity(df),
        ]
        
        # Log each check result
        for check in checks:
            status = "✓" if check.passed else "✗"
            log_func = logger.info if check.passed else logger.warning
            log_func(f"  {status} {check.check_name}: {check.message}")
        
        all_passed = all(check.passed for check in checks)
        
        report = DataQualityReport(
            timestamp=datetime.utcnow().isoformat(),
            total_rows=len(df),
            total_columns=len(df.columns),
            checks=checks,
            passed=all_passed,
        )
        
        if not all_passed and fail_on_error:
            failed_checks = [c.check_name for c in checks if not c.passed]
            raise DataQualityError(
                f"Data quality checks failed: {failed_checks}",
                report=report,
            )
        
        logger.info(f"Data quality check {'PASSED' if all_passed else 'FAILED'}")
        return report


class DataQualityError(Exception):
    """Exception raised when data quality checks fail."""

    def __init__(self, message: str, report: DataQualityReport):
        super().__init__(message)
        self.report = report


def validate_weather_data(
    df: pd.DataFrame,
    fail_on_error: bool = True,
) -> DataQualityReport:
    """
    Convenience function to validate weather data.
    
    This is the main entry point for the Airflow DAG.
    """
    checker = DataQualityChecker()
    return checker.run_all_checks(df, fail_on_error=fail_on_error)


if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    
    sample_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": np.random.uniform(10, 30, 100),
        "humidity": np.random.uniform(40, 80, 100),
        "pressure": np.random.uniform(1000, 1020, 100),
        "wind_speed": np.random.uniform(0, 20, 100),
    })
    
    report = validate_weather_data(sample_data, fail_on_error=False)
    print(f"\nQuality Report: {'PASSED' if report.passed else 'FAILED'}")

