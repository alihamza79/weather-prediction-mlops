#!/usr/bin/env python3
"""
Generate sample weather data for testing and development.

This script creates synthetic weather data that mimics the structure
of real OpenWeatherMap API data, useful for testing without API calls.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def generate_realistic_temperature(hours: int, base_temp: float = 15.0) -> np.ndarray:
    """Generate realistic temperature with daily and random patterns."""
    t = np.arange(hours)
    
    # Daily cycle (warmer during day, cooler at night)
    daily_cycle = 5 * np.sin(2 * np.pi * (t - 6) / 24)
    
    # Random walk for weather changes
    random_walk = np.cumsum(np.random.normal(0, 0.3, hours))
    random_walk = random_walk - np.mean(random_walk)  # Center around 0
    
    # Combine
    temperature = base_temp + daily_cycle + random_walk + np.random.normal(0, 1, hours)
    
    return temperature


def generate_sample_raw_data(n_days: int = 7, city: str = "London") -> list[Path]:
    """Generate sample raw weather data files."""
    print(f"Generating {n_days} days of sample raw data for {city}...")
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now(timezone.utc) - timedelta(days=n_days)
    files = []
    
    for hour in range(n_days * 24):
        timestamp = start_time + timedelta(hours=hour)
        temp = generate_realistic_temperature(1, base_temp=15)[0]
        
        data = {
            "timestamp": timestamp.isoformat(),
            "dt": int(timestamp.timestamp()),
            "city": city,
            "country": "GB",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "temperature": round(temp, 2),
            "feels_like": round(temp - 2 + np.random.uniform(-1, 1), 2),
            "temp_min": round(temp - np.random.uniform(1, 3), 2),
            "temp_max": round(temp + np.random.uniform(1, 3), 2),
            "pressure": round(1013 + np.random.uniform(-10, 10), 1),
            "humidity": round(np.clip(60 + np.random.uniform(-20, 20), 30, 95), 1),
            "visibility": int(np.clip(10000 + np.random.uniform(-3000, 0), 5000, 10000)),
            "wind_speed": round(np.clip(5 + np.random.uniform(-3, 5), 0, 20), 1),
            "wind_deg": int(np.random.uniform(0, 360)),
            "wind_gust": round(np.clip(8 + np.random.uniform(-2, 5), 0, 25), 1),
            "clouds": int(np.clip(50 + np.random.uniform(-40, 40), 0, 100)),
            "weather_main": np.random.choice(["Clear", "Clouds", "Rain", "Drizzle"], p=[0.3, 0.4, 0.2, 0.1]),
            "weather_description": "sample weather",
            "rain_1h": round(np.random.uniform(0, 2), 2) if np.random.random() > 0.7 else 0,
            "rain_3h": 0,
            "snow_1h": 0,
            "snow_3h": 0,
            "sunrise": int((timestamp.replace(hour=6, minute=0)).timestamp()),
            "sunset": int((timestamp.replace(hour=18, minute=0)).timestamp()),
        }
        
        filename = f"current_weather_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = RAW_DATA_DIR / filename
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        files.append(filepath)
    
    print(f"✅ Generated {len(files)} raw data files in {RAW_DATA_DIR}")
    return files


def generate_sample_processed_data(n_samples: int = 1000) -> Path:
    """Generate sample processed data ready for training."""
    print(f"Generating {n_samples} samples of processed data...")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate time series
    hours = np.arange(n_samples)
    temperature = generate_realistic_temperature(n_samples, base_temp=15)
    
    data = pd.DataFrame({
        # Time features
        "hour_sin": np.sin(2 * np.pi * (hours % 24) / 24),
        "hour_cos": np.cos(2 * np.pi * (hours % 24) / 24),
        "dow_sin": np.sin(2 * np.pi * ((hours // 24) % 7) / 7),
        "dow_cos": np.cos(2 * np.pi * ((hours // 24) % 7) / 7),
        "month_sin": np.sin(2 * np.pi * ((hours // (24 * 30)) % 12) / 12),
        "month_cos": np.cos(2 * np.pi * ((hours // (24 * 30)) % 12) / 12),
        "is_weekend": ((hours // 24) % 7 >= 5).astype(int),
        
        # Base features
        "temperature": temperature,
        "humidity": np.clip(60 + np.random.uniform(-20, 20, n_samples), 30, 95),
        "pressure": 1013 + np.random.uniform(-10, 10, n_samples),
        "wind_speed": np.clip(5 + np.random.uniform(-3, 5, n_samples), 0, 20),
        "clouds": np.clip(50 + np.random.uniform(-40, 40, n_samples), 0, 100),
        "visibility": np.clip(10000 + np.random.uniform(-3000, 0, n_samples), 5000, 10000),
    })
    
    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        data[f"temperature_lag_{lag}h"] = data["temperature"].shift(lag)
    
    # Rolling features
    for window in [3, 6, 12, 24]:
        data[f"temperature_rolling_mean_{window}h"] = data["temperature"].rolling(window).mean()
        data[f"temperature_rolling_std_{window}h"] = data["temperature"].rolling(window).std()
    
    # Difference features
    data["temperature_diff_1h"] = data["temperature"].diff(1)
    data["temperature_diff_3h"] = data["temperature"].diff(3)
    
    # Target (6 hours ahead)
    data["temperature_target"] = data["temperature"].shift(-6)
    
    # Drop rows with NaN
    data = data.dropna().reset_index(drop=True)
    
    # Save
    output_path = PROCESSED_DATA_DIR / "sample_processed_data.parquet"
    data.to_parquet(output_path, index=False)
    
    print(f"✅ Generated processed data with {len(data)} samples")
    print(f"   Features: {len(data.columns) - 1}")
    print(f"   Saved to: {output_path}")
    
    return output_path


def main():
    """Generate all sample data."""
    print("=" * 60)
    print("Weather Prediction MLOps - Sample Data Generator")
    print("=" * 60)
    
    # Generate raw data
    generate_sample_raw_data(n_days=7)
    
    print()
    
    # Generate processed data
    generate_sample_processed_data(n_samples=1000)
    
    print()
    print("=" * 60)
    print("✅ Sample data generation complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Run the training script: python -m src.models.train")
    print("2. Start the API: uvicorn src.api.app:app --reload")
    print("3. Run the Airflow DAG for the full pipeline")


if __name__ == "__main__":
    main()

