"""Data extraction module for OpenWeatherMap API."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from loguru import logger

from src.config import settings, RAW_DATA_DIR


class WeatherDataExtractor:
    """Extract weather data from OpenWeatherMap API."""

    def __init__(self):
        self.config = settings.openweathermap
        self.api_key = self.config.api_key
        self.city = self.config.city_name
        self.country = self.config.country_code
        
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured. Set OPENWEATHERMAP_API_KEY.")

    def _get_coordinates(self) -> tuple[float, float]:
        """Get latitude and longitude for the configured city."""
        geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": f"{self.city},{self.country}",
            "limit": 1,
            "appid": self.api_key,
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(geocoding_url, params=params)
            response.raise_for_status()
            data = response.json()
            
        if not data:
            raise ValueError(f"Could not find coordinates for {self.city}, {self.country}")
        
        return data[0]["lat"], data[0]["lon"]

    def fetch_current_weather(self) -> dict[str, Any]:
        """Fetch current weather data."""
        url = f"{self.config.base_url}/weather"
        params = {
            "q": f"{self.city},{self.country}",
            "appid": self.api_key,
            "units": "metric",  # Celsius
        }
        
        logger.info(f"Fetching current weather for {self.city}, {self.country}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        return self._parse_current_weather(data)

    def fetch_forecast(self) -> list[dict[str, Any]]:
        """Fetch 5-day/3-hour forecast data (40 data points)."""
        url = self.config.forecast_url
        params = {
            "q": f"{self.city},{self.country}",
            "appid": self.api_key,
            "units": "metric",
        }
        
        logger.info(f"Fetching 5-day forecast for {self.city}, {self.country}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        return [self._parse_forecast_item(item) for item in data.get("list", [])]

    def _parse_current_weather(self, data: dict) -> dict[str, Any]:
        """Parse current weather API response."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dt": data.get("dt"),
            "city": self.city,
            "country": self.country,
            "latitude": data.get("coord", {}).get("lat"),
            "longitude": data.get("coord", {}).get("lon"),
            "temperature": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "temp_min": data.get("main", {}).get("temp_min"),
            "temp_max": data.get("main", {}).get("temp_max"),
            "pressure": data.get("main", {}).get("pressure"),
            "humidity": data.get("main", {}).get("humidity"),
            "visibility": data.get("visibility"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "wind_deg": data.get("wind", {}).get("deg"),
            "wind_gust": data.get("wind", {}).get("gust"),
            "clouds": data.get("clouds", {}).get("all"),
            "weather_main": data.get("weather", [{}])[0].get("main"),
            "weather_description": data.get("weather", [{}])[0].get("description"),
            "rain_1h": data.get("rain", {}).get("1h", 0),
            "rain_3h": data.get("rain", {}).get("3h", 0),
            "snow_1h": data.get("snow", {}).get("1h", 0),
            "snow_3h": data.get("snow", {}).get("3h", 0),
            "sunrise": data.get("sys", {}).get("sunrise"),
            "sunset": data.get("sys", {}).get("sunset"),
        }

    def _parse_forecast_item(self, item: dict) -> dict[str, Any]:
        """Parse a single forecast item."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "forecast_dt": item.get("dt"),
            "forecast_datetime": item.get("dt_txt"),
            "city": self.city,
            "country": self.country,
            "temperature": item.get("main", {}).get("temp"),
            "feels_like": item.get("main", {}).get("feels_like"),
            "temp_min": item.get("main", {}).get("temp_min"),
            "temp_max": item.get("main", {}).get("temp_max"),
            "pressure": item.get("main", {}).get("pressure"),
            "humidity": item.get("main", {}).get("humidity"),
            "sea_level": item.get("main", {}).get("sea_level"),
            "grnd_level": item.get("main", {}).get("grnd_level"),
            "visibility": item.get("visibility"),
            "wind_speed": item.get("wind", {}).get("speed"),
            "wind_deg": item.get("wind", {}).get("deg"),
            "wind_gust": item.get("wind", {}).get("gust"),
            "clouds": item.get("clouds", {}).get("all"),
            "weather_main": item.get("weather", [{}])[0].get("main"),
            "weather_description": item.get("weather", [{}])[0].get("description"),
            "rain_3h": item.get("rain", {}).get("3h", 0),
            "snow_3h": item.get("snow", {}).get("3h", 0),
            "pop": item.get("pop", 0),  # Probability of precipitation
        }

    def extract_and_save(
        self,
        include_forecast: bool = True,
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Optional[Path]]:
        """
        Extract weather data and save to raw data directory.
        
        Returns:
            Tuple of (current_weather_path, forecast_path)
        """
        output_dir = output_dir or RAW_DATA_DIR
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Fetch and save current weather
        current_weather = self.fetch_current_weather()
        current_path = output_dir / f"current_weather_{timestamp}.json"
        
        with open(current_path, "w") as f:
            json.dump(current_weather, f, indent=2)
        
        logger.info(f"Saved current weather to {current_path}")
        
        forecast_path = None
        if include_forecast:
            # Fetch and save forecast
            forecast_data = self.fetch_forecast()
            forecast_path = output_dir / f"forecast_{timestamp}.json"
            
            with open(forecast_path, "w") as f:
                json.dump(forecast_data, f, indent=2)
            
            logger.info(f"Saved forecast to {forecast_path}")
        
        return current_path, forecast_path

    def load_historical_data(self, data_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Load all historical weather data from raw data directory.
        
        This combines all previously collected data points.
        """
        data_dir = data_dir or RAW_DATA_DIR
        
        # Load current weather files
        current_files = list(data_dir.glob("current_weather_*.json"))
        current_data = []
        
        for file_path in current_files:
            with open(file_path) as f:
                data = json.load(f)
                current_data.append(data)
        
        if not current_data:
            logger.warning("No historical weather data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(current_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} historical weather records")
        return df

    def load_forecast_data(self, data_dir: Optional[Path] = None) -> pd.DataFrame:
        """Load all forecast data from raw data directory."""
        data_dir = data_dir or RAW_DATA_DIR
        
        forecast_files = list(data_dir.glob("forecast_*.json"))
        all_forecasts = []
        
        for file_path in forecast_files:
            with open(file_path) as f:
                data = json.load(f)
                all_forecasts.extend(data)
        
        if not all_forecasts:
            logger.warning("No forecast data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_forecasts)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["forecast_datetime"] = pd.to_datetime(df["forecast_datetime"])
        df = df.sort_values(["timestamp", "forecast_datetime"]).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} forecast records")
        return df


def extract_weather_data(
    include_forecast: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Optional[Path]]:
    """
    Convenience function to extract weather data.
    
    This is the main entry point for the Airflow DAG.
    """
    extractor = WeatherDataExtractor()
    return extractor.extract_and_save(include_forecast=include_forecast, output_dir=output_dir)


if __name__ == "__main__":
    # Test extraction
    current_path, forecast_path = extract_weather_data()
    print(f"Current weather saved to: {current_path}")
    print(f"Forecast saved to: {forecast_path}")

