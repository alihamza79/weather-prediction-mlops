"""
FastAPI Weather Prediction Service with Prometheus Metrics.

This service provides:
- REST API for weather predictions
- Health check endpoints
- Prometheus metrics for monitoring
- Data drift detection
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.config import settings
from src.models.predict import WeatherPredictor

# ============================================================================
# Prometheus Metrics
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "weather_prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "weather_prediction_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Model metrics
PREDICTION_VALUE = Histogram(
    "weather_prediction_value_celsius",
    "Distribution of predicted temperature values",
    buckets=[-20, -10, 0, 10, 15, 20, 25, 30, 35, 40, 50],
)

# Data drift metrics
FEATURE_OUT_OF_RANGE = Counter(
    "weather_feature_out_of_range_total",
    "Count of predictions with out-of-range feature values",
    ["feature"],
)

DRIFT_RATIO = Gauge(
    "weather_data_drift_ratio",
    "Ratio of requests with potential data drift",
)

# Model info
MODEL_LOADED = Gauge(
    "weather_model_loaded",
    "Whether the model is loaded (1) or not (0)",
)

LAST_PREDICTION_TIMESTAMP = Gauge(
    "weather_last_prediction_timestamp",
    "Unix timestamp of last prediction",
)


# ============================================================================
# Request/Response Models
# ============================================================================


class WeatherFeatures(BaseModel):
    """Input features for weather prediction."""

    # Time features
    hour_sin: float = Field(..., description="Sine of hour (cyclical encoding)")
    hour_cos: float = Field(..., description="Cosine of hour (cyclical encoding)")
    dow_sin: float = Field(0.0, description="Sine of day of week")
    dow_cos: float = Field(1.0, description="Cosine of day of week")
    month_sin: float = Field(0.0, description="Sine of month")
    month_cos: float = Field(1.0, description="Cosine of month")
    is_weekend: int = Field(0, ge=0, le=1, description="Weekend indicator")

    # Current weather
    temperature: float = Field(..., ge=-60, le=60, description="Current temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    pressure: float = Field(..., ge=800, le=1100, description="Pressure (hPa)")
    wind_speed: float = Field(..., ge=0, le=200, description="Wind speed (m/s)")
    clouds: float = Field(50.0, ge=0, le=100, description="Cloud coverage (%)")
    visibility: float = Field(10000.0, ge=0, description="Visibility (m)")

    # Lag features (optional)
    temperature_lag_1h: float | None = Field(None, description="Temperature 1 hour ago")
    temperature_lag_3h: float | None = Field(None, description="Temperature 3 hours ago")
    temperature_lag_6h: float | None = Field(None, description="Temperature 6 hours ago")

    # Rolling features (optional)
    temperature_rolling_mean_6h: float | None = Field(None, description="6-hour rolling mean temp")
    temperature_rolling_std_6h: float | None = Field(None, description="6-hour rolling std temp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "dow_sin": 0.0,
                "dow_cos": 1.0,
                "month_sin": 0.0,
                "month_cos": 1.0,
                "is_weekend": 0,
                "temperature": 20.0,
                "humidity": 65.0,
                "pressure": 1013.0,
                "wind_speed": 5.0,
                "clouds": 40.0,
                "visibility": 10000.0,
                "temperature_lag_1h": 19.5,
                "temperature_lag_3h": 18.0,
                "temperature_rolling_mean_6h": 19.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction response model."""

    predicted_temperature: float = Field(
        ..., description="Predicted temperature 6 hours ahead (°C)"
    )
    forecast_horizon_hours: int = Field(6, description="Forecast horizon in hours")
    model_name: str = Field(..., description="Name of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")
    drift_warning: bool = Field(False, description="Whether potential data drift was detected")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    timestamp: str
    version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    features: list[WeatherFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[float]
    count: int
    timestamp: str


# ============================================================================
# Global State
# ============================================================================

predictor: WeatherPredictor | None = None
request_count_with_drift = 0
total_request_count = 0


# ============================================================================
# Drift Detection
# ============================================================================

# Expected feature ranges (based on training data statistics)
FEATURE_RANGES = {
    "temperature": (-30, 45),
    "humidity": (10, 100),
    "pressure": (950, 1050),
    "wind_speed": (0, 50),
    "clouds": (0, 100),
    "visibility": (0, 50000),
}


def check_drift(features: WeatherFeatures) -> bool:
    """
    Check if features are out of expected ranges (basic drift detection).

    Returns True if drift is detected.
    """
    drift_detected = False

    for feature_name, (min_val, max_val) in FEATURE_RANGES.items():
        value = getattr(features, feature_name, None)
        if value is not None:
            if value < min_val or value > max_val:
                FEATURE_OUT_OF_RANGE.labels(feature=feature_name).inc()
                drift_detected = True

    return drift_detected


def update_drift_ratio(has_drift: bool):
    """Update the drift ratio metric."""
    global request_count_with_drift, total_request_count

    total_request_count += 1
    if has_drift:
        request_count_with_drift += 1

    if total_request_count > 0:
        ratio = request_count_with_drift / total_request_count
        DRIFT_RATIO.set(ratio)


# ============================================================================
# Application Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global predictor

    # Startup
    logger.info("Starting Weather Prediction API...")

    try:
        predictor = WeatherPredictor()
        if predictor.is_loaded():
            MODEL_LOADED.set(1)
            logger.info("Model loaded successfully")
        else:
            MODEL_LOADED.set(0)
            logger.warning("No model found - API will return errors until model is trained")
    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(f"Failed to load model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Weather Prediction API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Weather Prediction API",
    description="Real-time weather temperature prediction service with MLOps monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware
# ============================================================================


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Add request metrics middleware."""
    start_time = time.time()

    response = await call_next(request)

    # Record latency
    latency = time.time() - start_time
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    # Record request count
    status = "success" if response.status_code < 400 else "error"
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()

    return response


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "Weather Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded() else "degraded",
        model_loaded=predictor.is_loaded() if predictor else False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WeatherFeatures):
    """
    Make a single weather prediction.

    Predicts temperature 6 hours ahead based on current weather features.
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    # Check for data drift
    drift_detected = check_drift(features)
    update_drift_ratio(drift_detected)

    try:
        # Make prediction
        prediction = predictor.predict_single(**features.model_dump())

        # Record prediction value
        PREDICTION_VALUE.observe(prediction)
        LAST_PREDICTION_TIMESTAMP.set(time.time())

        return PredictionResponse(
            predicted_temperature=round(prediction, 2),
            forecast_horizon_hours=settings.model.forecast_horizon_hours,
            model_name=settings.model.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_warning=drift_detected,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.

    Accepts multiple feature sets and returns predictions for all.
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    try:
        features_list = [f.model_dump() for f in request.features]
        predictions = predictor.predict(features_list)

        # Record metrics
        for pred in predictions:
            PREDICTION_VALUE.observe(pred)
        LAST_PREDICTION_TIMESTAMP.set(time.time())

        return BatchPredictionResponse(
            predictions=[round(p, 2) for p in predictions.tolist()],
            count=len(predictions),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "model_name": settings.model.model_name,
        "feature_count": len(predictor.feature_columns),
        "feature_columns": predictor.feature_columns,
        "target_column": predictor.target_column,
        "forecast_horizon_hours": settings.model.forecast_horizon_hours,
    }


@app.get("/model/feature-importance")
async def feature_importance():
    """Get feature importance from the model."""
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    importance = predictor.get_feature_importance()
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "feature_importance": sorted_importance,
        "top_10_features": dict(list(sorted_importance.items())[:10]),
    }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
    )
