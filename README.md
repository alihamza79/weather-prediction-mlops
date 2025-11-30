# 🌤️ Weather Prediction MLOps Project

A comprehensive Real-Time Predictive System (RPS) for weather forecasting, demonstrating end-to-end MLOps practices including automated data pipelines, model training, deployment, and monitoring.

## 📋 Project Overview

This project predicts temperature 6 hours ahead using weather data from OpenWeatherMap API. It showcases a complete MLOps pipeline with:

- **Data Pipeline**: Automated ETL with Apache Airflow
- **Data Quality**: Mandatory quality gates with validation checks
- **Model Training**: XGBoost with MLflow experiment tracking
- **Model Serving**: FastAPI REST API with Docker containerization
- **CI/CD**: GitHub Actions with CML for automated model comparison
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Weather Prediction MLOps                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ OpenWeather  │───▶│   Airflow    │───▶│    DVC       │                  │
│  │     API      │    │   (ETL)      │    │  (Storage)   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                             │                    │                          │
│                             ▼                    ▼                          │
│                      ┌──────────────┐    ┌──────────────┐                  │
│                      │   MLflow     │◀───│   Training   │                  │
│                      │  (Dagshub)   │    │   Script     │                  │
│                      └──────────────┘    └──────────────┘                  │
│                             │                                               │
│                             ▼                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Prometheus  │◀───│   FastAPI    │◀───│   Docker     │                  │
│  │              │    │   (API)      │    │  (Container) │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                          │
│  │   Grafana    │                                                          │
│  │ (Dashboard)  │                                                          │
│  └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **ML Framework** | XGBoost, Scikit-learn |
| **Orchestration** | Apache Airflow |
| **Experiment Tracking** | MLflow + Dagshub |
| **Data Versioning** | DVC |
| **API Framework** | FastAPI |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions, CML |
| **Monitoring** | Prometheus, Grafana |
| **Storage** | MinIO (S3-compatible) |

## 📁 Project Structure

```
weather-prediction-mlops/
├── .github/
│   └── workflows/
│       ├── ci-dev.yml          # Linting & tests (feature → dev)
│       ├── ci-test.yml         # Model training with CML (dev → test)
│       └── cd-master.yml       # Docker build & deploy (test → master)
├── airflow/
│   └── dags/
│       └── weather_etl_dag.py  # Main ETL + training DAG
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI prediction service
│   ├── data/
│   │   ├── extract.py          # OpenWeatherMap data extraction
│   │   ├── transform.py        # Feature engineering
│   │   └── quality_checks.py   # Data validation
│   ├── models/
│   │   ├── train.py            # Model training with MLflow
│   │   └── predict.py          # Prediction utilities
│   ├── utils/
│   │   └── logging.py          # Logging configuration
│   └── config.py               # Configuration management
├── monitoring/
│   ├── prometheus.yml          # Prometheus configuration
│   ├── prometheus_rules/       # Alert rules
│   └── grafana/
│       ├── provisioning/       # Datasources & dashboards config
│       └── dashboards/         # Dashboard JSON definitions
├── tests/
│   ├── test_api.py
│   ├── test_data_quality.py
│   └── test_transform.py
├── data/
│   ├── raw/                    # Raw API data
│   └── processed/              # Transformed data
├── models/                     # Saved models
├── reports/                    # Quality reports & metrics
├── docker-compose.yml          # Full stack deployment
├── Dockerfile                  # API container
├── Dockerfile.airflow          # Airflow container
├── pyproject.toml              # Dependencies
├── dvc.yaml                    # DVC pipeline
├── params.yaml                 # Model parameters
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/weather-prediction-mlops.git
cd weather-prediction-mlops

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# Copy example environment file
cp env.example .env

# Edit .env with your credentials:
# - OPENWEATHERMAP_API_KEY (get from https://openweathermap.org/api)
# - DAGSHUB_USERNAME
# - DAGSHUB_TOKEN
# - DAGSHUB_REPO_NAME
```

### 3. Initialize DVC

```bash
# Initialize DVC
dvc init

# Configure Dagshub as remote
dvc remote add -d dagshub https://dagshub.com/YOUR_USERNAME/YOUR_REPO.dvc
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user YOUR_USERNAME
dvc remote modify dagshub --local password YOUR_TOKEN
```

### 4. Run the Full Stack

```bash
# Start all services
docker-compose up -d

# Initialize Airflow (first time only)
docker-compose up airflow-init

# Check services
docker-compose ps
```

### 5. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI Docs** | http://localhost:8000/docs | - |
| **Airflow UI** | http://localhost:8080 | admin / admin |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | - |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |

## 📊 Usage

### Extract Weather Data

```python
from src.data.extract import WeatherDataExtractor

extractor = WeatherDataExtractor()
current_path, forecast_path = extractor.extract_and_save()
```

### Train Model

```python
from src.models.train import train_model

metrics = train_model()
print(f"Test RMSE: {metrics['test_rmse']:.4f}")
```

### Make Predictions

```bash
# Using the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hour_sin": 0.5,
    "hour_cos": 0.866,
    "temperature": 20.0,
    "humidity": 65.0,
    "pressure": 1013.0,
    "wind_speed": 5.0
  }'
```

### Run Tests

```bash
pytest tests/ -v
```

## 🔄 CI/CD Pipeline

### Branch Strategy

```
feature/* ──▶ dev ──▶ test ──▶ master
              │        │         │
              │        │         └── Docker build & deploy
              │        └── Model training + CML report
              └── Linting & unit tests
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `DAGSHUB_USERNAME` | Dagshub username |
| `DAGSHUB_TOKEN` | Dagshub access token |
| `DAGSHUB_REPO_NAME` | Repository name on Dagshub |
| `OPENWEATHERMAP_API_KEY` | OpenWeatherMap API key |
| `DOCKER_HUB_USERNAME` | Docker Hub username |
| `DOCKER_HUB_TOKEN` | Docker Hub access token |

## 📈 Monitoring

### Prometheus Metrics

The API exposes the following metrics at `/metrics`:

- `weather_prediction_requests_total` - Total prediction requests
- `weather_prediction_request_latency_seconds` - Request latency histogram
- `weather_prediction_value_celsius` - Prediction value distribution
- `weather_data_drift_ratio` - Ratio of out-of-distribution requests
- `weather_model_loaded` - Model load status

### Grafana Alerts

Pre-configured alerts for:

- High prediction latency (>500ms)
- Data drift detected (>10% OOD requests)
- Model not loaded
- High error rate (>10%)
- API down

## 🗂️ Data Pipeline (Airflow DAG)

The `weather_etl_training_pipeline` DAG runs every 6 hours:

1. **Extract**: Fetch weather data from OpenWeatherMap API
2. **Validate**: Run quality checks (fails DAG if checks fail)
3. **Transform**: Engineer features (lag, rolling, time encodings)
4. **Train**: Train XGBoost model with MLflow tracking
5. **Log**: Store artifacts in MLflow/Dagshub

## 🧪 Running Locally (Development)

### Without Docker

```bash
# Start the API
uvicorn src.api.app:app --reload --port 8000

# In another terminal, run Airflow standalone
export AIRFLOW_HOME=$(pwd)/airflow
airflow standalone
```

### With Docker (Recommended)

```bash
# Build and start
docker-compose up --build

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## 📝 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/model/info` | Model information |
| GET | `/model/feature-importance` | Feature importance |

### Example Request

```json
POST /predict
{
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
  "visibility": 10000.0
}
```

### Example Response

```json
{
  "predicted_temperature": 21.5,
  "forecast_horizon_hours": 6,
  "model_name": "weather_predictor",
  "timestamp": "2024-01-15T10:30:00Z",
  "drift_warning": false
}
```

## 🤝 Contributing

1. Create a feature branch from `dev`
2. Make changes and add tests
3. Submit PR to `dev` (requires 1 approval)
4. After merge, PR from `dev` to `test` triggers model training
5. If CML report passes, PR to `master` for deployment

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Team

MLOps Case Study Project - Built with ❤️ for learning MLOps best practices.

