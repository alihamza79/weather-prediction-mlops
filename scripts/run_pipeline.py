#!/usr/bin/env python3
"""
Run the complete ML pipeline manually (without Airflow).

This script is useful for:
- Local development and testing
- CI/CD pipelines
- Quick experiments
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.data.extract import WeatherDataExtractor
from src.data.quality_checks import validate_weather_data, DataQualityError
from src.data.transform import WeatherDataTransformer
from src.models.train import train_model


def run_extraction(use_sample: bool = False) -> bool:
    """Run data extraction step."""
    logger.info("=" * 60)
    logger.info("Step 1: Data Extraction")
    logger.info("=" * 60)
    
    if use_sample:
        logger.info("Using sample data (skipping API call)")
        from scripts.generate_sample_data import generate_sample_raw_data
        generate_sample_raw_data(n_days=3)
        return True
    
    try:
        extractor = WeatherDataExtractor()
        current_path, forecast_path = extractor.extract_and_save()
        logger.info(f"Extracted current weather to: {current_path}")
        logger.info(f"Extracted forecast to: {forecast_path}")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def run_validation() -> bool:
    """Run data validation step."""
    logger.info("=" * 60)
    logger.info("Step 2: Data Quality Validation")
    logger.info("=" * 60)
    
    try:
        extractor = WeatherDataExtractor()
        df = extractor.load_historical_data()
        
        if df.empty:
            logger.error("No data found for validation")
            return False
        
        report = validate_weather_data(df, fail_on_error=True)
        
        # Save report
        report_path = REPORTS_DIR / "quality_metrics.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Quality report saved to: {report_path}")
        logger.info("✅ Data quality validation PASSED")
        return True
        
    except DataQualityError as e:
        logger.error(f"Data quality validation FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


def run_transformation() -> Path:
    """Run data transformation step."""
    logger.info("=" * 60)
    logger.info("Step 3: Data Transformation")
    logger.info("=" * 60)
    
    try:
        extractor = WeatherDataExtractor()
        df = extractor.load_historical_data()
        
        transformer = WeatherDataTransformer()
        df_transformed = transformer.transform(df, create_target=True)
        
        data_path = transformer.save_processed_data(df_transformed)
        
        logger.info(f"Transformed {len(df)} rows → {len(df_transformed)} rows")
        logger.info(f"Features: {len(df_transformed.columns)}")
        logger.info(f"Saved to: {data_path}")
        
        return data_path
        
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        return None


def run_training(data_path: Path = None) -> dict:
    """Run model training step."""
    logger.info("=" * 60)
    logger.info("Step 4: Model Training")
    logger.info("=" * 60)
    
    try:
        metrics = train_model(data_path=data_path)
        
        logger.info("Training metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data instead of API",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step (not recommended)",
    )
    args = parser.parse_args()
    
    logger.info("🚀 Starting Weather Prediction ML Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Extraction
    if not args.skip_extraction:
        if not run_extraction(use_sample=args.sample):
            logger.error("Pipeline failed at extraction step")
            sys.exit(1)
    else:
        logger.info("Skipping extraction step")
    
    # Step 2: Validation
    if not args.skip_validation:
        if not run_validation():
            logger.error("Pipeline failed at validation step")
            sys.exit(1)
    else:
        logger.warning("Skipping validation step (not recommended for production)")
    
    # Step 3: Transformation
    data_path = run_transformation()
    if data_path is None:
        logger.error("Pipeline failed at transformation step")
        sys.exit(1)
    
    # Step 4: Training
    metrics = run_training(data_path)
    if metrics is None:
        logger.error("Pipeline failed at training step")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🎉 Pipeline completed successfully!")
    logger.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()

