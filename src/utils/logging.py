"""Logging configuration."""

import sys
from pathlib import Path

from loguru import logger

from src.config import LOGS_DIR


def setup_logging(
    level: str = "INFO",
    log_file: bool = True,
    log_dir: Path = LOGS_DIR,
) -> None:
    """
    Configure loguru logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Whether to write logs to file
        log_dir: Directory for log files
    """
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    if log_file:
        # Add file handler
        log_path = log_dir / "app_{time:YYYY-MM-DD}.log"
        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="30 days",
            compression="gz",
        )


def get_logger(name: str = None):
    """Get a logger instance."""
    return logger.bind(name=name) if name else logger
