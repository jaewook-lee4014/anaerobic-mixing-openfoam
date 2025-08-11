"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose output
        log_file: Optional log file path
        log_level: Logging level string
    """
    # Determine log level
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Set specific logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured (level: {logging.getLevelName(level)})")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)