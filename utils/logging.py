"""
Logging utility for the bioinformatics-prompts package.

This module provides a configured logger using the loguru package.
"""
from pathlib import Path
import sys
from loguru import logger


def setup_logger(log_file: str = None, log_level: str = "INFO"):
    """
    Configure and set up the logger for the bioinformatics-prompts package.
    
    Args:
        log_file: Path to log file. If None, logs only to console.
        log_level: Minimum log level to display. Options: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.
    
    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger with specified level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file logger if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level
        )
    
    return logger


# Export configured logger for use throughout the package
default_logger = setup_logger()