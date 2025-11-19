"""
Logging configuration for GenEEG project.

This module provides centralized logging configuration with support for:
- Console and file logging
- Different log levels for different modules
- Colored console output (optional)
- Rotation of log files

Usage:
    from utils.logger import setup_logging, get_logger
    
    # Setup logging once at application start
    setup_logging(log_level='INFO', log_file='geneeg.log')
    
    # Get logger in each module
    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.debug("This is a debug message")
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Log level colors
    DEBUG = '\033[36m'     # Cyan
    INFO = '\033[32m'      # Green
    WARNING = '\033[33m'   # Yellow
    ERROR = '\033[31m'     # Red
    CRITICAL = '\033[35m'  # Magenta


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color to console output.
    
    Only applies colors to console output, not to file logs.
    """
    
    COLORS = {
        'DEBUG': LogColors.DEBUG,
        'INFO': LogColors.INFO,
        'WARNING': LogColors.WARNING,
        'ERROR': LogColors.ERROR,
        'CRITICAL': LogColors.CRITICAL,
    }
    
    def format(self, record):
        """Format log record with color."""
        # Save original levelname
        levelname = record.levelname
        
        # Add color
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname
        
        return result


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    use_colors: bool = True,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration for the entire application.
    
    Args:
        log_level: Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (optional)
        log_dir: Directory for log files (default: './logs')
        use_colors: Use colored output for console (default: True)
        format_string: Custom format string (optional)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if use_colors:
        console_formatter = ColoredFormatter(format_string, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(format_string, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file is not None:
        # Create log directory if needed
        if log_dir is None:
            log_dir = './logs'
        
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = log_path / f"{Path(log_file).stem}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        
        # File format (no colors)
        file_formatter = logging.Formatter(format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_filename}")
    
    # Log the configuration
    logging.info(f"Logging initialized - Level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_module_log_level(module_name: str, level: str):
    """
    Set log level for a specific module.
    
    Useful for reducing verbosity of specific modules.
    
    Args:
        module_name: Name of the module
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level.upper()))
    logging.info(f"Set {module_name} log level to {level}")


def disable_module_logging(module_name: str):
    """
    Disable logging for a specific module.
    
    Args:
        module_name: Name of the module
    """
    logging.getLogger(module_name).disabled = True
    logging.info(f"Disabled logging for {module_name}")


# Predefined configurations
def setup_training_logging(output_dir: str = './outputs', log_level: str = 'INFO'):
    """
    Setup logging for training runs.
    
    Args:
        output_dir: Output directory for logs
        log_level: Log level
    """
    setup_logging(
        log_level=log_level,
        log_file='training.log',
        log_dir=Path(output_dir) / 'logs',
        use_colors=True
    )


def setup_testing_logging(log_level: str = 'DEBUG'):
    """
    Setup logging for testing (console only, debug level).
    
    Args:
        log_level: Log level (default: DEBUG)
    """
    setup_logging(
        log_level=log_level,
        log_file=None,
        use_colors=True
    )


def setup_production_logging(log_dir: str = './logs', log_level: str = 'WARNING'):
    """
    Setup logging for production (file only, minimal console).
    
    Args:
        log_dir: Directory for log files
        log_level: Log level (default: WARNING)
    """
    setup_logging(
        log_level=log_level,
        log_file='geneeg.log',
        log_dir=log_dir,
        use_colors=False
    )


# Context manager for temporary log level
class LogLevelContext:
    """
    Context manager to temporarily change log level.
    
    Usage:
        with LogLevelContext('DEBUG'):
            # Code here will see DEBUG logs
            logger.debug("This will be shown")
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        """
        Initialize context manager.
        
        Args:
            level: Temporary log level
            logger_name: Specific logger name (None = root logger)
        """
        self.level = level
        self.logger_name = logger_name
        self.original_level = None
    
    def __enter__(self):
        """Enter context - set new level."""
        if self.logger_name is None:
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(self.logger_name)
        
        self.original_level = logger.level
        logger.setLevel(getattr(logging, self.level.upper()))
        
        return logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original level."""
        if self.logger_name is None:
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(self.logger_name)
        
        logger.setLevel(self.original_level)


if __name__ == "__main__":
    # Test logging configuration
    print("Testing logging configuration...\n")
    
    # Test 1: Basic console logging
    print("1. Basic console logging with colors:")
    setup_logging(log_level='DEBUG', use_colors=True)
    
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test 2: File logging
    print("\n2. File logging:")
    setup_logging(log_level='INFO', log_file='test.log', log_dir='./test_logs')
    logger.info("This message goes to both console and file")
    
    # Test 3: Module-specific log levels
    print("\n3. Module-specific log levels:")
    test_logger = get_logger("test_module")
    test_logger.debug("Debug from test_module (should not show)")
    set_module_log_level("test_module", "DEBUG")
    test_logger.debug("Debug from test_module (should show now)")
    
    # Test 4: Context manager
    print("\n4. Temporary log level with context manager:")
    logger.debug("Debug outside context (should not show)")
    with LogLevelContext('DEBUG'):
        logger.debug("Debug inside context (should show)")
    logger.debug("Debug outside context again (should not show)")
    
    print("\n✅ Logging test complete!")
