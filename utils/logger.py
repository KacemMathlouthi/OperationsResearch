import functools
import logging
import sys
import os
from dotenv import load_dotenv
from uvicorn.logging import ColourizedFormatter
from typing import Any, Callable

load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Mapping of string levels to logging constants
LOG_LEVEL_MAPPING = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Set the actual logging level
LOG_LEVEL = LOG_LEVEL_MAPPING.get(LOG_LEVEL, logging.INFO)  # Default to INFO if invalid


# Custom colorized formatter to apply colors specifically to log levels
class CustomColourizedFormatter(ColourizedFormatter):
    def format(self, record: logging.LogRecord) -> str:
        # Define color mappings for different log levels
        level_color_map = {
            "DEBUG": "\033[34m",  # Blue
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[41m",  # Red background
        }

        # Reset color
        reset = "\033[0m"

        # Apply color to the log level name
        record.levelname = (
            f"{level_color_map.get(record.levelname, '')}{record.levelname}{reset}"
        )

        # Format the log message using the parent class's format method
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Creates a logger object that reads its log level from .env.

    Args:
        name (str): name given to the logger

    Returns:
        logging.Logger: logger object to be used for logging
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)  # Set level based on .env file

    # Prevent adding multiple handlers if already exists
    if not logger.hasHandlers():
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(LOG_LEVEL)  # Set handler level

        # Create a custom formatter with colored log levels
        formatter = CustomColourizedFormatter(
            "{asctime} | {levelname:<8} | {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_colors=True,
        )

        # Add formatter to the handler
        ch.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(ch)

    return logger


# Logger decorator implementation
def log_function_call(logger: logging.Logger) -> Callable:
    """A decorator that logs the function calls and results.

    Args:
        logger (logging.Logger): The logger instance to use for logging.

    Returns:
        Callable: A wrapper function that logs the execution details.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Log the function call with arguments
            logger.debug(
                f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}"
            )
            result = func(*args, **kwargs)
            # Log the function result
            logger.debug(f"{func.__name__} returned {result}")
            return result

        return wrapper

    return decorator


logger = logging.getLogger(__name__)
