"""Logging configuration for Story Factory."""

import logging
import sys
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log file location
DEFAULT_LOG_FILE = Path(__file__).parent.parent / "logs" / "story_factory.log"


class ContextFilter(logging.Filter):
    """Add context information to log records."""

    def __init__(self) -> None:
        super().__init__()
        self.correlation_id: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record if available."""
        if self.correlation_id:
            record.correlation_id = self.correlation_id
        else:
            record.correlation_id = "-"
        return True


# Global context filter instance
_context_filter = ContextFilter()


def setup_logging(level: str = "INFO", log_file: str | None = "default") -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: File path for logs. "default" uses logs/story_factory.log,
                  None disables file logging.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter with correlation ID
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(correlation_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with context filter
    # Note: Filter must be on HANDLERS, not logger, for child logger records
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_context_filter)
    root_logger.addHandler(console_handler)

    # File handler - always enabled by default with rotation
    if log_file == "default":
        log_path = DEFAULT_LOG_FILE
    elif log_file:
        log_path = Path(log_file)
    else:
        log_path = None

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Custom handler that flushes after every write
        class FlushingRotatingFileHandler(RotatingFileHandler):
            """RotatingFileHandler that flushes immediately after each log."""

            def emit(self, record: logging.LogRecord) -> None:
                super().emit(record)
                self.flush()

        # Use rotating handler with immediate flush
        # Max 10MB per file, keep 5 backup files (50MB total)
        file_handler = FlushingRotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",  # 10MB
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_context_filter)
        root_logger.addHandler(file_handler)

        # Log the log file location
        root_logger.info(f"Logging to file: {log_path} (max 10MB, 5 backups)")

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("nicegui").setLevel(logging.WARNING)


@contextmanager
def log_context(correlation_id: str | None = None) -> Generator[str]:
    """Context manager for setting correlation ID in logs.

    Args:
        correlation_id: Optional correlation ID. If not provided, generates a new UUID.

    Yields:
        The correlation ID being used.

    Example:
        with log_context("request-123"):
            logger.info("Processing request")  # Will include correlation_id in log
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]

    old_id = _context_filter.correlation_id
    _context_filter.correlation_id = correlation_id
    try:
        yield correlation_id
    finally:
        _context_filter.correlation_id = old_id


@contextmanager
def log_performance(logger: logging.Logger, operation: str) -> Generator[None]:
    """Context manager for logging operation performance.

    Args:
        logger: Logger instance to use
        operation: Name of the operation being timed

    Example:
        with log_performance(logger, "story_generation"):
            generate_story()  # Will log duration after completion
    """
    start_time = time.time()
    logger.info(f"{operation}: Starting")
    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{operation}: Failed after {duration:.2f}s - {e}")
        raise
    else:
        duration = time.time() - start_time
        logger.info(f"{operation}: Completed in {duration:.2f}s")
