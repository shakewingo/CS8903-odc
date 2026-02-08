import logging
import sys


def get_logger(name: str = "cs8903", level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name:  Logger name (shows in log output). Use __name__ in scripts,
               or a descriptive string like "landcover" in notebooks.
        level: Logging level — "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
