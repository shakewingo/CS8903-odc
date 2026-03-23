import logging
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

def get_logger(name: str = "cs8903", level: str = "INFO", log_file: str = None, stream: bool = True) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name:  Logger name (shows in log output). Use __name__ in scripts,
               or a descriptive string like "landcover" in notebooks.
        level: Logging level — "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        log_file: Optional file path to also write logs to.
        stream: Whether to log to stdout. Set False to log only to file.
    """
    logger = logging.getLogger(name)
    # Clear existing handlers to allow reconfiguration (e.g. notebook re-runs)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

def minmax_normalize(data: Dict) -> Dict:
    """MinMax normalization for Dict of values"""
    scaler = MinMaxScaler()
    
    keys = list(data.keys())
    values_2d = np.array(list(data.values())).reshape(-1, 1)
    
    scaled_values = scaler.fit_transform(values_2d).flatten()
    
    return {k: round(float(v), 4) for k, v in zip(keys, scaled_values)}
