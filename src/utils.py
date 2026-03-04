import logging
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

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

def minmax_normalize(data: Dict) -> Dict:
    """MinMax normalization for Dict of values"""
    scaler = MinMaxScaler()
    
    keys = list(data.keys())
    values_2d = np.array(list(data.values())).reshape(-1, 1)
    
    scaled_values = scaler.fit_transform(values_2d).flatten()
    
    return {k: round(float(v), 4) for k, v in zip(keys, scaled_values)}
