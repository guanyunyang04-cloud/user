"""Small, current technical-research pipeline.

The package intentionally keeps one data contract and separates forecasting
from portfolio replay.  Legacy experiments remain under ``path_policy`` only
as historical references while this package becomes the active research path.
"""

from .data import FEATURE_FAMILIES, TechnicalData, load_data

__all__ = ["FEATURE_FAMILIES", "TechnicalData", "load_data"]
