"""The Lab: upload data, and an untrained model learns it from scratch on your CPU in seconds."""

from .automl import SPEEDS, Trained, train_table
from .data import DataError, profile, read_table, read_text, sniff_kind

__all__ = ["SPEEDS", "Trained", "train_table", "DataError", "profile", "read_table", "read_text", "sniff_kind"]
