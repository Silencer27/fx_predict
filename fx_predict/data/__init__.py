"""
Data processing module for FX prediction
"""

from fx_predict.data.dataset import FXDataset
from fx_predict.data.graph_builder import build_adjacency_matrix

__all__ = [
    "FXDataset",
    "build_adjacency_matrix",
]
