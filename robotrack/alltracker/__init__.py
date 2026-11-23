"""AllTracker (Harley et al., 2025), vendored from https://github.com/aharley/alltracker under its MIT licence (see
LICENSE in this folder): the network (net.py, blocks.py) and the utilities training uses (basic, samp, loss, data, misc).
Changes from upstream: import paths; NaN-safe probabilities in loss.py's BCE losses; a seq_name field in data.VideoData."""
from .net import Net

__all__ = ["Net"]
