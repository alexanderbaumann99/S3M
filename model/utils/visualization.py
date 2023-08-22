import numpy as np


def visu(vertices: np.ndarray) -> np.ndarray:
    """Creates a color map based on a point cloud"""
    min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(
        vertices, axis=0, keepdims=True
    )
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap
