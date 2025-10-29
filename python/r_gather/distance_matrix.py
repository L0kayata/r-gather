# distance_matrix.py
import numpy as np
from .data_structures import Point

# compute distance matrix for a set of points
def distance_matrix(points: list[Point]) -> np.ndarray:
    
    # double loop
    """
    n = len(points)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(
                points[i].coordinate - points[j].coordinate
            )
            matrix[i, j] = dist
            matrix[j, i] = dist
    
    return matrix
    """
    
    # vectorized O(n²m)/O(n²m)
    coords = np.array([p.coordinate for p in points])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    matrix = np.linalg.norm(diff, axis=-1)
    return matrix