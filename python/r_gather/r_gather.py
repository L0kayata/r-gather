# r_gather.py
import numpy as np
from .data_structures import Point, Cluster
from .distance_matrix import compute_distance_matrix

def compute_r_gather(points: list[Point], r: float) -> list[Cluster]:

    # Compute distance matrix and candidate radii
    distance_matrix = compute_distance_matrix(points)
    candidate_radii = np.unique(distance_matrix / 2)

    # Find the smallest R in candidate radii
    for R in candidate_radii:
        
        # Condition 1
        if not check_condition_1(distance_matrix, R, r):
            return False

        # Condition 2




    #clusters = []
    #return clusters

def check_condition_1(distance_matrix: np.ndarray, R: float, r: int) -> bool:
    # Each point p in the candidate radii should have
    # at least r − 1 other points within distance 2R of p.
    neighbor_counts = np.sum(distance_matrix <= 2 * R, axis = 1)
    return np.all(neighbor_counts >= r)
