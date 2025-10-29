import numpy as np
from r_gather.data_structures import Point, Cluster
from r_gather.distance_matrix import compute_distance_matrix

def test_distance_matrix():
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 4.0])),
        Point(id=2, coordinate=np.array([6.0, 8.0])),
    ]
    distance_matrix = compute_distance_matrix(points)
    expected = np.array([
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 5.0],
        [10.0, 5.0, 0.0],
    ])
    assert np.allclose(distance_matrix, expected), "Distance matrix computation is incorrect."

def test_distance_matrix_type():
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 4.0])),
        Point(id=2, coordinate=np.array([6.0, 8.0])),
    ]
    distance_matrix = compute_distance_matrix(points)
    assert isinstance(distance_matrix, np.ndarray), "Distance matrix should be a numpy ndarray."

def test_distance_matrix_half():
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 4.0])),
        Point(id=2, coordinate=np.array([6.0, 8.0])),
    ]
    distance_matrix = compute_distance_matrix(points)
    half_distance_matrix = distance_matrix / 2
    expected_half = np.array([
        [0.0, 2.5, 5.0],
        [2.5, 0.0, 2.5],
        [5.0, 2.5, 0.0],
    ])
    assert np.allclose(half_distance_matrix, expected_half), "Half distance matrix computation is incorrect."

def test_distance_matrix_half_unique():
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 4.0])),
        Point(id=2, coordinate=np.array([6.0, 8.0])),
    ]
    distance_matrix = compute_distance_matrix(points)
    half_distance_matrix = distance_matrix / 2
    unique_half = np.unique(half_distance_matrix)
    expected_unique_half = np.array([0.0, 2.5, 5.0])
    assert np.allclose(unique_half, expected_unique_half), "Unique half distance matrix computation is incorrect."

if __name__ == '__main__':
    test_distance_matrix()
    test_distance_matrix_type()
    test_distance_matrix_half()
    test_distance_matrix_half_unique()
    print("All distance matrix tests passed.")