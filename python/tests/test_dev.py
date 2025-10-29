import numpy as np
from r_gather.data_structures import Point, Cluster
from r_gather.distance_matrix import distance_matrix

def test_distance_matrix():
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 4.0])),
        Point(id=2, coordinate=np.array([6.0, 8.0])),
    ]
    
    dist_matrix = distance_matrix(points)
    
    expected = np.array([
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 5.0],
        [10.0, 5.0, 0.0],
    ])
    
    assert isinstance(dist_matrix, np.ndarray), "Distance matrix should be a numpy ndarray."
    assert np.allclose(dist_matrix, expected), "Distance matrix computation is incorrect."
    
if __name__ == '__main__':
    test_distance_matrix()
    print("All tests passed.")