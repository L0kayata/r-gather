import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import random
import time


from r_gather.data_structures import Point
from r_gather.r_gather import compute_r_gather
from r_gather.visualization import visualize_clustering, visualize_clustering_with_stats


def test_visualization():
    """Generate 100 random points, run r-Gather, and visualize"""
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Generate 100 random points
    
    np.random.seed(random.randint(0, 10000))
    points = [
        Point(id=i, coordinate=np.random.uniform(0, 100, 2))
        for i in range(100)
    ]
    
    """
    points = [
        Point(id=0, coordinate=np.array([1, 1])),
        Point(id=3, coordinate=np.array([1, 2])),
        Point(id=2, coordinate=np.array([2, 1])),
        Point(id=1, coordinate=np.array([2, 2])),
    ]
    """
    """
    points = [
        Point(id=0, coordinate=np.array([10, 10])),
        Point(id=1, coordinate=np.array([15, 15])),
        Point(id=2, coordinate=np.array([20, 20])),
        Point(id=3, coordinate=np.array([25, 25])),
        Point(id=4, coordinate=np.array([30, 30])),
        Point(id=5, coordinate=np.array([35, 35])),
    ]
    """
    """
    points = [
        Point(id=0, coordinate=np.array([3, 3])),
        Point(id=1, coordinate=np.array([5, 5])),
        Point(id=2, coordinate=np.array([1, 1])),
        Point(id=3, coordinate=np.array([6, 6])),
        Point(id=4, coordinate=np.array([4, 4])),
        Point(id=5, coordinate=np.array([2, 2])),
    ]
    """
    
    
    
    # Run r-Gather algorithm
    r = 5
    clusters = compute_r_gather(points, r)
    
    # Generate visualizations
    fig1, ax1 = visualize_clustering(points, clusters, r)
    plt.savefig('output/clustering.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, axes2 = visualize_clustering_with_stats(points, clusters, r)
    plt.savefig('output/clustering_stats.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)


if __name__ == '__main__':
    start_time = time.time()
    test_visualization()
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds")