import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import random


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
        Point(id=i, coordinate=np.random.uniform(0, 1000, 2))
        for i in range(1000)
    ]
    
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
    test_visualization()