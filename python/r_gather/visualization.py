# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from r_gather.data_structures import Point, Cluster

def visualize_clustering(points: list[Point], clusters: list[Cluster], 
                         r: int, title: str = "r-Gather Clustering Result"):
    """
    Visualize the r-Gather clustering result
    
    Args:
        points: List of all points
        clusters: List of clusters
        r: Minimum cluster size parameter
        title: Title of the plot
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if len(clusters) == 0:
        ax.text(0.5, 0.5, 'No valid clustering found', 
               ha='center', va='center', fontsize=16, color='red')
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig, ax
    
    # Generate distinct colors for each cluster
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
    
    # Plot each cluster
    for idx, cluster in enumerate(clusters):
        color = colors[idx]
        
        # Plot cluster members
        member_coords = np.array([m.coordinate for m in cluster.members])
        ax.scatter(member_coords[:, 0], member_coords[:, 1], 
                  c=[color], s=100, alpha=0.6, 
                  label=f'Cluster {cluster.id} (n={cluster.size()})')
        
        # Plot cluster center with star marker
        center = cluster.coordinate
        ax.scatter(center[0], center[1], 
                  c=[color], s=300, marker='*', 
                  edgecolors='black', linewidths=2, zorder=5)
        
        # Draw circle representing cluster radius
        circle = plt.Circle((center[0], center[1]), 
                           cluster.radius, 
                           color=color, fill=False, 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        
        # Add cluster ID label near center
        ax.annotate(f'C{cluster.id}', 
                   xy=(center[0], center[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{title}\n(r={r}, Total Points={len(points)}, Clusters={len(clusters)})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, ax


def visualize_clustering_with_stats(points: list[Point], clusters: list[Cluster], 
                                   r: int, title: str = "r-Gather Clustering Result"):
    """
    Visualize the r-Gather clustering result with detailed statistics
    
    Args:
        points: List of all points
        clusters: List of clusters
        r: Minimum cluster size parameter
        title: Title of the plot
    
    Returns:
        fig, (ax1, ax2): Matplotlib figure and axes objects
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    if len(clusters) == 0:
        ax1.text(0.5, 0.5, 'No valid clustering found', 
                ha='center', va='center', fontsize=16, color='red')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax2.text(0.5, 0.5, 'No statistics available', 
                ha='center', va='center', fontsize=16, color='red')
        return fig, (ax1, ax2)
    
    # Left plot: Clustering visualization
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
    
    for idx, cluster in enumerate(clusters):
        color = colors[idx]
        
        # Plot cluster members
        member_coords = np.array([m.coordinate for m in cluster.members])
        ax1.scatter(member_coords[:, 0], member_coords[:, 1], 
                   c=[color], s=100, alpha=0.6, 
                   label=f'Cluster {cluster.id} (n={cluster.size()})')
        
        # Plot cluster center
        center = cluster.coordinate
        ax1.scatter(center[0], center[1], 
                   c=[color], s=300, marker='*', 
                   edgecolors='black', linewidths=2, zorder=5)
        
        # Draw circle representing cluster radius
        circle = plt.Circle((center[0], center[1]), 
                           cluster.radius, 
                           color=color, fill=False, 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax1.add_patch(circle)
        
        # Add cluster ID label
        ax1.annotate(f'C{cluster.id}', 
                    xy=(center[0], center[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'{title}\n(r={r}, Points={len(points)}, Clusters={len(clusters)})', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Right plot: Statistics
    cluster_sizes = [c.size() for c in clusters]
    cluster_radii = [c.radius for c in clusters]
    max_radius = max(cluster_radii)
    
    # Create twin axis for dual y-axis
    ax2_twin = ax2.twinx()
    
    x_pos = np.arange(len(clusters))
    width = 0.35
    
    # Bar chart for cluster sizes and radii
    bars1 = ax2.bar(x_pos - width/2, cluster_sizes, width, 
                    label='Cluster Size', color='steelblue', alpha=0.8)
    bars2 = ax2_twin.bar(x_pos + width/2, cluster_radii, width, 
                         label='Cluster Radius', color='coral', alpha=0.8)
    
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Cluster Size', fontsize=12, color='steelblue')
    ax2_twin.set_ylabel('Cluster Radius', fontsize=12, color='coral')
    ax2.set_title('Cluster Statistics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'C{c.id}' for c in clusters])
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='coral')
    
    # Add horizontal line for minimum cluster size r
    ax2.axhline(y=r, color='red', linestyle='--', linewidth=2, 
               label=f'Min size (r={r})', alpha=0.7)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add text box with summary statistics
    stats_text = (
        f"Summary Statistics:\n"
        f"─────────────────\n"
        f"Total Points: {len(points)}\n"
        f"Total Clusters: {len(clusters)}\n"
        f"Min Cluster Size (r): {r}\n"
        f"Max Cluster Radius: {max_radius:.2f}\n"
        f"Avg Cluster Size: {np.mean(cluster_sizes):.2f}\n"
        f"Avg Cluster Radius: {np.mean(cluster_radii):.2f}\n"
        f"Min Cluster Size: {min(cluster_sizes)}\n"
        f"Max Cluster Size: {max(cluster_sizes)}"
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    return fig, (ax1, ax2)