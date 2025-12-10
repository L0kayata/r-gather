# r_gather.py
import numpy as np
from .data_structures import Point, Cluster
from .distance_matrix import compute_distance_matrix
from .flow_network import flow_network_verification

def compute_r_gather(points: list[Point], r: float) -> list[Cluster]:

    # Compute distance matrix and candidate radii
    distance_matrix = compute_distance_matrix(points)
    candidate_radii = np.unique(distance_matrix / 2)

    # Find the smallest R in candidate radii
    for R in candidate_radii:

        if R == 0:
            continue

        # Condition 1
        if not check_condition_1(distance_matrix, R, r):
            continue

        # Condition 2
        success, clusters = check_condition_2(points, distance_matrix, R, r)
        if success:
            return clusters

    # No valid clustering found
    return []

def compute_r_gather_binary_search(points: list[Point], r: float) -> list[Cluster]:
    # Compute distance matrix and candidate radii
    distance_matrix = compute_distance_matrix(points)
    candidate_radii = np.unique(distance_matrix / 2)
    candidate_radii = candidate_radii[candidate_radii > 0]  # Remove 0
    
    if len(candidate_radii) == 0:
        return []
    
    # Binary search for the smallest R
    left, right = 0, len(candidate_radii) - 1
    result_R = None
    result_clusters = []
    
    while left <= right:
        mid = (left + right) // 2
        R = candidate_radii[mid]
        
        # Check Condition 1
        if not check_condition_1(distance_matrix, R, r):
            # R is too small, search right half
            left = mid + 1
            continue
        
        # Check Condition 2
        success, clusters = check_condition_2(points, distance_matrix, R, r)
        
        if success:
            # Found a valid R, try to find smaller one
            result_R = R
            result_clusters = clusters
            right = mid - 1
        else:
            # R doesn't work, need larger R
            left = mid + 1
    
    return result_clusters

def check_condition_1(distance_matrix: np.ndarray, R: float, r: int) -> bool:
    # Each point p in the candidate radii should have
    # at least r − 1 other points within distance 2R of p.
    neighbor_counts = np.sum(distance_matrix <= 2 * R, axis = 1)
    return np.all(neighbor_counts >= r)

def check_condition_1_grid(distance_matrix: np.ndarray, R: float, r: int, 
                      points: list = None) -> bool:
    """
    Check Condition 1: Each point should have at least r-1 other points 
    within distance 2R (including itself, so total >= r).
    
    Grid-based optimization: Divide space into cells of size 2R, 
    only check neighbors in adjacent cells.
    
    Args:
        distance_matrix: Pre-computed distance matrix
        R: Current radius value
        r: Minimum cluster size
        points: Optional list of points for grid-based optimization
    
    Returns:
        True if condition 1 is satisfied, False otherwise
    """
    n = distance_matrix.shape[0]
    
    # Basic validation
    if n < r:
        return False
    
    if R <= 0:
        return False
    
    # If points not provided, use original vectorized method
    if points is None:
        neighbor_counts = np.sum(distance_matrix <= 2 * R, axis=1)
        return np.all(neighbor_counts >= r)
    
    # Grid-based optimization
    from itertools import product
    
    cell_size = 2 * R
    d = len(points[0].coordinate)
    
    # For high dimensions (>5), 3^d neighbor cells becomes too large
    # Fall back to original method
    if d > 5:
        neighbor_counts = np.sum(distance_matrix <= 2 * R, axis=1)
        return np.all(neighbor_counts >= r)
    
    # Map point coordinate to grid cell index
    def get_cell_index(coord):
        return tuple(int(np.floor(c / cell_size)) for c in coord)
    
    # Build grid: cell_index -> list of point indices
    grid = {}
    for i, p in enumerate(points):
        cell = get_cell_index(p.coordinate)
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(i)
    
    # Pre-compute all neighbor offsets: {-1, 0, 1}^d
    offsets = list(product([-1, 0, 1], repeat=d))
    
    threshold = 2 * R
    
    # Check each point
    for i, p in enumerate(points):
        cell = get_cell_index(p.coordinate)
        neighbor_count = 0
        
        # Check all neighboring cells (including self)
        for offset in offsets:
            neighbor_cell = tuple(cell[k] + offset[k] for k in range(d))
            if neighbor_cell in grid:
                for j in grid[neighbor_cell]:
                    # Use pre-computed distance matrix
                    if distance_matrix[i][j] <= threshold:
                        neighbor_count += 1
                        # Early exit: found enough neighbors
                        if neighbor_count >= r:
                            break
                if neighbor_count >= r:
                    break
        
        # Early exit: this point doesn't have enough neighbors
        if neighbor_count < r:
            return False
    
    return True

def check_condition_2(points: list[Point], distance_matrix: np.ndarray, R: float, r: int):
    """
    Check condition 2: Initial clustering and flow network verification
    
    Returns:
        (success, clusters) - success is True if condition satisfied, 
                             clusters is the resulting clustering
    """
    n = len(points)
    
    # Phase 2.1: Initial clustering construction
    centers = initial_clustering(distance_matrix, R, r)
    
    if not centers:
        return False, []
    
    # Phase 2.2: Flow network verification
    success, assignments = flow_network_verification(n, centers, distance_matrix, R, r)
    
    if not success:
        return False, []
    
    # Build final clusters
    clusters = build_clusters_from_assignments(points, centers, assignments, distance_matrix)
    return True, clusters

def initial_clustering(distance_matrix: np.ndarray, R: float, r: int) -> list[int]:
    """
    Phase 2.1: Initial clustering construction
    
    Algorithm (from paper):
    1. All nodes start unmarked
    2. Select arbitrary point p as center
    3. If there are at least r unmarked points within 2R of p (including p),
       form cluster and mark all points within 2R
    4. Repeat until cannot continue
    5. All points must be marked for success
    
    Returns:
        List of center indices, or empty list if failed
    """
    n = distance_matrix.shape[0]
    marked = np.zeros(n, dtype=bool)
    centers = []
    
    while not np.all(marked):
        # Find an unmarked point to potentially be a center
        unmarked_indices = np.where(~marked)[0]
        
        # Try to find a point that can be a center
        found_center = False
        for p_idx in unmarked_indices:
            # Check if there are at least r unmarked points within 2R of p
            # (including p itself)
            within_2R = distance_matrix[p_idx] <= 2 * R
            # unmarked_within_2R = within_2R & ~marked
            
            if np.sum(within_2R) >= r:
                # Form a cluster with center at p
                centers.append(p_idx)
                # Mark all points within 2R of p (including p)
                marked[within_2R] = True
                found_center = True
                break
        
        # If no center can be formed, stop
        if not found_center:
            break
    
    # Check if all points are marked
    if not np.all(marked):
        return []
    
    return centers



def build_clusters_from_assignments(points: list[Point], centers: list[int], 
                                   assignments: dict, distance_matrix: np.ndarray):
    """
    Build final clusters from assignments
    """
    clusters = []
    
    # Group points by center
    center_to_points = {}
    for point_idx, center_idx in assignments.items():
        if center_idx not in center_to_points:
            center_to_points[center_idx] = []
        center_to_points[center_idx].append(points[point_idx])
    
    # Create clusters
    cluster_id = 0
    for center_idx, member_points in center_to_points.items():
        # Calculate actual radius for this cluster (max distance from center to any member)
        max_dist = 0
        for member in member_points:
            dist = distance_matrix[member.id][center_idx]
            max_dist = max(max_dist, dist)
        
        cluster = Cluster(
            id=cluster_id,
            coordinate=points[center_idx].coordinate,
            members=member_points,
            radius=max_dist
        )
        clusters.append(cluster)
        cluster_id += 1
    
    return clusters






