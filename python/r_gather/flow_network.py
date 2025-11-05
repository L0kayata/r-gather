# flow_network.py
import numpy as np
import networkx as nx

def build_flow_network(n: int, centers: list[int], 
                      distance_matrix: np.ndarray, R: float, r: int) -> nx.DiGraph:
    """
    Build flow network for verification
    
    The network structure:
    - Source s
    - Centers C (represented as center_{idx})
    - Points V (represented as point_{idx})
    - Sink t
    
    Edges:
    - s -> c (capacity r) for each c in C
    - c -> v (capacity 1) if distance(c,v) <= 2R
    - v -> t (capacity 1) for each v in V
    
    Args:
        n: Number of points
        centers: List of center indices
        distance_matrix: Distance matrix between points
        R: Current radius value
        r: Minimum cluster size
    
    Returns:
        G: The flow network as a DiGraph
    """
    G = nx.DiGraph()
    
    # Add source and sink nodes
    G.add_node('source')
    G.add_node('sink')
    
    # Add center nodes and edges from source
    for center_idx in centers:
        center_node = f'center_{center_idx}'
        G.add_node(center_node)
        G.add_edge('source', center_node, capacity=r)
    
    # Add point nodes and edges to sink
    for i in range(n):
        point_node = f'point_{i}'
        G.add_node(point_node)
        G.add_edge(point_node, 'sink', capacity=1)
    
    # Add edges from centers to points (if distance <= 2R)
    for center_idx in centers:
        center_node = f'center_{center_idx}'
        for i in range(n):
            if distance_matrix[center_idx][i] <= 2 * R:
                point_node = f'point_{i}'
                G.add_edge(center_node, point_node, capacity=1)
    
    return G

def flow_network_verification(n: int, centers: list[int], 
                             distance_matrix: np.ndarray, R: float, r: int):
    """
    Phase 2.2: Flow network verification and reassignment
    """
    # Build flow network
    G = build_flow_network(n, centers, distance_matrix, R, r)
    
    # Compute maximum flow from source to sink
    flow_value, flow_dict = nx.maximum_flow(
        G, 'source', 'sink',
        flow_func=nx.algorithms.flow.shortest_augmenting_path
    )
    
    # Check if flow equals r * |C|
    expected_flow = r * len(centers)
    if flow_value < expected_flow - 0.5:
        return False, {}
    
    # Extract assignments from flow
    assignments = {}
    center_assignment_counts = {c: 0 for c in centers}
    
    for center_idx in centers:
        center_node = f'center_{center_idx}'
        for point_idx in range(n):
            point_node = f'point_{point_idx}'
            if point_node in flow_dict.get(center_node, {}):
                flow_amount = flow_dict[center_node][point_node]
                if flow_amount > 0.5:
                    assert point_idx not in assignments, "Point assigned twice in flow!"
                    assignments[point_idx] = center_idx
                    center_assignment_counts[center_idx] += 1
    
    # Verify each center got exactly r points from flow
    for center_idx, count in center_assignment_counts.items():
        if count != r:
            # This shouldn't happen if flow is correct
            return False, {}
    
    # Handle remaining nodes (those not in flow solution)
    for point_idx in range(n):
        if point_idx not in assignments:
            # Find any center within 2R
            assigned = False
            for center_idx in centers:
                if distance_matrix[point_idx][center_idx] <= 2 * R:
                    assignments[point_idx] = center_idx
                    assigned = True
                    break
            if not assigned:
                return False, {}
    
    return True, assignments