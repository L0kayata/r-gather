# r-Gather Algorithm Implementation

### Algorithm Flow

**Step 1: Generate Candidate Radii**
- Function: `compute_distance_matrix(points)`
- Compute pairwise distances between all points
- Generate candidate radii: R = d_ij / 2 for all point pairs (i,j)
- Total candidates: O(n²)

**Step 2: Find Minimum Feasible Radius**
- Function: `compute_r_gather(points, r)`
- Iterate through candidate R values in ascending order
- For each R, verify two conditions:

  **Condition 1** - Function: `check_condition_1(distance_matrix, R, r)`
  - Each point must have at least r-1 neighbors within distance 2R

  **Condition 2** - Function: `check_condition_2(points, distance_matrix, R, r)`
  - Phase 2.1 - Function: `initial_clustering(distance_matrix, R, r)`
    - Greedily select cluster centers
    - Mark points within 2R of each center
  - Phase 2.2 - Function: `flow_network_verification(n, centers, distance_matrix, R, r)`
    - Build flow network: `build_flow_network(n, centers, distance_matrix, R, r)`
    - Compute maximum flow using NetworkX
    - Verify if exactly r points can be assigned to each center

**Step 3: Construct Final Clustering**
- Function: `build_clusters_from_assignments(points, centers, assignments, distance_matrix)`
- Build clusters based on flow network assignments
- Return the clustering result

## Test

open path `/python`

run `pytest -v`

## Complexity Analysis
| Function | Time Complexity | Space Complexity |
|---|---|---|
| `compute_distance_matrix` | $O(n^2d)$ | $O(n^2d)$ |
| `check_condition_1` | $O(n^2)$ | $O(n^2)$ |
| `initial_clustering` | $O(n^2)$ | $O(n)$ |
| `build_flow_network` | $O(n^2/r)$ | $O(n^2/r)$ |
| `flow_network_verification` | **$O(n^5/r^2)$** | $O(n^2/r)$ |
| `check_condition_2` | $O(n^5/r^2)$ | $O(n^2/r)$ |
| `build_clusters_from_assignments` | $O(n)$ | $O(n)$ |
| `compute_r_gather` | **$O(n^7/r^2)$** | $O(n^2d)$ |