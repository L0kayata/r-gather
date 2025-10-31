# r-Gather Algorithm Implementation

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