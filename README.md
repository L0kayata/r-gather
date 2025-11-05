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

run `pytest -v` for correctness test

run `python -m tests.test_visualization` for visual test

## Complexity Analysis

Step 1:

* Compute pairwise distances:
    * `coords = np.array([p.coordinate for p in points])`: $O(n \cdot d)$
    * `diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]`: $O(n^2 \cdot d)$
    * `matrix = np.linalg.norm(diff, axis = -1)`: $O(n^2 \cdot d)$
    * $O(n \cdot d) + O(n^2 \cdot d) + O(n^2 \cdot d) = O(n^2)$
* Generate candidate radii:
    * `distance_matrix / 2`: $O(n^2)$
    * `np.unique()`: $O(n^2 \cdot log(n^2))$
    * $O(n^2)$ + $O(n^2 \cdot log(n^2))$ = $O(n^2 \cdot log(n))$
* $O(n^2 \cdot d) + O(n^2 \cdot log(n)) = O(n^2 \cdot log(n))$

Step 2:
* For each R in candidate radii:
    * $O(n^2)$
    * (Assuming the worst-case scenario, after sorting, there are no duplicate values. The actual value may in fact be much less than this.)
* Condition 1:
    * `neighbor_counts = np.sum(distance_matrix <= 2 * R, axis = 1)`: $O(n^2)$
    * `return np.all(neighbor_counts >= r)`: $O(n)$
    * $O(n^2) + O(n) = O(n^2)$
* Condition 2:
    * Phase 2.1:
        * `while not np.all(marked):`: $O(n/r)$
        * `for p_idx in unmarked_indices:`: $O(n)$
        * within for: $O(n)$
        * $O(n/r) \times O(n) \times O(n) = O(n^3)$
        * (Each point is considered as a potential center at most once. Once a point becomes a center or is marked, it is no longer considered. The actual complexity is close to $O(n^2)$.)
    * Phase 2.2
        * $|C| \le \lfloor n/r \rfloor$ (worst-case scenario)
        * V
            * $1$ source vertex
            * $1$ sink vertex
            * $|C|$ center vertexs
            * $n$ point vertexs
        * $V = 2 + |C| + n$
        * $V \le 2 + n/r + n = O(n)$
        * E
            * $|C|$ Source → Centers
            * $|C| \cdot n$ Centers → Points (worst-case scenario)
            * $n$ Points → Sink
        * $E \le |C| + |C| \cdot n + n = n/r + n^2/r + n = O(n^2)$
        * Edmonds-Karp: $O(V \cdot E^2) = O(n^5)$
        * Dinic: $O(V^2 \cdot E) = O(n^4)$
        * (Due to spatial locality, the number of edges from the centers to the points is far less than $|C| \cdot n$. The actual $E \ll n^2/r$, also $|C| \ll \lfloor n/r \rfloor$ in actually, so the actual running time is much better than the worst-case scenario.)
* $O(n^2) \cdot (O(n^3) + O(n^4)) = O(n^6)$ (The actual value may in fact be much less than this.)

Step 3:
* `build_clusters_from_assignments()`: $O(n)$

Final Time Complexity: $O(n^6)$ (The actual value may in fact be much less than this.)





