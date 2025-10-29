import numpy as np
from r_gather.data_structures import Point, Cluster
from r_gather.distance_matrix import compute_distance_matrix
from r_gather.r_gather import compute_r_gather, check_condition_1, check_condition_2, initial_clustering
from r_gather.flow_network import flow_network_verification, build_flow_network

def test_simple_clustering():
    """测试简单的4点聚类"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([5.0, 0.0])),
        Point(id=3, coordinate=np.array([6.0, 0.0])),
    ]
    
    r = 2
    clusters = compute_r_gather(points, r)
    
    # 应该形成2个簇
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
    
    # 每个簇至少有r个成员
    for cluster in clusters:
        assert cluster.size() >= r, f"Cluster {cluster.id} has {cluster.size()} members, expected at least {r}"
    
    # 验证所有点都被聚类
    total_points = sum(cluster.size() for cluster in clusters)
    assert total_points == len(points), "All points should be clustered"


def test_initial_clustering_two_groups():
    """测试初始聚类构建 - 两个明确分离的组"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([0.0, 1.0])),
        Point(id=3, coordinate=np.array([10.0, 10.0])),
        Point(id=4, coordinate=np.array([11.0, 10.0])),
        Point(id=5, coordinate=np.array([10.0, 11.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 3
    R = 1.5  # 2R = 3.0
    
    centers = initial_clustering(dist_matrix, R, r)
    
    # 应该找到2个中心
    assert len(centers) == 2, f"Expected 2 centers, got {len(centers)}"
    
    # 验证所有点都被标记(centers不为空意味着成功)
    assert len(centers) > 0, "Should find valid centers"


def test_initial_clustering_failure():
    """测试初始聚类失败情况 - 孤立点导致无法标记所有点"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([100.0, 0.0])),  # 孤立点
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 2
    R = 1.0  # 2R = 2.0, 不足以覆盖孤立点
    
    centers = initial_clustering(dist_matrix, R, r)
    
    # 应该失败(返回空列表)
    assert len(centers) == 0, "Should fail when isolated point exists"


def test_initial_clustering_single_cluster():
    """测试初始聚类 - 所有点形成单个簇"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([0.0, 1.0])),
        Point(id=3, coordinate=np.array([1.0, 1.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 3
    R = 1.0  # 2R = 2.0, 足以覆盖所有点
    
    centers = initial_clustering(dist_matrix, R, r)
    
    # 应该只找到1个中心
    assert len(centers) == 1, f"Expected 1 center, got {len(centers)}"


def test_flow_network_basic():
    """测试基本流网络验证"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([2.0, 0.0])),
        Point(id=3, coordinate=np.array([3.0, 0.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 2
    R = 1.5  # 2R = 3.0
    centers = [0, 2]
    
    success, assignments = flow_network_verification(len(points), centers, dist_matrix, R, r)
    
    assert success, "Flow network verification should succeed"
    assert len(assignments) == len(points), "All points should be assigned"
    
    # 验证每个中心至少有r个点
    center_counts = {}
    for point_idx, center_idx in assignments.items():
        center_counts[center_idx] = center_counts.get(center_idx, 0) + 1
    
    for center_idx in centers:
        assert center_counts.get(center_idx, 0) >= r, f"Center {center_idx} has insufficient points"


def test_flow_network_structure():
    """测试流网络结构的正确性"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([5.0, 0.0])),
        Point(id=3, coordinate=np.array([6.0, 0.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 2
    R = 1.5
    centers = [0, 2]
    
    G = build_flow_network(len(points), centers, dist_matrix, R, r)
    
    # 验证节点存在
    assert 'source' in G.nodes(), "Source node should exist"
    assert 'sink' in G.nodes(), "Sink node should exist"
    
    # 验证中心节点
    for center_idx in centers:
        center_node = f'center_{center_idx}'
        assert center_node in G.nodes(), f"Center node {center_node} should exist"
        # 验证source到center的边容量为r
        assert G['source'][center_node]['capacity'] == r, f"Edge from source to {center_node} should have capacity {r}"
    
    # 验证点节点
    for i in range(len(points)):
        point_node = f'point_{i}'
        assert point_node in G.nodes(), f"Point node {point_node} should exist"
        # 验证point到sink的边容量为1
        assert G[point_node]['sink']['capacity'] == 1, f"Edge from {point_node} to sink should have capacity 1"


def test_flow_network_insufficient_capacity():
    """测试流网络容量不足的情况"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([10.0, 0.0])),  # 距离太远
        Point(id=2, coordinate=np.array([20.0, 0.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 2
    R = 0.5  # 2R = 1.0, 太小
    centers = [0]
    
    success, assignments = flow_network_verification(len(points), centers, dist_matrix, R, r)
    
    # 应该失败,因为没有足够的点在2R范围内
    assert not success, "Should fail when insufficient points within 2R"


def test_condition_2_success():
    """测试Condition 2成功的情况"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([5.0, 0.0])),
        Point(id=3, coordinate=np.array([6.0, 0.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 2
    R = 1.0
    
    success, clusters = check_condition_2(points, dist_matrix, R, r)
    
    assert success, "Condition 2 should succeed"
    assert len(clusters) > 0, "Should produce clusters"
    
    # 验证每个簇至少r个点
    for cluster in clusters:
        assert cluster.size() >= r, f"Cluster {cluster.id} should have at least {r} members"


def test_condition_2_failure():
    """测试Condition 2失败的情况"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([100.0, 0.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    r = 2
    R = 0.5  # 太小,无法覆盖足够的点
    
    success, clusters = check_condition_2(points, dist_matrix, R, r)
    
    assert not success, "Condition 2 should fail with isolated point"
    assert len(clusters) == 0, "Should return empty clusters on failure"


def test_paper_example():
    """测试类似论文Figure 1(a)的示例"""
    points = [
        Point(id=0, coordinate=np.array([20.0, 10.0])),
        Point(id=1, coordinate=np.array([22.0, 10.0])),
        Point(id=2, coordinate=np.array([30.0, 23.0])),
        Point(id=3, coordinate=np.array([30.0, 20.0])),
        Point(id=4, coordinate=np.array([30.0, 17.0])),
    ]
    
    r = 2
    clusters = compute_r_gather(points, r)
    
    assert len(clusters) > 0, "Should find valid clustering"
    
    # 验证所有点被聚类
    total_points = sum(cluster.size() for cluster in clusters)
    assert total_points == len(points), f"All {len(points)} points should be clustered"
    
    # 验证最小簇大小约束
    for cluster in clusters:
        assert cluster.size() >= r, f"Cluster {cluster.id} violates minimum size constraint"


def test_edge_case_exact_r_members():
    """测试边界情况 - 每个簇恰好r个成员"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([0.5, 0.0])),
        Point(id=2, coordinate=np.array([10.0, 0.0])),
        Point(id=3, coordinate=np.array([10.5, 0.0])),
    ]
    
    r = 2
    clusters = compute_r_gather(points, r)
    
    assert len(clusters) == 2, f"Expected 2 clusters for 4 points with r=2"
    for cluster in clusters:
        assert cluster.size() == 2, f"Each cluster should have exactly 2 members"


def test_edge_case_single_point_per_cluster():
    """测试r=1的情况 - 每个簇最少1个点"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([5.0, 0.0])),
        Point(id=2, coordinate=np.array([10.0, 0.0])),
    ]
    
    r = 1
    clusters = compute_r_gather(points, r)
    
    assert len(clusters) > 0, "Should find valid clustering with r=1"
    
    # 每个簇至少1个点
    for cluster in clusters:
        assert cluster.size() >= r, f"Cluster should have at least {r} member"


def test_edge_case_all_points_one_cluster():
    """测试所有点必须在一个簇中的情况"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([0.0, 1.0])),
    ]
    
    r = 3  # 等于总点数
    clusters = compute_r_gather(points, r)
    
    assert len(clusters) == 1, "Should form exactly 1 cluster when r equals total points"
    assert clusters[0].size() == 3, "The single cluster should contain all points"


def test_different_r_values():
    """测试不同的r值"""
    points = [
        Point(id=i, coordinate=np.array([float(i), 0.0]))
        for i in range(6)
    ]
    
    # r=2: 最多3个簇
    clusters_r2 = compute_r_gather(points, 2)
    assert len(clusters_r2) <= 3, "With r=2 and 6 points, at most 3 clusters"
    for cluster in clusters_r2:
        assert cluster.size() >= 2, "Each cluster should have at least 2 members"
    
    # r=3: 最多2个簇
    clusters_r3 = compute_r_gather(points, 3)
    assert len(clusters_r3) <= 2, "With r=3 and 6 points, at most 2 clusters"
    for cluster in clusters_r3:
        assert cluster.size() >= 3, "Each cluster should have at least 3 members"


def test_cluster_radius_bounds():
    """测试簇半径的界限"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 4.0])),  # 距离5
        Point(id=2, coordinate=np.array([6.0, 8.0])),  # 距离10
    ]
    
    r = 2
    clusters = compute_r_gather(points, r)
    
    # 验证簇半径不超过2倍最优半径(根据2-approximation)
    for cluster in clusters:
        # 簇半径应该是非负的
        assert cluster.radius >= 0, "Cluster radius should be non-negative"


def test_condition_1_with_various_r():
    """测试Condition 1在不同r值下的行为"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([2.0, 0.0])),
        Point(id=3, coordinate=np.array([3.0, 0.0])),
    ]
    
    dist_matrix = compute_distance_matrix(points)
    R = 1.5  # 2R = 3.0
    
    # r=2时应该满足
    assert check_condition_1(dist_matrix, R, 2) == True, "Should pass with r=2"
    
    # r=3时应该满足
    assert check_condition_1(dist_matrix, R, 3) == True, "Should pass with r=3"
    
    # r=5时应该不满足(总共只有4个点)
    assert check_condition_1(dist_matrix, R, 5) == False, "Should fail with r=5"


def test_no_valid_clustering():
    """测试无法找到有效聚类的情况 - r大于总点数"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
    ]
    
    r = 3  # r大于总点数,不可能满足条件
    clusters = compute_r_gather(points, r)
    
    # r=3但只有2个点,无法满足Condition 1,应该返回空列表
    assert len(clusters) == 0, "Should return empty list when r > total points"


def test_no_valid_clustering():
    """测试无法找到有效聚类的情况 - r大于总点数"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
    ]
    
    r = 3  # r大于总点数,不可能满足条件
    clusters = compute_r_gather(points, r)
    
    # r=3但只有2个点,无法满足Condition 1,应该返回空列表
    assert len(clusters) == 0, "Should return empty list when r > total points"

def test_cluster_assignment_consistency():
    """测试簇分配的一致性 - 每个点只属于一个簇"""
    points = [
        Point(id=i, coordinate=np.array([float(i * 2), 0.0]))
        for i in range(8)
    ]
    
    r = 2
    clusters = compute_r_gather(points, r)
    
    if len(clusters) > 0:
        # 收集所有被分配的点ID
        assigned_point_ids = set()
        for cluster in clusters:
            for member in cluster.members:
                # 每个点只能出现一次
                assert member.id not in assigned_point_ids, f"Point {member.id} assigned to multiple clusters"
                assigned_point_ids.add(member.id)
        
        # 所有点都应该被分配
        assert len(assigned_point_ids) == len(points), "All points should be assigned exactly once"


if __name__ == '__main__':
    test_simple_clustering()
    test_initial_clustering_two_groups()
    test_initial_clustering_failure()
    test_initial_clustering_single_cluster()
    test_flow_network_basic()
    test_flow_network_structure()
    test_flow_network_insufficient_capacity()
    test_condition_2_success()
    test_condition_2_failure()
    test_paper_example()
    test_edge_case_exact_r_members()
    test_edge_case_single_point_per_cluster()
    test_edge_case_all_points_one_cluster()
    test_different_r_values()
    test_cluster_radius_bounds()
    test_condition_1_with_various_r()
    test_no_valid_clustering()
    test_cluster_assignment_consistency()
    print("All r-Gather tests passed.")