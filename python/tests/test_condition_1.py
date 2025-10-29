import numpy as np
from r_gather.data_structures import Point
from r_gather.distance_matrix import compute_distance_matrix
from r_gather.r_gather import check_condition_1

def test_condition_1_basic():
    """测试基本情况：3个点形成等边三角形"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([3.0, 0.0])),
        Point(id=2, coordinate=np.array([1.5, 2.598])),
    ]

    dist_matrix = compute_distance_matrix(points)
    r = 2
    
    # R=1.0, 2R=2.0, 应该不满足
    assert check_condition_1(dist_matrix, 1.0, r) == False
    
    # R=1.5, 2R=3.0, 应该满足
    assert check_condition_1(dist_matrix, 1.5, r) == True
    
    # R=2.0, 2R=4.0, 应该满足
    assert check_condition_1(dist_matrix, 2.0, r) == True


def test_condition_1_edge_case():
    """测试边界情况：点恰好在2R距离上"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([4.0, 0.0])),
        Point(id=2, coordinate=np.array([8.0, 0.0])),
    ]

    dist_matrix = compute_distance_matrix(points)
    r = 2
    
    # R=2.0, 2R=4.0, 点0和点1距离恰好为4，应该满足
    assert check_condition_1(dist_matrix, 2.0, r) == True


def test_condition_1_insufficient():
    """测试不满足条件的情况：孤立点"""
    points = [
        Point(id=0, coordinate=np.array([0.0, 0.0])),
        Point(id=1, coordinate=np.array([1.0, 0.0])),
        Point(id=2, coordinate=np.array([100.0, 0.0])),
    ]

    dist_matrix = compute_distance_matrix(points)
    r = 2
    
    # R=1.0, 2R=2.0, 点2是孤立的，应该不满足
    assert check_condition_1(dist_matrix, 1.0, r) == False


def test_condition_1_paper_example():
    """测试论文Figure 1(a)的示例"""
    points = [
        Point(id=0, coordinate=np.array([20, 10])),
        Point(id=1, coordinate=np.array([22, 10])),
        Point(id=2, coordinate=np.array([30, 23])),
        Point(id=3, coordinate=np.array([30, 20])),
        Point(id=4, coordinate=np.array([30, 17])),
    ]

    dist_matrix = compute_distance_matrix(points)
    r = 2
    
    # R=1.0, 2R=2.0, 应该不满足
    assert check_condition_1(dist_matrix, 1.0, r) == False
    
    # R=1.5, 2R=3.0, 应该满足
    assert check_condition_1(dist_matrix, 1.5, r) == True
    
    # R=2.0, 2R=4.0, 应该满足
    assert check_condition_1(dist_matrix, 2.0, r) == True
    
    # R=3.0, 2R=6.0, 应该满足
    assert check_condition_1(dist_matrix, 3.0, r) == True


if __name__ == '__main__':
    test_condition_1_basic()
    test_condition_1_edge_case()
    test_condition_1_insufficient()
    test_condition_1_paper_example()
    print("All condition 1 tests passed.")