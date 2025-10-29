# data_structure.py
import numpy as np
from dataclasses import dataclass

# Point Data Structure
@dataclass
class Point:
    id: int
    coordinate: np.ndarray

# Cluster Data Structure
@dataclass
class Cluster:
    id: int
    coordinate: np.ndarray
    members: list[Point]
    radius: float
    
    def size(self) -> int:
        return len(self.members)