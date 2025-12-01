"""
UAV mobility and collision avoidance model
"""

import numpy as np
from typing import Tuple, List


class UAVModel:
    """UAV mobility model with collision avoidance constraints"""
    
    def __init__(self, 
                 num_uavs: int,
                 max_speed: float,
                 min_distance: float,
                 altitude: float,
                 dt: float):
        """
        Initialize UAV model
        
        Args:
            num_uavs: Number of UAVs
            max_speed: Maximum flight speed (m/s)
            min_distance: Minimum safety distance between UAVs (m)
            altitude: Fixed flight altitude (m)
            dt: Time slot duration (s)
        """
        self.N = num_uavs
        self.V_max = max_speed
        self.d_min = min_distance
        self.H = altitude
        self.dt = dt
        
        # Initialize positions: (N, 2) array [x, y]
        self.positions = None
        
    def initialize_positions(self, area_size: float, seed: int = None):
        """
        Initialize UAV positions randomly in the area
        
        Args:
            area_size: Size of the operational area (m)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random initial positions
        self.positions = np.random.uniform(0, area_size, size=(self.N, 2))
        
        # Ensure minimum distance between UAVs
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.d_min:
                    # Move UAV j away
                    direction = (self.positions[j] - self.positions[i]) / (dist + 1e-6)
                    self.positions[j] = self.positions[i] + direction * self.d_min * 1.1
    
    def get_positions(self) -> np.ndarray:
        """Get current UAV positions"""
        return self.positions.copy()
    
    def set_positions(self, positions: np.ndarray):
        """Set UAV positions"""
        self.positions = positions.copy()
    
    def check_speed_constraint(self, new_positions: np.ndarray) -> bool:
        """
        Check if speed constraint is satisfied
        
        Args:
            new_positions: New positions to check (N, 2)
            
        Returns:
            True if constraint satisfied
        """
        if self.positions is None:
            return True
        
        distances = np.linalg.norm(new_positions - self.positions, axis=1)
        return np.all(distances <= self.V_max * self.dt)
    
    def check_collision_constraint(self, positions: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Check collision avoidance constraints
        
        Args:
            positions: UAV positions to check (N, 2)
            
        Returns:
            (is_valid, violation_matrix)
        """
        violations = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.d_min:
                    violations[i, j] = self.d_min - dist
                    violations[j, i] = violations[i, j]
        
        is_valid = np.all(violations == 0)
        return is_valid, violations
    
    def compute_velocities(self, new_positions: np.ndarray) -> np.ndarray:
        """
        Compute velocities from position change
        
        Args:
            new_positions: New positions (N, 2)
            
        Returns:
            Velocities (N, 2)
        """
        if self.positions is None:
            return np.zeros((self.N, 2))
        
        return (new_positions - self.positions) / self.dt

