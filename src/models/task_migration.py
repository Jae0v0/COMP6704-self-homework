"""
Task migration and service placement model
"""

import numpy as np
from typing import Tuple


class TaskMigrationModel:
    """Task migration and service placement model"""
    
    def __init__(self,
                 num_uavs: int,
                 num_users: int,
                 num_services: int,
                 service_sizes: np.ndarray,
                 storage_capacities: np.ndarray,
                 backhaul_rate: float):
        """
        Initialize task migration model
        
        Args:
            num_uavs: Number of UAVs
            num_users: Number of users
            num_services: Number of service types
            service_sizes: Storage size for each service (S,)
            storage_capacities: Storage capacity for each UAV (N,)
            backhaul_rate: Inter-UAV backhaul link rate (bits/s)
        """
        self.N = num_uavs
        self.U = num_users
        self.S = num_services
        self.D_sizes = service_sizes
        self.M_max = storage_capacities
        self.R_backhaul = backhaul_rate
        
        # Service placement: x[i, s] = 1 if service s is cached at UAV i
        self.service_placement = None
        
        # Task migration: m[i, j, s] = 1 if task s migrates from UAV i to j
        self.migration = None
        
        # Task offloading: a[u, i, s] = 1 if user u offloads task s to UAV i
        self.offloading = None
    
    def initialize_service_placement(self, seed: int = None):
        """
        Initialize service placement randomly
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.service_placement = np.zeros((self.N, self.S), dtype=int)
        
        # Randomly place services while respecting storage constraints
        for i in range(self.N):
            available_storage = self.M_max[i]
            services = list(range(self.S))
            np.random.shuffle(services)
            
            for s in services:
                if self.D_sizes[s] <= available_storage:
                    self.service_placement[i, s] = 1
                    available_storage -= self.D_sizes[s]
    
    def check_storage_constraint(self, service_placement: np.ndarray) -> bool:
        """
        Check if storage constraints are satisfied
        
        Args:
            service_placement: Service placement matrix (N, S)
            
        Returns:
            True if constraints satisfied
        """
        for i in range(self.N):
            total_size = np.sum(service_placement[i] * self.D_sizes)
            if total_size > self.M_max[i]:
                return False
        return True
    
    def check_migration_constraint(self,
                                  migration: np.ndarray,
                                  service_placement: np.ndarray) -> bool:
        """
        Check if migration constraints are satisfied
        Migration can only go to UAVs that have the service cached
        
        Args:
            migration: Migration matrix (N, N, S)
            service_placement: Service placement matrix (N, S)
            
        Returns:
            True if constraints satisfied
        """
        for i in range(self.N):
            for j in range(self.N):
                for s in range(self.S):
                    if migration[i, j, s] > 0 and service_placement[j, s] == 0:
                        return False
        return True
    
    def compute_migration_delay(self,
                               migration: np.ndarray,
                               data_size: float) -> float:
        """
        Compute migration delay
        
        Args:
            migration: Migration matrix (N, N, S)
            data_size: Task data size (bits)
            
        Returns:
            Total migration delay (s)
        """
        # Count number of migrations (excluding self-migration)
        num_migrations = np.sum(migration) - np.sum(np.diagonal(migration, axis1=0, axis2=1))
        
        if num_migrations == 0:
            return 0.0
        
        return num_migrations * data_size / self.R_backhaul
    
    def compute_computation_delay(self,
                                 cpu_cycles_per_bit: float,
                                 data_size: float,
                                 computation_frequency: float,
                                 offloading_ratio: float = 1.0) -> float:
        """
        Compute computation delay
        Based on mappo-xwc-code: T_comp = DC / f
        where DC = D * C * rho (data size * cycles per bit * offloading ratio)
        
        Args:
            cpu_cycles_per_bit: Required CPU cycles per bit (C)
            data_size: Task data size (bits) (D)
            computation_frequency: Allocated computation frequency (Hz) (f)
            offloading_ratio: Fraction of task offloaded (rho), default 1.0
            
        Returns:
            Computation delay (s)
        """
        # DC = D * C * rho
        total_cycles = cpu_cycles_per_bit * data_size * offloading_ratio
        # T = DC / f
        delay = total_cycles / (computation_frequency + 1e-10)
        return delay
    
    def compute_total_delay(self,
                           data_size: float,
                           cpu_cycles_per_bit: float,
                           transmission_rate: float,
                           computation_frequency: float,
                           offloading_ratio: float = 1.0,
                           migration_delay: float = 0.0) -> float:
        """
        Compute total service delay including transmission, migration, and computation
        Based on mappo-xwc-code delay model
        
        Args:
            data_size: Task data size (bits)
            cpu_cycles_per_bit: CPU cycles per bit
            transmission_rate: Transmission rate (bits/s)
            computation_frequency: Computation frequency (Hz)
            offloading_ratio: Fraction offloaded (0-1)
            migration_delay: Additional migration delay (s)
            
        Returns:
            Total delay (s)
        """
        # Local computation delay
        local_delay = self.compute_computation_delay(
            cpu_cycles_per_bit, data_size, computation_frequency, 1.0 - offloading_ratio
        )
        
        if offloading_ratio > 0:
            # Transmission delay
            transmission_delay = (data_size * offloading_ratio) / (transmission_rate + 1e-10)
            # Remote computation delay
            remote_delay = self.compute_computation_delay(
                cpu_cycles_per_bit, data_size, computation_frequency, offloading_ratio
            )
            # Total offloading delay
            offloading_delay = transmission_delay + migration_delay + remote_delay
            # Total delay is max of local and offloading (parallel execution)
            total_delay = max(local_delay, offloading_delay)
        else:
            total_delay = local_delay
        
        return total_delay

