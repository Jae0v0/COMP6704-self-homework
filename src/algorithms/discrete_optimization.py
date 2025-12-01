"""
Discrete resource optimization via penalty method and DC programming
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict


class DiscreteOptimizer:
    """Discrete resource optimization using penalty method"""
    
    def __init__(self,
                 num_uavs: int,
                 num_users: int,
                 num_services: int,
                 penalty_factor: float = 100.0):
        """
        Initialize discrete optimizer
        
        Args:
            num_uavs: Number of UAVs
            num_users: Number of users
            num_services: Number of service types
            penalty_factor: Penalty factor lambda for binary constraints
        """
        self.N = num_uavs
        self.U = num_users
        self.S = num_services
        self.lambda_penalty = penalty_factor
    
    def optimize(self,
                channel_gains: np.ndarray,
                transmission_rates: np.ndarray,
                service_sizes: np.ndarray,
                storage_capacities: np.ndarray,
                compute_capacities: np.ndarray,
                data_size: float,
                cpu_cycles_per_bit: float,
                max_delay: float,
                current_x: np.ndarray = None,
                current_m: np.ndarray = None,
                current_a: np.ndarray = None) -> Dict:
        """
        Optimize discrete variables (service placement, migration, offloading)
        and continuous variables (bandwidth allocation, compute frequency)
        
        Args:
            channel_gains: Channel gains (U, N)
            transmission_rates: Base transmission rates (U, N)
            service_sizes: Service storage sizes (S,)
            storage_capacities: UAV storage capacities (N,)
            compute_capacities: UAV compute capacities (N,)
            data_size: Task data size (bits)
            cpu_cycles_per_bit: CPU cycles per bit
            max_delay: Maximum allowed delay (s)
            current_x: Current service placement (N, S) for warm start
            current_m: Current migration matrix (N, N, S) for warm start
            current_a: Current offloading matrix (U, N, S) for warm start
            
        Returns:
            Dictionary with optimized variables
        """
        # Initialize variables if not provided
        if current_x is None:
            current_x = np.random.rand(self.N, self.S)
        if current_m is None:
            current_m = np.random.rand(self.N, self.N, self.S)
        if current_a is None:
            current_a = np.random.rand(self.U, self.N, self.S)
        
        # CVXPY variables
        x = cp.Variable((self.N, self.S), nonneg=True)
        m = cp.Variable((self.N, self.N, self.S), nonneg=True)
        a = cp.Variable((self.U, self.N, self.S), nonneg=True)
        eta = cp.Variable((self.U, self.N), nonneg=True)  # Bandwidth allocation
        f = cp.Variable((self.N, self.S), nonneg=True)  # Compute frequency
        
        # Constraints
        constraints = []
        
        # C1: Storage constraint
        for i in range(self.N):
            constraints.append(cp.sum(x[i, :] * service_sizes) <= storage_capacities[i])
        
        # C2: Bandwidth constraint
        for i in range(self.N):
            constraints.append(cp.sum(eta[:, i]) <= 1.0)
        
        # C3: Compute capacity constraint
        for i in range(self.N):
            constraints.append(cp.sum(f[i, :]) <= compute_capacities[i])
        
        # C4: Migration constraint: m[i,j,s] <= x[j,s]
        for i in range(self.N):
            for j in range(self.N):
                for s in range(self.S):
                    constraints.append(m[i, j, s] <= x[j, s])
        
        # C5: Delay constraint (simplified linear approximation)
        # T = D/R + migration_delay + computation_delay
        for u in range(self.U):
            for i in range(self.N):
                for s in range(self.S):
                    if transmission_rates[u, i] > 0:
                        # Transmission delay
                        trans_delay = data_size / (eta[u, i] * transmission_rates[u, i] + 1e-10)
                        # Migration delay (simplified)
                        mig_delay = cp.sum(m[i, :, s]) * data_size / 1e6  # Simplified
                        # Computation delay
                        comp_delay = cpu_cycles_per_bit * data_size / (f[i, s] + 1e-10)
                        total_delay = trans_delay + mig_delay + comp_delay
                        constraints.append(a[u, i, s] * total_delay <= max_delay)
        
        # C6: Box constraints
        constraints.append(x <= 1.0)
        constraints.append(m <= 1.0)
        constraints.append(a <= 1.0)
        
        # Objective: Maximize utility - penalty term
        # Note: In Block 1 (discrete optimization), with fixed trajectories,
        # the channel gains are constant, so we can approximate the log-utility
        # For exact implementation, we would need: sum_u omega_u * log(1 + sum_{i,s} a_{u,i,s} * 1/T_{u,s})
        # where T_{u,s} = D/R + migration_delay + computation_delay
        # Here we use a surrogate objective that encourages offloading decisions
        # The exact log-utility is computed in the main BCD-MPC loop
        
        # Surrogate utility: maximize offloading decisions weighted by transmission rates
        # This approximates maximizing throughput, which is related to the log-utility
        utility_terms = []
        for u in range(self.U):
            for i in range(self.N):
                for s in range(self.S):
                    if transmission_rates[u, i] > 0:
                        # Approximate throughput contribution
                        # Higher transmission rate -> lower delay -> higher utility
                        utility_terms.append(a[u, i, s] * transmission_rates[u, i] / 1e6)  # Normalize
        
        utility = cp.sum(utility_terms) if utility_terms else cp.sum(a)
        
        # Penalty term for binary constraints (DC programming)
        penalty_x = self.lambda_penalty * cp.sum(x - cp.multiply(x, x))
        penalty_m = self.lambda_penalty * cp.sum(m - cp.multiply(m, m))
        penalty_a = self.lambda_penalty * cp.sum(a - cp.multiply(a, a))
        
        # Linearize concave part using Taylor expansion
        # -x^2 >= -x_k^2 - 2*x_k*(x - x_k)
        linearized_penalty_x = 0
        linearized_penalty_m = 0
        linearized_penalty_a = 0
        
        for i in range(self.N):
            for s in range(self.S):
                x_k = current_x[i, s]
                linearized_penalty_x += self.lambda_penalty * (
                    x[i, s] - x_k**2 - 2 * x_k * (x[i, s] - x_k)
                )
        
        for i in range(self.N):
            for j in range(self.N):
                for s in range(self.S):
                    m_k = current_m[i, j, s]
                    linearized_penalty_m += self.lambda_penalty * (
                        m[i, j, s] - m_k**2 - 2 * m_k * (m[i, j, s] - m_k)
                    )
        
        for u in range(self.U):
            for i in range(self.N):
                for s in range(self.S):
                    a_k = current_a[u, i, s]
                    linearized_penalty_a += self.lambda_penalty * (
                        a[u, i, s] - a_k**2 - 2 * a_k * (a[u, i, s] - a_k)
                    )
        
        objective = cp.Maximize(utility - linearized_penalty_x - linearized_penalty_m - linearized_penalty_a)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                # Round binary variables
                x_rounded = np.round(x.value)
                m_rounded = np.round(m.value)
                a_rounded = np.round(a.value)
                
                return {
                    'x': np.clip(x_rounded, 0, 1).astype(int),
                    'm': np.clip(m_rounded, 0, 1),
                    'a': np.clip(a_rounded, 0, 1),
                    'eta': np.clip(eta.value, 0, 1),
                    'f': np.maximum(f.value, 0),
                    'status': 'success'
                }
            else:
                return {
                    'x': current_x,
                    'm': current_m,
                    'a': current_a,
                    'eta': np.ones((self.U, self.N)) / (self.U + 1),
                    'f': np.ones((self.N, self.S)) * compute_capacities[:, None] / (self.S + 1),
                    'status': 'failed'
                }
        except Exception as e:
            print(f"Optimization error: {e}")
            return {
                'x': current_x,
                'm': current_m,
                'a': current_a,
                'eta': np.ones((self.U, self.N)) / (self.U + 1),
                'f': np.ones((self.N, self.S)) * compute_capacities[:, None] / (self.S + 1),
                'status': 'error'
            }

