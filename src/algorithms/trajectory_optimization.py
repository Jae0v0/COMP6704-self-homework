"""
Trajectory optimization via IRLS-SCA
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict


class TrajectoryOptimizer:
    """Trajectory optimization using IRLS and SCA"""
    
    def __init__(self,
                 num_uavs: int,
                 num_users: int,
                 min_distance: float,
                 max_speed: float,
                 dt: float,
                 penalty_factor: float = 1000.0):
        """
        Initialize trajectory optimizer
        
        Args:
            num_uavs: Number of UAVs
            num_users: Number of users
            min_distance: Minimum safety distance (m)
            max_speed: Maximum speed (m/s)
            dt: Time slot duration (s)
            penalty_factor: Penalty factor for soft constraints
        """
        self.N = num_uavs
        self.U = num_users
        self.d_min = min_distance
        self.V_max = max_speed
        self.dt = dt
        self.M_penalty = penalty_factor
    
    def compute_irls_weights(self,
                            uav_positions: np.ndarray,
                            user_positions: np.ndarray,
                            user_weights: np.ndarray,
                            channel_gains: np.ndarray) -> np.ndarray:
        """
        Compute IRLS weights for log-utility maximization
        
        Args:
            uav_positions: Current UAV positions (N, 2)
            user_positions: User positions (U, 2)
            user_weights: User priority weights (U,)
            channel_gains: Current channel gains (U, N)
            
        Returns:
            IRLS weights (U, N)
        """
        # Compute SNR (simplified)
        snr = channel_gains * 100  # Simplified SNR calculation
        
        # Weight formula: w_{u,i} = omega_u / (1 + SNR_{u,i})
        weights = np.zeros((self.U, self.N))
        for u in range(self.U):
            for i in range(self.N):
                weights[u, i] = user_weights[u] / (1 + snr[u, i] + 1e-10)
        
        return weights
    
    def optimize(self,
                current_positions: np.ndarray,
                user_positions: np.ndarray,
                user_weights: np.ndarray,
                channel_gains: np.ndarray,
                previous_positions: np.ndarray = None) -> Dict:
        """
        Optimize UAV trajectories using IRLS-SCA
        
        Args:
            current_positions: Current UAV positions (N, 2)
            user_positions: User positions (U, 2)
            user_weights: User priority weights (U,)
            channel_gains: Current channel gains (U, N)
            previous_positions: Previous positions for speed constraint (N, 2)
            
        Returns:
            Dictionary with optimized positions
        """
        # Compute IRLS weights
        irls_weights = self.compute_irls_weights(
            current_positions, user_positions, user_weights, channel_gains
        )
        
        # CVXPY variables
        q = cp.Variable((self.N, 2))
        delta = cp.Variable((self.N, self.N), nonneg=True)  # Slack variables for soft constraints
        
        # Objective: Minimize weighted distance (IRLS)
        objective_terms = []
        for u in range(self.U):
            for i in range(self.N):
                if irls_weights[u, i] > 0:
                    dist_sq = cp.sum_squares(q[i, :] - user_positions[u, :])
                    objective_terms.append(irls_weights[u, i] * dist_sq)
        
        # Add penalty for constraint violations
        penalty = self.M_penalty * cp.sum(delta)
        
        objective = cp.Minimize(cp.sum(objective_terms) + penalty)
        
        # Constraints
        constraints = []
        
        # C1: Speed constraint
        if previous_positions is not None:
            for i in range(self.N):
                dist = cp.norm(q[i, :] - previous_positions[i, :], 2)
                constraints.append(dist <= self.V_max * self.dt)
        
        # C2: Collision avoidance (SCA with soft constraints)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Current positions
                q_i_k = current_positions[i, :]
                q_j_k = current_positions[j, :]
                
                # Linearized constraint using SCA
                # ||q_i - q_j||^2 >= d_min^2
                # Linearize: -||q_i_k - q_j_k||^2 + 2*(q_i_k - q_j_k)^T*(q_i - q_j) >= d_min^2 - delta
                diff_k = q_i_k - q_j_k
                dist_k_sq = np.sum(diff_k**2)
                
                linearized = (-dist_k_sq + 
                             2 * diff_k @ (q[i, :] - q[j, :]))
                
                constraints.append(linearized >= self.d_min**2 - delta[i, j])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                new_positions = q.value
                
                # Ensure positions are valid
                if new_positions is None:
                    new_positions = current_positions.copy()
                else:
                    # Clip to reasonable bounds
                    new_positions = np.clip(new_positions, -1e6, 1e6)
                
                return {
                    'positions': new_positions,
                    'delta': delta.value if delta.value is not None else np.zeros((self.N, self.N)),
                    'status': 'success'
                }
            else:
                return {
                    'positions': current_positions.copy(),
                    'delta': np.zeros((self.N, self.N)),
                    'status': 'failed'
                }
        except Exception as e:
            print(f"Trajectory optimization error: {e}")
            return {
                'positions': current_positions.copy(),
                'delta': np.zeros((self.N, self.N)),
                'status': 'error'
            }

