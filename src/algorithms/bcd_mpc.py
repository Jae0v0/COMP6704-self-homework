"""
Online BCD-MPC Algorithm for Joint Service Deployment and Trajectory Planning
"""

import numpy as np
from typing import Dict, Tuple
from ..models.uav_model import UAVModel
from ..models.communication import CommunicationModel
from ..models.task_migration import TaskMigrationModel
from ..models.energy import EnergyModel
from .discrete_optimization import DiscreteOptimizer
from .trajectory_optimization import TrajectoryOptimizer


class BCDMPC:
    """Online Block Coordinate Descent with Model Predictive Control"""
    
    def __init__(self, config: Dict):
        """
        Initialize BCD-MPC algorithm
        
        Args:
            config: Configuration dictionary with all system parameters
        """
        self.config = config
        
        # Extract parameters
        self.N = config['num_uavs']
        self.U = config['num_users']
        self.S = config['num_services']
        self.T = config['num_time_slots']
        self.H_mpc = config.get('mpc_horizon', 5)  # MPC prediction horizon
        self.max_iter = config.get('max_iterations', 10)
        self.tolerance = config.get('tolerance', 1e-3)
        self.lambda_penalty = config.get('penalty_factor', 100.0)
        self.H_mpc = config.get('mpc_horizon', 1)
        self.area_size = config.get('area_size', 1000.0)
        self.user_prediction_noise = config.get('user_prediction_noise', self.area_size * 0.01)
        self.battery_capacity = config.get('battery_capacity', 1e6)
        self.energy_usage = np.zeros(self.N)
        self.energy_history = []
        
        # Initialize models
        self.uav_model = UAVModel(
            num_uavs=self.N,
            max_speed=config['max_speed'],
            min_distance=config['min_distance'],
            altitude=config['altitude'],
            dt=config['dt']
        )
        
        self.comm_model = CommunicationModel(
            bandwidth=config['bandwidth'],
            user_power=config['user_power'],
            noise_power=config['noise_power'],
            path_loss_exponent=config['path_loss_exponent'],
            reference_gain=config['reference_gain'],
            altitude=config['altitude']
        )
        
        self.migration_model = TaskMigrationModel(
            num_uavs=self.N,
            num_users=self.U,
            num_services=self.S,
            service_sizes=config['service_sizes'],
            storage_capacities=config['storage_capacities'],
            backhaul_rate=config['backhaul_rate']
        )
        
        self.energy_model = EnergyModel(
            blade_power=config.get('blade_power', 59.03),
            induced_power=config.get('induced_power', 79.07),
            tip_speed=config.get('tip_speed', 120.0),
            fuselage_drag_coefficient=config.get('fuselage_drag_coefficient', 0.6),
            air_density=config.get('air_density', 1.225),
            rotor_disk_area=config.get('rotor_disk_area', 0.5030),
            rotor_solidity=config.get('rotor_solidity', 0.05),
            mean_rotor_induced_velocity=config.get('mean_rotor_induced_velocity', 3.6),
            computation_coefficient=config.get('computation_coefficient', 1e-27),
            communication_power=config.get('communication_power', 10.0)
        )
        
        # Initialize optimizers
        self.discrete_optimizer = DiscreteOptimizer(
            num_uavs=self.N,
            num_users=self.U,
            num_services=self.S,
            penalty_factor=self.lambda_penalty
        )
        
        self.trajectory_optimizer = TrajectoryOptimizer(
            num_uavs=self.N,
            num_users=self.U,
            min_distance=config['min_distance'],
            max_speed=config['max_speed'],
            dt=config['dt'],
            penalty_factor=config.get('trajectory_penalty', 1000.0)
        )
        
        # State variables
        self.positions_history = []
        self.service_placement = None
        self.migration = None
        self.offloading = None
        self.bandwidth_allocation = None
        self.compute_frequency = None
        self.planned_user_positions = []
        
    def initialize(self, area_size: float, user_positions: np.ndarray, seed: int = None):
        """
        Initialize system state
        
        Args:
            area_size: Size of operational area (m)
            user_positions: User positions (U, 2)
            seed: Random seed
        """
        # Initialize UAV positions
        self.uav_model.initialize_positions(area_size, seed)
        
        # Initialize service placement
        self.migration_model.initialize_service_placement(seed)
        self.service_placement = self.migration_model.service_placement
        
        # Initialize other variables
        self.migration = np.zeros((self.N, self.N, self.S))
        self.offloading = np.zeros((self.U, self.N, self.S))
        self.bandwidth_allocation = np.ones((self.U, self.N)) / (self.U + 1)
        self.compute_frequency = np.ones((self.N, self.S)) * (
            self.config['compute_capacities'][:, None] / (self.S + 1)
        )
        
        self.positions_history = [self.uav_model.get_positions()]
        self.energy_usage = np.zeros(self.N)
        self.energy_history = []
        self.planned_user_positions = []

    def _predict_future_user_positions(self, user_positions: np.ndarray) -> list:
        """
        Predict future user positions for MPC horizon using a simple random walk model.
        Returns a list of length H_mpc (at least 1), where index 0 corresponds to current positions.
        """
        predictions = []
        current_positions = user_positions.copy()
        for _ in range(max(1, self.H_mpc)):
            predictions.append(current_positions.copy())
            noise = np.random.normal(0, self.user_prediction_noise, size=current_positions.shape)
            current_positions = np.clip(current_positions + noise, 0, self.area_size)
        return predictions

    def _aggregate_channel_gains(self, uav_positions: np.ndarray, future_user_positions: list) -> np.ndarray:
        """
        Aggregate channel gains across predicted user positions for MPC lookahead.
        We average the gains to capture expected future conditions.
        """
        aggregated = None
        for positions in future_user_positions:
            gains = self.comm_model.compute_channel_gain(uav_positions, positions)
            aggregated = gains if aggregated is None else aggregated + gains
        return aggregated / len(future_user_positions)

    def _enforce_energy_budget(self, previous_positions: np.ndarray, new_positions: np.ndarray) -> np.ndarray:
        """
        Enforce per-UAV energy constraints by scaling movements when exceeding the budget.
        Updates self.energy_usage and returns possibly adjusted positions.
        """
        dt = self.config['dt']
        velocities = (new_positions - previous_positions) / dt
        propulsion_energy = self.energy_model.compute_propulsion_energy(velocities, dt)
        communication_energy = np.ones(self.N) * self.energy_model.P_comm * dt
        total_energy = propulsion_energy + communication_energy
        self.energy_usage += total_energy

        violations = self.energy_usage - self.battery_capacity
        violating_indices = np.where(violations > 0)[0]

        if violating_indices.size > 0:
            for idx in violating_indices:
                # Scale movement to keep within remaining battery capacity
                remaining = max(self.battery_capacity - (self.energy_usage[idx] - total_energy[idx]), 0)
                if total_energy[idx] > 0:
                    scale = remaining / total_energy[idx]
                else:
                    scale = 0.0
                scale = np.clip(scale, 0.0, 1.0)
                new_positions[idx] = previous_positions[idx] + (new_positions[idx] - previous_positions[idx]) * scale
                # Recompute energy for this UAV with scaled movement
                velocity = (new_positions[idx] - previous_positions[idx]) / dt
                adjusted_propulsion = self.energy_model.compute_propulsion_energy(velocity[np.newaxis, :], dt)[0]
                adjusted_total = adjusted_propulsion + self.energy_model.P_comm * dt
                self.energy_usage[idx] = min(self.energy_usage[idx] - total_energy[idx] + adjusted_total, self.battery_capacity)

        self.energy_history.append(self.energy_usage.copy())
        return new_positions
    
    def compute_utility(self,
                       offloading: np.ndarray,
                       transmission_rates: np.ndarray,
                       user_weights: np.ndarray,
                       migration: np.ndarray = None,
                       compute_frequency: np.ndarray = None) -> float:
        """
        Compute weighted log-sum utility according to paper formula:
        sum_{u} omega_u * log(1 + sum_{i,s} a_{u,i,s}^t * 1/T_{u,s}^t)
        
        Args:
            offloading: Offloading decisions (U, N, S)
            transmission_rates: Transmission rates (U, N) in bits/s
            user_weights: User weights (U,)
            migration: Migration matrix (N, N, S), optional
            compute_frequency: Compute frequency allocation (N, S), optional
            
        Returns:
            Utility value
        """
        if migration is None:
            migration = self.migration if hasattr(self, 'migration') else np.zeros((self.N, self.N, self.S))
        if compute_frequency is None:
            compute_frequency = self.compute_frequency if hasattr(self, 'compute_frequency') else np.ones((self.N, self.S)) * 1e8
        
        data_size = self.config['data_size']
        cpu_cycles_per_bit = self.config['cpu_cycles_per_bit']
        backhaul_rate = self.config['backhaul_rate']
        
        utility = 0.0
        for u in range(self.U):
            user_throughput = 0.0
            for i in range(self.N):
                for s in range(self.S):
                    if offloading[u, i, s] > 0 and transmission_rates[u, i] > 0:
                        # Compute total service delay T_{u,s}^t
                        # T = D/R + migration_delay + computation_delay
                        
                        # Transmission delay
                        trans_delay = data_size / (transmission_rates[u, i] + 1e-10)
                        
                        # Migration delay (if task is migrated)
                        mig_delay = 0.0
                        if migration is not None:
                            # Check if task is migrated from UAV i to another UAV
                            for j in range(self.N):
                                if j != i and migration[i, j, s] > 0:
                                    mig_delay = data_size / (backhaul_rate + 1e-10)
                                    break
                        
                        # Computation delay
                        comp_delay = (cpu_cycles_per_bit * data_size) / (compute_frequency[i, s] + 1e-10)
                        
                        # Total delay
                        total_delay = trans_delay + mig_delay + comp_delay
                        
                        # Throughput = 1 / delay (as in paper)
                        throughput = 1.0 / (total_delay + 1e-10)
                        user_throughput += offloading[u, i, s] * throughput
            
            # Weighted log-sum utility
            utility += user_weights[u] * np.log(1 + user_throughput)
        
        return utility
    
    def solve_time_slot(self,
                       t: int,
                       user_positions: np.ndarray,
                       user_weights: np.ndarray) -> Dict:
        """
        Solve optimization for a single time slot
        
        Args:
            t: Current time slot
            user_positions: User positions (U, 2)
            user_weights: User priority weights (U,)
            
        Returns:
            Dictionary with optimization results
        """
        current_positions = self.uav_model.get_positions()
        previous_positions = self.positions_history[-1] if len(self.positions_history) > 1 else current_positions
        
        # Warm start
        q_k = current_positions.copy()
        x_k = self.service_placement.copy()
        m_k = self.migration.copy()
        a_k = self.offloading.copy()
        
        converged = False
        iteration = 0
        future_user_positions = self._predict_future_user_positions(user_positions)
        self.planned_user_positions = future_user_positions.copy()
        
        while not converged and iteration < self.max_iter:
            # Block 1: Discrete resource optimization
            channel_gains_current = self.comm_model.compute_channel_gain(q_k, user_positions)
            if self.H_mpc > 1:
                aggregated_gains = self._aggregate_channel_gains(q_k, future_user_positions)
            else:
                aggregated_gains = channel_gains_current

            transmission_rates = self.comm_model.compute_transmission_rate(
                aggregated_gains, self.bandwidth_allocation
            )
            
            discrete_result = self.discrete_optimizer.optimize(
                channel_gains=aggregated_gains,
                transmission_rates=transmission_rates,
                service_sizes=self.config['service_sizes'],
                storage_capacities=self.config['storage_capacities'],
                compute_capacities=self.config['compute_capacities'],
                data_size=self.config['data_size'],
                cpu_cycles_per_bit=self.config['cpu_cycles_per_bit'],
                max_delay=self.config['max_delay'],
                current_x=x_k,
                current_m=m_k,
                current_a=a_k
            )
            
            x_k = discrete_result['x'].astype(float)
            m_k = discrete_result['m']
            a_k = discrete_result['a']
            self.bandwidth_allocation = discrete_result['eta']
            self.compute_frequency = discrete_result['f']
            
            # Block 2: Trajectory optimization
            trajectory_result = self.trajectory_optimizer.optimize(
                current_positions=q_k,
                user_positions=user_positions,
                user_weights=user_weights,
                channel_gains=channel_gains_current,
                previous_positions=previous_positions
            )
            
            q_k_new = trajectory_result['positions']
            
            # Check convergence
            position_change = np.linalg.norm(q_k_new - q_k)
            if position_change < self.tolerance:
                converged = True
            
            q_k = q_k_new
            iteration += 1
        
        # Round binary variables
        self.service_placement = np.round(x_k).astype(int)
        self.migration = np.round(m_k)
        self.offloading = np.round(a_k)
        
        # Update UAV positions
        q_k = self._enforce_energy_budget(previous_positions, q_k)
        self.uav_model.set_positions(q_k)
        self.positions_history.append(q_k.copy())
        
        # Compute final utility with complete delay calculation
        final_channel_gains = self.comm_model.compute_channel_gain(q_k, user_positions)
        final_transmission_rates = self.comm_model.compute_transmission_rate(
            final_channel_gains, self.bandwidth_allocation
        )
        utility = self.compute_utility(
            self.offloading, 
            final_transmission_rates, 
            user_weights,
            migration=self.migration,
            compute_frequency=self.compute_frequency
        )
        
        return {
            'positions': q_k,
            'service_placement': self.service_placement,
            'migration': self.migration,
            'offloading': self.offloading,
            'utility': utility,
            'iterations': iteration,
            'converged': converged
        }
    
    def run(self,
           area_size: float,
           user_positions: np.ndarray,
           user_weights: np.ndarray = None,
           seed: int = None) -> Dict:
        """
        Run the complete BCD-MPC algorithm
        
        Args:
            area_size: Size of operational area (m)
            user_positions: User positions (U, 2)
            user_weights: User priority weights (U,), defaults to uniform
            seed: Random seed
            
        Returns:
            Dictionary with complete results
        """
        if user_weights is None:
            user_weights = np.ones(self.U) / self.U
        
        # Initialize
        self.initialize(area_size, user_positions, seed)
        
        # Run for all time slots
        results = []
        utilities = []
        
        for t in range(self.T):
            result = self.solve_time_slot(t, user_positions, user_weights)
            results.append(result)
            utilities.append(result['utility'])
        
        # Compute total utility
        total_utility = sum(utilities)
        
        return {
            'results': results,
            'utilities': utilities,
            'total_utility': total_utility,
            'positions_history': self.positions_history,
            'final_positions': self.uav_model.get_positions(),
            'final_service_placement': self.service_placement
        }

