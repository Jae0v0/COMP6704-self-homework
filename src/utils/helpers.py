"""
Helper utility functions
"""

import numpy as np
import yaml
from typing import Dict, Tuple


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    import numpy as np
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert scalar values to arrays where needed
    if isinstance(config.get('storage_capacities'), (int, float)):
        config['storage_capacities'] = np.full(
            config['num_uavs'], config['storage_capacities']
        )
    
    if isinstance(config.get('compute_capacities'), (int, float)):
        config['compute_capacities'] = np.full(
            config['num_uavs'], config['compute_capacities']
        )
    
    # Convert service_sizes to numpy array
    if isinstance(config.get('service_sizes'), list):
        config['service_sizes'] = np.array(config['service_sizes'])
    
    return config


def generate_user_positions(num_users: int,
                           area_size: float,
                           hotspot_centers: np.ndarray = None,
                           hotspot_std: float = 100.0,
                           seed: int = None) -> np.ndarray:
    """
    Generate user positions, optionally clustered around hotspots
    
    Args:
        num_users: Number of users
        area_size: Size of operational area (m)
        hotspot_centers: Hotspot center positions (K, 2), None for uniform distribution
        hotspot_std: Standard deviation for hotspot distribution (m)
        seed: Random seed
        
    Returns:
        User positions (U, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if hotspot_centers is None:
        # Uniform random distribution
        positions = np.random.uniform(0, area_size, size=(num_users, 2))
    else:
        # Clustered distribution around hotspots
        K = hotspot_centers.shape[0]
        users_per_hotspot = num_users // K
        positions = []
        
        for k in range(K):
            center = hotspot_centers[k]
            hotspot_positions = np.random.normal(
                center, hotspot_std, size=(users_per_hotspot, 2)
            )
            # Clip to area bounds
            hotspot_positions = np.clip(hotspot_positions, 0, area_size)
            positions.append(hotspot_positions)
        
        # Add remaining users uniformly
        remaining = num_users - K * users_per_hotspot
        if remaining > 0:
            uniform_positions = np.random.uniform(
                0, area_size, size=(remaining, 2)
            )
            positions.append(uniform_positions)
        
        positions = np.vstack(positions)
    
    return positions


def compute_energy_consumption(positions_history: list,
                              dt: float,
                              energy_model) -> np.ndarray:
    """
    Compute energy consumption from position history
    
    Args:
        positions_history: List of position arrays (T+1, N, 2)
        dt: Time slot duration (s)
        energy_model: EnergyModel instance
        
    Returns:
        Total energy consumption per UAV (N,)
    """
    T = len(positions_history) - 1
    if T == 0:
        return np.zeros(energy_model.N)
    
    velocities = []
    for t in range(T):
        v_t = (positions_history[t+1] - positions_history[t]) / dt
        velocities.append(v_t)
    
    velocities = np.array(velocities)  # (T, N, 2)
    total_energy = energy_model.compute_total_energy(velocities, dt)
    
    return total_energy


def create_default_config() -> Dict:
    """
    Create default configuration dictionary
    
    Returns:
        Default configuration
    """
    config = {
        # System parameters
        'num_uavs': 30,
        'num_users': 80,
        'num_services': 5,
        'num_time_slots': 30,
        'area_size': 1000.0,  # meters
        
        # UAV parameters
        'max_speed': 20.0,  # m/s
        'min_distance': 50.0,  # meters
        'altitude': 100.0,  # meters
        'dt': 1.0,  # seconds
        
        # Communication parameters
        'bandwidth': 1e6,  # Hz
        'user_power': 0.1,  # W
        'noise_power': 1e-10,  # W
        'path_loss_exponent': 2.0,
        'reference_gain': 1e-3,
        'backhaul_rate': 10e6,  # bits/s
        
        # Task parameters
        'data_size': 1e6,  # bits
        'cpu_cycles_per_bit': 1000.0,
        'max_delay': 1.0,  # seconds
        'service_sizes': np.array([10, 20, 15, 25, 30]),  # MB
        'storage_capacities': np.full(30, 100),  # MB per UAV
        'compute_capacities': np.full(30, 1e9),  # Hz per UAV
        
        # Energy parameters (based on mappo-xwc-code)
        'battery_capacity': 5000.0,  # J
        'blade_power': 59.03,  # P0 (W)
        'induced_power': 79.07,  # Pi (W)
        'tip_speed': 120.0,  # Utip (m/s)
        'fuselage_drag_coefficient': 0.6,  # d0
        'air_density': 1.225,  # rho (kg/m^3)
        'rotor_disk_area': 0.5030,  # A (m^2)
        'rotor_solidity': 0.05,  # s
        'mean_rotor_induced_velocity': 3.6,  # V0 (m/s)
        'computation_coefficient': 1e-27,  # kappa (J·s^2/cycle^3)
        'communication_power': 10.0,  # W
        
        # Algorithm parameters
        'mpc_horizon': 5,
        'max_iterations': 10,
        'tolerance': 1e-3,
        'penalty_factor': 100.0,
        'trajectory_penalty': 1000.0
    }
    
    return config

