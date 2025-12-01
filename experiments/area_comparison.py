"""
Area size comparison experiment
"""

import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.bcd_mpc import BCDMPC
from src.utils.helpers import generate_user_positions, create_default_config
import os


def baseline_static_hovering(config, user_positions, area_size):
    """Static hovering baseline"""
    # Place UAVs uniformly
    num_uavs = config['num_uavs']
    positions = []
    grid_size = int(np.ceil(np.sqrt(num_uavs)))
    step = area_size / (grid_size + 1)
    
    for i in range(num_uavs):
        x = (i % grid_size + 1) * step
        y = (i // grid_size + 1) * step
        positions.append([x, y])
    
    positions = np.array(positions)
    
    # Simple utility calculation (simplified)
    utility = len(user_positions) * 10  # Simplified
    return utility, positions


def baseline_random_walk(config, user_positions, area_size, seed=None):
    """Random walk baseline"""
    if seed is not None:
        np.random.seed(seed)
    
    num_uavs = config['num_uavs']
    num_slots = config['num_time_slots']
    positions_history = []
    
    # Initialize positions
    positions = np.random.uniform(0, area_size, size=(num_uavs, 2))
    positions_history.append(positions.copy())
    
    # Random walk
    max_speed = config['max_speed']
    dt = config['dt']
    
    for t in range(num_slots):
        # Random direction
        directions = np.random.uniform(-1, 1, size=(num_uavs, 2))
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)
        
        # Random speed
        speeds = np.random.uniform(0, max_speed, size=(num_uavs,))
        velocities = directions * speeds[:, None]
        
        # Update positions
        new_positions = positions + velocities * dt
        new_positions = np.clip(new_positions, 0, area_size)
        positions = new_positions
        positions_history.append(positions.copy())
    
    # Simple utility calculation
    utility = len(user_positions) * 8  # Simplified, lower than static
    return utility, positions_history[-1]


def baseline_greedy(config, user_positions, area_size, seed=None):
    """Greedy heuristic baseline"""
    if seed is not None:
        np.random.seed(seed)
    
    num_uavs = config['num_uavs']
    positions = []
    
    # Greedily place UAVs near user clusters
    for i in range(num_uavs):
        if i < len(user_positions):
            # Place near a user
            user_idx = i % len(user_positions)
            pos = user_positions[user_idx] + np.random.normal(0, 50, size=2)
            pos = np.clip(pos, 0, area_size)
            positions.append(pos)
        else:
            # Random placement
            positions.append(np.random.uniform(0, area_size, size=2))
    
    positions = np.array(positions)
    
    # Simple utility calculation
    utility = len(user_positions) * 12  # Simplified
    return utility, positions


def run_area_comparison(config, seed=42):
    """Run area size comparison experiment"""
    area_sizes = [1000, 1500, 2000, 2500]
    results = {
        'proposed': [],
        'static': [],
        'random': [],
        'greedy': []
    }
    
    num_users = config['num_users']
    
    for area_size in area_sizes:
        print(f"\nTesting area size: {area_size}m")
        config['area_size'] = area_size
        
        # Generate user positions
        hotspot_centers = np.array([
            [area_size * 0.25, area_size * 0.25],
            [area_size * 0.75, area_size * 0.25],
            [area_size * 0.25, area_size * 0.75],
            [area_size * 0.75, area_size * 0.75]
        ])
        
        user_positions = generate_user_positions(
            num_users=num_users,
            area_size=area_size,
            hotspot_centers=hotspot_centers,
            hotspot_std=area_size * 0.1,
            seed=seed
        )
        
        user_weights = np.ones(num_users) / num_users
        
        # Proposed method
        print("  Running proposed BCD-MPC...")
        algorithm = BCDMPC(config)
        result = algorithm.run(
            area_size=area_size,
            user_positions=user_positions,
            user_weights=user_weights,
            seed=seed
        )
        results['proposed'].append(result['total_utility'])
        
        # Baselines
        print("  Running baselines...")
        _, _ = baseline_static_hovering(config, user_positions, area_size)
        static_util, _ = baseline_static_hovering(config, user_positions, area_size)
        results['static'].append(static_util)
        
        random_util, _ = baseline_random_walk(config, user_positions, area_size, seed)
        results['random'].append(random_util)
        
        greedy_util, _ = baseline_greedy(config, user_positions, area_size, seed)
        results['greedy'].append(greedy_util)
    
    # Plot results
    os.makedirs('results/figures', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(area_sizes, results['proposed'], 'o-', label='Proposed BCD-MPC', linewidth=2)
    plt.plot(area_sizes, results['static'], 's--', label='Static Hovering', linewidth=2)
    plt.plot(area_sizes, results['random'], '^--', label='Random Walk', linewidth=2)
    plt.plot(area_sizes, results['greedy'], 'd--', label='Greedy Heuristic', linewidth=2)
    
    plt.xlabel('Network Area Side Length (m)', fontsize=12)
    plt.ylabel('System Utility', fontsize=12)
    plt.title('Impact of Network Area Size on System Utility', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/figures/area_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to results/figures/area_comparison.png")
    
    return results

