"""
Sensitivity analysis experiment
"""

import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.bcd_mpc import BCDMPC
from src.utils.helpers import generate_user_positions
import os


def run_sensitivity_analysis(config, seed=42):
    """Run sensitivity analysis"""
    area_size = config['area_size']
    
    # Vary user density
    user_counts = [40, 60, 80, 100, 120, 140]
    utilities_users = []
    
    hotspot_centers = np.array([
        [area_size * 0.25, area_size * 0.25],
        [area_size * 0.75, area_size * 0.25],
        [area_size * 0.25, area_size * 0.75],
        [area_size * 0.75, area_size * 0.75]
    ])
    
    for num_users in user_counts:
        print(f"Testing user count: {num_users}")
        config['num_users'] = num_users
        
        user_positions = generate_user_positions(
            num_users=num_users,
            area_size=area_size,
            hotspot_centers=hotspot_centers,
            hotspot_std=area_size * 0.1,
            seed=seed
        )
        
        user_weights = np.ones(num_users) / num_users
        
        algorithm = BCDMPC(config)
        result = algorithm.run(
            area_size=area_size,
            user_positions=user_positions,
            user_weights=user_weights,
            seed=seed
        )
        utilities_users.append(result['total_utility'])
    
    # Reset user count
    config['num_users'] = 80
    
    # Time evolution
    config['num_time_slots'] = 30
    num_users = config['num_users']
    
    user_positions = generate_user_positions(
        num_users=num_users,
        area_size=area_size,
        hotspot_centers=hotspot_centers,
        hotspot_std=area_size * 0.1,
        seed=seed
    )
    
    user_weights = np.ones(num_users) / num_users
    
    algorithm = BCDMPC(config)
    result = algorithm.run(
        area_size=area_size,
        user_positions=user_positions,
        user_weights=user_weights,
        seed=seed
    )
    
    utilities_time = result['utilities']
    time_slots = list(range(len(utilities_time)))
    
    # Plot results
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # User density
    axes[0, 0].plot(user_counts, utilities_users, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Users', fontsize=12)
    axes[0, 0].set_ylabel('Total Utility', fontsize=12)
    axes[0, 0].set_title('Scalability vs. User Density', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time evolution
    axes[0, 1].plot(time_slots, utilities_time, '-', linewidth=2)
    axes[0, 1].set_xlabel('Time Slot', fontsize=12)
    axes[0, 1].set_ylabel('Cumulative Utility', fontsize=12)
    axes[0, 1].set_title('Cumulative Utility Evolution', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative utility
    cumulative_utility = np.cumsum(utilities_time)
    axes[1, 0].plot(time_slots, cumulative_utility, '-', linewidth=2)
    axes[1, 0].set_xlabel('Time Slot', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Utility', fontsize=12)
    axes[1, 0].set_title('Cumulative Utility Over Time', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Utility per slot (bar chart)
    axes[1, 1].bar(time_slots, utilities_time, alpha=0.7)
    axes[1, 1].set_xlabel('Time Slot', fontsize=12)
    axes[1, 1].set_ylabel('Utility per Slot', fontsize=12)
    axes[1, 1].set_title('Utility per Time Slot', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to results/figures/sensitivity_analysis.png")
    
    return {
        'user_counts': user_counts,
        'utilities_users': utilities_users,
        'utilities_time': utilities_time,
        'cumulative_utility': cumulative_utility
    }

