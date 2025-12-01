"""
Energy consumption and robustness analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.bcd_mpc import BCDMPC
from src.utils.helpers import generate_user_positions
import os


def run_energy_analysis(config, seed=42):
    """Run energy consumption analysis"""
    # Vary mission duration
    durations = [10, 20, 30, 40, 50]
    utilities_duration = []
    
    area_size = config['area_size']
    num_users = config['num_users']
    
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
    
    for duration in durations:
        print(f"Testing mission duration: {duration} slots")
        config['num_time_slots'] = duration
        
        algorithm = BCDMPC(config)
        result = algorithm.run(
            area_size=area_size,
            user_positions=user_positions,
            user_weights=user_weights,
            seed=seed
        )
        utilities_duration.append(result['total_utility'])
    
    # Vary energy budget
    energy_budgets = [0.6, 0.8, 1.0, 1.2, 1.4]
    utilities_energy = []
    
    base_energy = config['battery_capacity']
    config['num_time_slots'] = 30  # Reset
    
    for energy_factor in energy_budgets:
        print(f"Testing energy budget factor: {energy_factor}")
        config['battery_capacity'] = base_energy * energy_factor
        
        algorithm = BCDMPC(config)
        result = algorithm.run(
            area_size=area_size,
            user_positions=user_positions,
            user_weights=user_weights,
            seed=seed
        )
        utilities_energy.append(result['total_utility'])
    
    # Reset energy budget
    config['battery_capacity'] = base_energy
    
    # Plot results
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mission duration
    axes[0].plot(durations, utilities_duration, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Mission Duration (time slots)', fontsize=12)
    axes[0].set_ylabel('Total Utility', fontsize=12)
    axes[0].set_title('Impact of Mission Duration', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Energy budget
    axes[1].plot(energy_budgets, utilities_energy, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Normalized Energy Budget', fontsize=12)
    axes[1].set_ylabel('Total Utility', fontsize=12)
    axes[1].set_title('Impact of Energy Budget', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/energy_analysis.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to results/figures/energy_analysis.png")
    
    return {
        'durations': durations,
        'utilities_duration': utilities_duration,
        'energy_budgets': energy_budgets,
        'utilities_energy': utilities_energy
    }

