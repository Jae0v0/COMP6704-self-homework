"""
Quick example script to run the optimization
"""

import numpy as np
from main import load_and_prepare_config
from src.algorithms.bcd_mpc import BCDMPC
from src.utils.helpers import generate_user_positions

def run_example():
    """Run a quick example"""
    print("="*60)
    print("Multi-UAV Edge Computing Network Optimization - Example")
    print("="*60)
    
    # Load config
    config = load_and_prepare_config('configs/default_config.yaml')
    
    # Reduce problem size for quick demo
    config['num_uavs'] = 10
    config['num_users'] = 20
    config['num_time_slots'] = 10
    
    print(f"\nConfiguration:")
    print(f"  - Number of UAVs: {config['num_uavs']}")
    print(f"  - Number of Users: {config['num_users']}")
    print(f"  - Number of Time Slots: {config['num_time_slots']}")
    print(f"  - Area Size: {config['area_size']}m")
    
    # Generate user positions
    area_size = config['area_size']
    num_users = config['num_users']
    
    hotspot_centers = np.array([
        [area_size * 0.25, area_size * 0.25],
        [area_size * 0.75, area_size * 0.75]
    ])
    
    user_positions = generate_user_positions(
        num_users=num_users,
        area_size=area_size,
        hotspot_centers=hotspot_centers,
        hotspot_std=area_size * 0.1,
        seed=42
    )
    
    user_weights = np.ones(num_users) / num_users
    
    # Initialize and run algorithm
    print("\nInitializing algorithm...")
    algorithm = BCDMPC(config)
    
    print("Running optimization...")
    results = algorithm.run(
        area_size=area_size,
        user_positions=user_positions,
        user_weights=user_weights,
        seed=42
    )
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Total Utility: {results['total_utility']:.2f}")
    print(f"Average Utility per Slot: {np.mean(results['utilities']):.2f}")
    print(f"Final UAV Positions Shape: {results['final_positions'].shape}")
    print("="*60)
    
    # Plot trajectories if matplotlib is available
    try:
        from visualization.trajectory_plot import plot_trajectories
        import os
        os.makedirs('results/figures', exist_ok=True)
        plot_trajectories(
            positions_history=results['positions_history'],
            user_positions=user_positions,
            area_size=area_size,
            title="Optimized UAV Trajectories",
            save_path="results/figures/example_trajectories.png"
        )
        print("\nTrajectory plot saved to results/figures/example_trajectories.png")
    except Exception as e:
        print(f"\nCould not generate trajectory plot: {e}")
    
    return results

if __name__ == '__main__':
    run_example()

