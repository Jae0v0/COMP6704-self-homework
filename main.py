"""
Main entry point for Multi-UAV Edge Computing Network Optimization
"""

import argparse
import numpy as np
from src.algorithms.bcd_mpc import BCDMPC
from src.utils.helpers import load_config, generate_user_positions, create_default_config
import yaml


def load_and_prepare_config(config_path: str = None) -> dict:
    """
    Load configuration and prepare it for the algorithm
    
    Args:
        config_path: Path to config file, None for default
        
    Returns:
        Prepared configuration dictionary
    """
    if config_path:
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using default config")
            config = create_default_config()
    else:
        config = create_default_config()
    
    # Ensure arrays are properly set
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


def main():
    parser = argparse.ArgumentParser(
        description='Multi-UAV Edge Computing Network Optimization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_and_prepare_config(args.config)
    
    # Generate user positions (clustered around hotspots)
    print("Generating user positions...")
    area_size = config['area_size']
    num_users = config['num_users']
    
    # Create hotspot centers
    num_hotspots = 4
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
        seed=args.seed
    )
    
    # User weights (uniform for now)
    user_weights = np.ones(num_users) / num_users
    
    # Initialize algorithm
    print("Initializing BCD-MPC algorithm...")
    algorithm = BCDMPC(config)
    
    # Run algorithm
    print("Running optimization...")
    results = algorithm.run(
        area_size=area_size,
        user_positions=user_positions,
        user_weights=user_weights,
        seed=args.seed
    )
    
    # Print results
    print("\n" + "="*50)
    print("Optimization Results")
    print("="*50)
    print(f"Total Utility: {results['total_utility']:.2f}")
    print(f"Average Utility per Time Slot: {np.mean(results['utilities']):.2f}")
    print(f"Final UAV Positions: {results['final_positions'].shape}")
    print("="*50)
    
    # Save results
    import os
    os.makedirs(args.output, exist_ok=True)
    
    # Save positions history
    np.save(
        os.path.join(args.output, 'positions_history.npy'),
        np.array(results['positions_history'])
    )
    
    # Save utilities
    np.save(
        os.path.join(args.output, 'utilities.npy'),
        np.array(results['utilities'])
    )
    
    print(f"\nResults saved to {args.output}/")
    
    return results


if __name__ == '__main__':
    main()

