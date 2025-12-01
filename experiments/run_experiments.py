"""
Main experiment runner
"""

import argparse
import numpy as np
from src.algorithms.bcd_mpc import BCDMPC
from src.utils.helpers import load_config, generate_user_positions, create_default_config
from .area_comparison import run_area_comparison
from .energy_analysis import run_energy_analysis
from .sensitivity_analysis import run_sensitivity_analysis


def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['area_comparison', 'energy_analysis', 'sensitivity', 'all'],
        default='all',
        help='Experiment to run'
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
    
    args = parser.parse_args()
    
    if args.config:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            print(f"Config file {args.config} not found, using default config")
            config = create_default_config()
    else:
        config = create_default_config()
    
    if args.experiment == 'area_comparison' or args.experiment == 'all':
        print("Running area comparison experiment...")
        run_area_comparison(config, seed=args.seed)
    
    if args.experiment == 'energy_analysis' or args.experiment == 'all':
        print("Running energy analysis experiment...")
        run_energy_analysis(config, seed=args.seed)
    
    if args.experiment == 'sensitivity' or args.experiment == 'all':
        print("Running sensitivity analysis experiment...")
        run_sensitivity_analysis(config, seed=args.seed)
    
    print("All experiments completed!")


if __name__ == '__main__':
    main()

