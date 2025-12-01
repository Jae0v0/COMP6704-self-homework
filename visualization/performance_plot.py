"""
Performance visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_utility_comparison(utilities_dict: dict, save_path: str = None):
    """
    Plot utility comparison across different methods
    
    Args:
        utilities_dict: Dictionary with method names and utility values
        save_path: Path to save figure
    """
    methods = list(utilities_dict.keys())
    utilities = [utilities_dict[m] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, utilities, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylabel('Total Utility', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_utility_over_time(utilities_time: list, save_path: str = None):
    """
    Plot utility evolution over time
    
    Args:
        utilities_time: List of utility values per time slot
        save_path: Path to save figure
    """
    time_slots = list(range(len(utilities_time)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time_slots, utilities_time, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Utility', fontsize=12)
    ax.set_title('Utility Evolution Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Utility over time plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

