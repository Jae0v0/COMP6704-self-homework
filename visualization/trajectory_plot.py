"""
Trajectory visualization tool
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os


def plot_trajectories(positions_history: list,
                     user_positions: np.ndarray,
                     area_size: float,
                     title: str = "UAV Trajectories",
                     save_path: str = None):
    """
    Plot UAV trajectories
    
    Args:
        positions_history: List of position arrays (T+1, N, 2)
        user_positions: User positions (U, 2)
        area_size: Size of operational area
        title: Plot title
        save_path: Path to save figure
    """
    positions_array = np.array(positions_history)  # (T+1, N, 2)
    T, N, _ = positions_array.shape
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot user positions
    ax.scatter(user_positions[:, 0], user_positions[:, 1],
              c='gray', marker='x', s=100, label='Users', linewidths=2)
    
    # Plot trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    
    for i in range(N):
        trajectory = positions_array[:, i, :]  # (T+1, 2)
        
        # Create line segments for trajectory
        points = trajectory.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=[colors[i]], alpha=0.6, linewidth=1.5)
        ax.add_collection(lc)
        
        # Mark start and end positions
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                  c=[colors[i]], marker='o', s=100, edgecolors='black', linewidths=1)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                  c=[colors[i]], marker='s', s=100, edgecolors='black', linewidths=1)
    
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_trajectories(results_dict: dict,
                              user_positions: np.ndarray,
                              area_size: float,
                              save_path: str = None):
    """
    Plot multiple trajectory results for comparison
    
    Args:
        results_dict: Dictionary with method names as keys and positions_history as values
        user_positions: User positions (U, 2)
        area_size: Size of operational area
        save_path: Path to save figure
    """
    num_methods = len(results_dict)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    method_names = list(results_dict.keys())
    
    for idx, method_name in enumerate(method_names):
        if idx >= 4:
            break
        
        ax = axes[idx]
        positions_history = results_dict[method_name]
        positions_array = np.array(positions_history)
        T, N, _ = positions_array.shape
        
        # Plot user positions
        ax.scatter(user_positions[:, 0], user_positions[:, 1],
                  c='gray', marker='x', s=80, linewidths=1.5)
        
        # Plot trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, N))
        
        for i in range(N):
            trajectory = positions_array[:, i, :]
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   c=colors[i], alpha=0.6, linewidth=1)
            
            # Mark start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      c=[colors[i]], marker='o', s=60, edgecolors='black', linewidths=0.5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      c=[colors[i]], marker='s', s=60, edgecolors='black', linewidths=0.5)
        
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title(method_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(len(method_names), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('..')
    
    # Load example data
    positions_history = [np.random.uniform(0, 1000, size=(30, 2)) for _ in range(31)]
    user_positions = np.random.uniform(0, 1000, size=(80, 2))
    
    plot_trajectories(
        positions_history=positions_history,
        user_positions=user_positions,
        area_size=1000.0,
        title="Example UAV Trajectories",
        save_path="results/figures/trajectory_example.png"
    )

