"""
Communication model for UAV-user links
"""

import numpy as np
from typing import Tuple


class CommunicationModel:
    """FDMA communication model with Shannon capacity"""
    
    def __init__(self,
                 bandwidth: float,
                 user_power: float,
                 noise_power: float,
                 path_loss_exponent: float,
                 reference_gain: float,
                 altitude: float):
        """
        Initialize communication model
        
        Args:
            bandwidth: Total bandwidth (Hz)
            user_power: User transmit power (W)
            noise_power: Noise power (W)
            path_loss_exponent: Path loss exponent (typically 2)
            reference_gain: Reference channel gain
            altitude: UAV altitude (m)
        """
        self.B = bandwidth
        self.P_u = user_power
        self.sigma2 = noise_power
        self.alpha = path_loss_exponent
        self.beta0 = reference_gain
        self.H = altitude
    
    def compute_channel_gain(self, 
                            uav_positions: np.ndarray,
                            user_positions: np.ndarray,
                            use_mimo: bool = False) -> np.ndarray:
        """
        Compute channel power gain using free-space path loss model
        Based on mappo-xwc-code: h = p * 1e-3 / (d^2 + H^2) / sigma2
        
        Args:
            uav_positions: UAV positions (N, 2)
            user_positions: User positions (U, 2)
            use_mimo: Whether to use MIMO channel model (for future extension)
            
        Returns:
            Channel gains (U, N)
        """
        U = user_positions.shape[0]
        N = uav_positions.shape[0]
        gains = np.zeros((U, N))
        
        for u in range(U):
            for i in range(N):
                # Horizontal distance
                d_horizontal = np.linalg.norm(uav_positions[i] - user_positions[u])
                # 3D distance squared (matching mappo-xwc-code format)
                d_3d_sq = self.H**2 + d_horizontal**2
                # Channel gain: beta0 / d^alpha
                # In mappo-xwc-code: h = p * 1e-3 / (d^2 + H^2) / sigma2
                # So gain = p * 1e-3 / (d^2 + H^2) = p * 1e-3 / d_3d_sq
                # We use: beta0 / d_3d^(alpha/2) for consistency
                gains[u, i] = self.beta0 / (d_3d_sq ** (self.alpha / 2))
        
        return gains
    
    def compute_transmission_rate(self,
                                 channel_gains: np.ndarray,
                                 bandwidth_allocation: np.ndarray) -> np.ndarray:
        """
        Compute achievable transmission rate using Shannon's formula
        
        Args:
            channel_gains: Channel gains (U, N)
            bandwidth_allocation: Bandwidth allocation fractions (U, N), sum <= 1 per UAV
            
        Returns:
            Transmission rates (U, N) in bits/s
        """
        # SNR = P_u * h / sigma^2
        snr = self.P_u * channel_gains / self.sigma2
        
        # Rate = eta * B * log2(1 + SNR)
        rates = bandwidth_allocation * self.B * np.log2(1 + snr)
        
        return rates
    
    def compute_delay(self,
                     data_size: float,
                     transmission_rate: float) -> float:
        """
        Compute transmission delay
        
        Args:
            data_size: Task data size (bits)
            transmission_rate: Transmission rate (bits/s)
            
        Returns:
            Delay in seconds
        """
        return data_size / (transmission_rate + 1e-10)  # Avoid division by zero

