"""
Energy consumption model for UAVs
Based on rotary-wing UAV power consumption model from mappo-xwc-code
"""

import numpy as np


class EnergyModel:
    """UAV energy consumption model including propulsion, computation and communication"""
    
    def __init__(self,
                 blade_power: float = 59.03,
                 induced_power: float = 79.07,
                 tip_speed: float = 120.0,
                 fuselage_drag_coefficient: float = 0.6,
                 air_density: float = 1.225,
                 rotor_disk_area: float = 0.5030,
                 rotor_solidity: float = 0.05,
                 mean_rotor_induced_velocity: float = 3.6,
                 computation_coefficient: float = 1e-27,
                 communication_power: float = 10.0):
        """
        Initialize energy model with parameters from mappo-xwc-code
        
        Args:
            blade_power: Blade profile power constant P0 (W), default 59.03
            induced_power: Induced power constant Pi (W), default 79.07
            tip_speed: Tip speed of rotor Utip (m/s), default 120
            fuselage_drag_coefficient: Fuselage drag coefficient d0, default 0.6
            air_density: Air density rho (kg/m^3), default 1.225
            rotor_disk_area: Rotor disk area A (m^2), default 0.5030
            rotor_solidity: Rotor solidity s, default 0.05
            mean_rotor_induced_velocity: Mean rotor induced velocity V0 (m/s), default 3.6
            computation_coefficient: Computation energy coefficient kappa (J·s^2/cycle^3), default 1e-27
            communication_power: Communication power consumption (W), default 10.0
        """
        self.P0 = blade_power
        self.Pi = induced_power
        self.Utip = tip_speed
        self.d0 = fuselage_drag_coefficient
        self.rho = air_density
        self.A = rotor_disk_area
        self.s = rotor_solidity
        self.V0 = mean_rotor_induced_velocity
        self.kappa = computation_coefficient
        self.P_comm = communication_power
    
    def compute_propulsion_power(self, velocity: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Compute propulsion power using rotary-wing model
        Based on: P_fly = P0*(1+3*v^2/Utip^2) + Pi*(sqrt(1+v^4/(4*V0^4))-v^2/(2*V0^2))^0.5 
                  + 0.5*d0*rho*s*A*|v|^3
        
        Args:
            velocity: UAV velocities (N, 2) in m/s or (N,) speed values
            dt: Time slot duration (s)
            
        Returns:
            Propulsion power for each UAV (N,)
        """
        if len(velocity.shape) == 2:
            speeds = np.linalg.norm(velocity, axis=1)  # (N,)
        else:
            speeds = np.abs(velocity)  # (N,)
        
        # Blade profile power
        P_profile = self.P0 * (1 + 3 * speeds**2 / self.Utip**2)
        
        # Induced power
        v4_term = speeds**4 / (4 * self.V0**4)
        v2_term = speeds**2 / (2 * self.V0**2)
        P_induced = self.Pi * np.sqrt(np.maximum(1 + v4_term - v2_term, 0))
        
        # Parasite power
        P_parasite = 0.5 * self.d0 * self.rho * self.s * self.A * np.abs(speeds)**3
        
        power = P_profile + P_induced + P_parasite
        
        return power
    
    def compute_propulsion_energy(self, velocity: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Compute propulsion energy consumption
        
        Args:
            velocity: UAV velocities (N, 2) or (N,)
            dt: Time slot duration (s)
            
        Returns:
            Propulsion energy for each UAV (N,)
        """
        power = self.compute_propulsion_power(velocity, dt)
        return power * dt
    
    def compute_computation_energy(self, 
                                   frequency: np.ndarray,
                                   cpu_cycles: np.ndarray) -> np.ndarray:
        """
        Compute computation energy consumption
        E_comp = kappa * f^2 * DC
        Based on mappo-xwc-code: calc_comp_energy(f, DC) = RC * f^2 * DC
        
        Args:
            frequency: Computation frequency (Hz) - shape (N,) or (N, S)
            cpu_cycles: Total CPU cycles (cycles) - shape (N,) or (N, S)
            
        Returns:
            Computation energy (J)
        """
        # Ensure same shape
        if frequency.shape != cpu_cycles.shape:
            if len(frequency.shape) == 1 and len(cpu_cycles.shape) == 2:
                frequency = frequency[:, np.newaxis]
            elif len(frequency.shape) == 2 and len(cpu_cycles.shape) == 1:
                cpu_cycles = cpu_cycles[:, np.newaxis]
        
        energy = self.kappa * frequency**2 * cpu_cycles
        return np.sum(energy, axis=-1) if len(energy.shape) > 1 else energy
    
    def compute_total_energy(self,
                            velocities: np.ndarray,
                            computation_frequencies: np.ndarray = None,
                            cpu_cycles: np.ndarray = None,
                            dt: float = 1.0) -> np.ndarray:
        """
        Compute total energy consumption (propulsion + computation + communication)
        
        Args:
            velocities: UAV velocities over time (T, N, 2) or (T, N)
            computation_frequencies: Computation frequencies (T, N) or (N,), optional
            cpu_cycles: CPU cycles (T, N) or (N,), optional
            dt: Time slot duration (s)
            
        Returns:
            Total energy consumption for each UAV (N,)
        """
        T = velocities.shape[0]
        N = velocities.shape[1] if len(velocities.shape) > 1 else velocities.shape[0]
        total_energy = np.zeros(N)
        
        for t in range(T):
            v_t = velocities[t] if len(velocities.shape) > 1 else velocities[t:t+1]
            
            # Propulsion energy
            propulsion_energy = self.compute_propulsion_energy(v_t, dt)
            total_energy += propulsion_energy
            
            # Computation energy
            if computation_frequencies is not None and cpu_cycles is not None:
                f_t = computation_frequencies[t] if len(computation_frequencies.shape) > 1 else computation_frequencies
                c_t = cpu_cycles[t] if len(cpu_cycles.shape) > 1 else cpu_cycles
                comp_energy = self.compute_computation_energy(f_t, c_t)
                total_energy += comp_energy
            
            # Communication energy
            total_energy += self.P_comm * dt
        
        return total_energy
    
    def compute_total_energy(self,
                            velocities: np.ndarray,
                            dt: float) -> np.ndarray:
        """
        Compute total energy consumption (propulsion + communication)
        
        Args:
            velocities: UAV velocities over time (T, N, 2)
            dt: Time slot duration (s)
            
        Returns:
            Total energy consumption for each UAV (N,)
        """
        T = velocities.shape[0]
        N = velocities.shape[1]
        total_energy = np.zeros(N)
        
        for t in range(T):
            propulsion_power = self.compute_propulsion_power(velocities[t])
            total_power = propulsion_power + self.P_comm
            total_energy += total_power * dt
        
        return total_energy
    
    def check_energy_constraint(self,
                               total_energy: np.ndarray,
                               battery_capacity: float) -> Tuple[bool, np.ndarray]:
        """
        Check if energy constraints are satisfied
        
        Args:
            total_energy: Total energy consumption (N,)
            battery_capacity: Battery capacity (J)
            
        Returns:
            (is_valid, violations)
        """
        violations = np.maximum(0, total_energy - battery_capacity)
        is_valid = np.all(violations == 0)
        return is_valid, violations

