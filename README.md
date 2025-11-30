# Multi-UAV Edge Computing Network Optimization

This repository implements the framework described in *Maximizing Total Data Collection in Multi-UAV Edge Networks: A Joint Optimization of Service, Migration, and Trajectory*. The system performs online joint optimization of service placement, task migration, and UAV trajectories through a Block Coordinate Descent (BCD) and Model Predictive Control (MPC) pipeline.

## Key Features
- **Full-stack modeling**: UAV mobility & collision avoidance, FDMA communication, service caching & task migration, and energy/compute/storage constraints.
- **Optimization pipeline**: BCD + MPC, where the discrete block uses penalty + DC programming and the continuous block uses IRLS-SCA.
- **Weighted log-sum utility**: Objective `Σ_u ω_u log(1 + Σ_{i,s} a_{u,i,s}/T_{u,s})` matches the paper exactly.
- **Predictive MPC**: Configurable `mpc_horizon` with future user prediction and aggregated channel gains to stabilize resource allocation.
- **Energy enforcement**: Rotorcraft energy model + automatic movement scaling guarantee compliance with `battery_capacity`, while `energy_history` keeps battery usage logs.
- **Experiment toolkit**: Area comparison, energy sensitivity, comprehensive sensitivity analysis, and visualization scripts.

## Installation
```bash
git clone https://github.com/YourUsername/UAV-MEC-Project.git
cd UAV-MEC-Project

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# or install in editable mode
pip install -e .
```

## Quick Start
```bash
python run_example.py                        # small-scale demo
python main.py --config configs/default_config.yaml
python experiments/run_experiments.py --experiment area_comparison
python experiments/run_experiments.py --experiment all
```

### Configuration Highlights
- `configs/default_config.yaml` exposes UAV/user counts, area size, velocity/safety constraints, bandwidth, task profiles, etc.
- Energy model parameters (P0, Pi, Utip, d0, rho, A, s, V0, battery_capacity) can be changed directly.
- Algorithm knobs: `mpc_horizon`, `max_iterations`, `penalty_factor`, `trajectory_penalty`, `user_prediction_noise`.

## System Modeling
### UAV Mobility & Collision Avoidance
- Velocity constraint `‖q_i^t - q_i^{t-1}‖ ≤ V_max Δt`.
- Collision avoidance `‖q_i^t - q_j^t‖ ≥ d_min`, enforced via SCA with slack variables.

### Communication Model
- FDMA bandwidth allocation `η_{u,i}^t` with `Σ_u η_{u,i}^t ≤ 1`.
- Channel gains follow free-space path loss; rate `R = η B log2(1+SNR)`.

### Task Migration & Service Placement
- Service cache `x_{i,s}` limited by `M_i^{max}`.
- Migration decisions `m_{i→j,s}` must satisfy `m_{i→j,s} ≤ x_{j,s}`.
- Total latency `T = D/R + T_mig + T_comp`, with `T_comp = D C / f`.

### Energy Model
- Rotorcraft power = profile power + induced power + parasite power.
- Computation energy `E_comp = κ f^2 DC`, communication energy `E_comm = P_comm Δt`.
- After each slot, `_enforce_energy_budget` recomputes energy usage and scales per-UAV movement when necessary.

## Algorithm Overview
1. **Block 1 – Discrete Resource Optimization**
   - Variables: service cache `x`, migration `m`, offloading `a`, bandwidth `η`, compute frequency `f`.
   - Penalty term `λ(x - x^2)` with Taylor linearization implements the DC subproblem.
   - Surrogate objective leverages aggregated channel gains from MPC lookahead.
2. **Block 2 – Trajectory Optimization (IRLS-SCA)**
   - IRLS weights `w_{u,i} = ω_u / (1 + SNR)` emphasize fairness.
   - SCA linearizes collision constraints; slack `δ_{ij}` plus penalties ensures safety.
3. **Predictive MPC**
   - When `mpc_horizon > 1`, `_predict_future_user_positions` generates random-walk forecasts; `_aggregate_channel_gains` averages them for Block 1.
   - Only the current slot decision is executed; future steps are re-optimized online.
4. **Weighted Log-Sum Utility**
   - `compute_utility` evaluates `Σ_u ω_u log(1 + Σ_{i,s} a_{u,i,s} / T_{u,s})` with exact transmission, migration, and computation delays.

## Experiment Scripts
- `experiments/area_comparison.py`: impact of area size on utility.
- `experiments/energy_analysis.py`: mission duration & battery budget sensitivity.
- `experiments/sensitivity_analysis.py`: user density scaling, temporal evolution, cumulative utility.
- Visualization tools: `visualization/trajectory_plot.py`, `visualization/performance_plot.py`.

## Dependencies
- numpy ≥ 1.21.0
- scipy ≥ 1.7.0
- cvxpy ≥ 1.2.0
- matplotlib ≥ 3.4.0
- pyyaml ≥ 5.4.0
- tqdm ≥ 4.62.0
- scikit-learn ≥ 1.0.0

## License
MIT License. For academic and research use only.
