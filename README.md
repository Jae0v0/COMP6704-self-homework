# Multi-UAV Edge Computing Network Optimization

本项目实现了论文 *Maximizing Total Data Collection in Multi-UAV Edge Networks: A Joint Optimization of Service, Migration, and Trajectory* 所提出的联合优化框架。系统通过在线 BCD-MPC 结构，在每个时隙动态优化服务部署、任务迁移与 UAV 轨迹。

## 功能概览
- **系统三层建模**：UAV 运动/碰撞避免、FDMA 通信、任务迁移与服务缓存、能量/计算/存储约束
- **优化框架**：Block Coordinate Descent + Model Predictive Control，其中离散块采用惩罚 + DC 编程、连续块采用 IRLS-SCA
- **加权对数效用**：目标函数完全遵循论文形式 `Σ_u ω_u log(1 + Σ_{i,s} a_{u,i,s}/T_{u,s})`
- **MPC 预测**：支持配置 `mpc_horizon`，通过预测未来用户位置并聚合信道增益生成更稳健的资源分配
- **能量约束**：旋翼机能量模型 + 自动缩放移动，确保满足 `battery_capacity`，并记录 `energy_history`
- **实验工具**：面积对比、能量分析、敏感性分析及轨迹/性能可视化脚本

## 安装
```bash
git clone https://github.com/YourUsername/UAV-MEC-Project.git
cd UAV-MEC-Project

python -m venv venv
source venv/bin/activate  # Windows 使用 venv\\Scripts\\activate
pip install -r requirements.txt
# 或开发模式安装
pip install -e .
```

## 快速开始
```bash
python run_example.py                        # 缩减规模示例
python main.py --config configs/default_config.yaml
python experiments/run_experiments.py --experiment area_comparison
python experiments/run_experiments.py --experiment all
```

### 配置参数
- `configs/default_config.yaml` 中可调整 UAV 数量、用户数量、区域大小、速度/安全距离、带宽、任务规模等
- 能量模型参数（P0、Pi、Utip、d0、rho、A、s、V0、battery_capacity）可直接修改
- `mpc_horizon`、`max_iterations`、`penalty_factor`、`trajectory_penalty`、`user_prediction_noise` 控制算法行为

## 系统建模
### UAV 运动与碰撞避免
- 速度约束 `‖q_i^t - q_i^{t-1}‖ ≤ V_max Δt`
- 碰撞约束 `‖q_i^t - q_j^t‖ ≥ d_min` 通过 SCA + 松弛变量处理

### 通信模型
- FDMA 带宽分配 `η_{u,i}^t`，约束 `Σ_u η_{u,i}^t ≤ 1`
- 信道增益基于自由空间路径损耗，速率 `R = η B log2(1+SNR)`

### 任务迁移与服务放置
- UAV 缓存 `x_{i,s}` 受存储容量 `M_i^{max}` 限制
- 任务迁移 `m_{i→j,s}` 需满足 `m_{i→j,s} ≤ x_{j,s}`
- 总时延 `T = D/R + T_mig + T_comp`，其中 `T_comp = D C / f`

### 能量模型
- 旋翼机功率：叶片轮廓功率 + 诱导功率 + 寄生功率
- 计算能量 `E_comp = κ f^2 DC`，通信能量 `E_comm = P_comm Δt`
- 每时隙计算能量并更新 `energy_usage`，若超限则缩放对应 UAV 的位移保持能量预算

## 算法实现
1. **Block 1：离散资源优化**
   - 变量：服务缓存 `x`、迁移 `m`、卸载 `a`、带宽 `η`、计算频率 `f`
   - 惩罚项 `λ(x - x^2)` + 泰勒线性化构成 DC 子问题
   - 代理目标结合聚合信道增益、预测带宽，提高 MPC lookahead 表现
2. **Block 2：轨迹优化 (IRLS-SCA)**
   - IRLS 权重 `w_{u,i} = ω_u / (1 + SNR)` 加强公平性
   - SCA 线性化碰撞约束，松弛变量 `δ_{ij}` + 惩罚确保安全
3. **MPC 预测**
   - `mpc_horizon > 1` 时，`_predict_future_user_positions` 生成随机游走预测，并在离散块中聚合未来信道增益
   - 仅执行当前时隙决策，下一时隙滚动更新
4. **加权对数效用**
   - `compute_utility` 精确计算 `Σ_u ω_u log(1 + Σ_{i,s} a_{u,i,s} / T_{u,s})`

## 实验脚本
- `experiments/area_comparison.py`：区域扩大对效用的影响
- `experiments/energy_analysis.py`：任务时长/能量预算敏感性
- `experiments/sensitivity_analysis.py`：用户密度、时间演化与累积效用
- 可视化：`visualization/trajectory_plot.py`、`visualization/performance_plot.py`

## 依赖
- numpy ≥ 1.21.0
- scipy ≥ 1.7.0
- cvxpy ≥ 1.2.0
- matplotlib ≥ 3.4.0
- pyyaml ≥ 5.4.0
- tqdm ≥ 4.62.0
- scikit-learn ≥ 1.0.0

## 许可证
MIT License，仅用于教学与科研目的。
