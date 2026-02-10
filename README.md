# Compositional Neural Architecture Search for Multi-Task RL State Encoding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **研究目标**: 自动发现强化学习中多任务状态编码的最优模块化架构

---

## 📖 项目概述

### 核心问题

在多任务强化学习中，不同任务可能依赖状态空间的不同子集。例如：
- **任务A**可能只需要位置信息
- **任务B**可能只需要速度信息
- **任务C**可能需要位置+角度信息

**关键挑战**: 如何自动发现每个任务应该使用哪些状态特征？

### 我们的方法

本项目实现了一个**模块化神经架构搜索（NAS）框架**，用于：

1. 将状态空间分解为独立的**状态块（State Blocks）**
2. 为每个状态块学习独立的**编码模块（Encoder Modules）**
3. 通过**架构搜索**自动发现每个任务的最优模块组合
4. 使用**强化学习**训练和评估候选架构

### 当前版本

⚠️ **注意**: 本仓库实现的是**简化版本**，用于验证核心思想：
- 使用 **Contextual Bandit**（单步MDP）而非完整MDP
- 使用 **合成数据**而非真实环境
- 使用 **随机搜索**而非高级NAS算法

**最终目标**: 扩展到真实的多任务RL环境（如Meta-World、DeepMind Control Suite）

---

## 🎯 核心概念

### 1. 状态块（State Blocks）

状态被分解为独立的块：`s = (b1, b2, b3, b4)`

| Block | 类型 | 范围 | 说明 |
|-------|------|------|------|
| b1 | 实数 | [-1, 1] | 线性特征 |
| b2 | 实数 | [0, 2π] | 周期性特征（角度）|
| b3 | 二值 | {0, 1} | 离散特征 |
| b4 | 2D向量 | [-1, 1]² | 向量特征 |

### 2. 编码模块（Encoder Modules）

每个状态块有独立的神经网络编码器：

```python
h1 = m1(b1; θ1)  # Block 1 的编码器
h2 = m2(b2; θ2)  # Block 2 的编码器
h3 = m3(b3; θ3)  # Block 3 的编码器
h4 = m4(b4; θ4)  # Block 4 的编码器
```

### 3. 架构（Architecture）

定义每个任务使用哪些模块：

```python
# 示例架构
architecture = {
    'task1': [0, 1],  # Task 1 使用 b1 和 b2
    'task2': [2],     # Task 2 使用 b3
    'task3': [0, 3],  # Task 3 使用 b1 和 b4
}
```

### 4. 架构搜索

在所有可能的架构中找到最优的：

```
搜索空间大小 = (C(4,1) + C(4,2))^3 = 10^3 = 1000 种架构
```

---

## 🚀 快速开始

### 环境配置

#### 1. 克隆仓库
```bash
git clone https://github.com/your-username/rl-architecture-search.git
cd rl-architecture-search
```

#### 2. 创建虚拟环境（推荐）
```bash
# 使用 conda
conda create -n rl-nas python=3.8
conda activate rl-nas

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.21.0
torch>=2.0.0
matplotlib>=3.5.0
jupyter>=1.0.0
tqdm>=4.65.0
```

#### 4. 启动Jupyter
```bash
jupyter notebook
```

---

## 📁 项目结构

```
simulation/
├── Big_README.md                      # 本文件（项目总览）
├── README.md                          # 简要说明
├── CLAUDE.md                          # 项目需求规格
├── requirements.txt                   # Python依赖
│
├── 📓 核心实验 Notebooks
│   ├── phase1_synthetic_mdp.ipynb         # Phase 1: 合成MDP和Ground Truth
│   ├── phase2_fixed_architectures.ipynb   # Phase 2: 固定架构对比
│   └── phase3_rl_training.ipynb           # Phase 3: 架构搜索
│
├── 📄 文档
│   ├── PROJECT_STRUCTURE.md           # 项目结构详细说明
│   ├── RL_VS_SUPERVISED.md            # RL vs 监督学习对比
│   ├── RL_VERSION_SUMMARY.md          # RL版本使用指南
│   ├── PHASE3_GUIDE.md                # Phase 3 详细指南
│   ├── BUGFIX_LOG.md                  # 问题修复记录
│   └── FILE_LIST.txt                  # 文件清单
│
└── 📊 实验结果
    ├── phase1_reward_decomposition.png
    ├── phase2_training_curves.png
    ├── phase2_performance_comparison.png
    ├── phase3_rl_training_curves.png
    ├── phase3_search_progress.png
    └── phase3_performance_distribution.png
```

---

## 🔬 实验说明

### Phase 1: 合成MDP和Ground Truth

**目标**: 理解问题的组合结构

**文件**: `phase1_synthetic_mdp.ipynb`

**内容**:
1. 定义合成MDP环境
2. 实现3个任务的Ground Truth奖励函数：
   - **Task 1**: `r1(s) = b1 + sin(b2)` （依赖b1, b2）
   - **Task 2**: `r2(s) = 2*b3 - 1` （依赖b3）
   - **Task 3**: `r3(s) = b1 + ||b4||` （依赖b1, b4）
3. 可视化奖励分解

**运行时间**: ~1分钟

**关键输出**:
- Ground Truth架构定义
- 奖励函数可视化

---

### Phase 2: 固定架构对比

**目标**: 验证架构选择的重要性

**文件**: `phase2_fixed_architectures.ipynb`

**内容**:
1. **Task 2.1 - 单体基线（Monolithic Baseline）**
   - 所有任务使用全部状态块
   - 无模块化结构

2. **Task 2.2 - 正确的模块化架构**
   - 使用Ground Truth架构
   - 每个任务只使用相关的状态块

3. **Task 2.3 - 错误的架构**
   - 故意使用错误的状态块组合
   - 观察性能下降

**训练方式**: 强化学习（Experience Replay + TD Learning）

**运行时间**: ~5分钟

**关键发现**:
- ✅ 正确架构显著优于单体基线
- ✅ 错误架构性能明显下降
- ✅ 架构选择比训练时间更重要

---

### Phase 3: 架构搜索

**目标**: 自动发现最优架构

**文件**: `phase3_rl_training.ipynb`

**内容**:

#### 3.1 架构搜索空间
- 定义所有可能的架构（1000种）
- 每个任务可使用1-2个状态块

#### 3.2 架构评估函数
```python
def evaluate_architecture_rl(architecture, env, n_episodes=1000):
    # 1. 创建新模型
    model = ModularQNetwork(architecture)

    # 2. 用RL训练
    train_with_rl(model, env, n_episodes=1000)

    # 3. 评估性能（MSE）
    return mse
```

#### 3.3 搜索算法

**随机搜索（Random Search）**:
```python
random_results = random_search(
    arch_space, env,
    n_trials=100,              # 尝试100个架构
    n_episodes_per_arch=1000   # 每个架构训练1000 episodes
)
```

**穷举搜索（Exhaustive Search）**（可选）:
```python
exhaustive_results = exhaustive_search(
    arch_space, env,
    n_episodes_per_arch=1000,
    max_architectures=50       # 评估前50个架构
)
```

**运行时间**:
- 随机搜索（20个架构）: ~40分钟
- 随机搜索（100个架构）: ~3.5小时
- 穷举搜索（1000个架构）: ~35小时

**关键发现**:
- ✅ 随机搜索在10%采样率下找到接近最优架构
- ✅ 最佳架构MSE=0.053 vs Ground Truth MSE=0.045
- ✅ 只有task3的一个block选择错误（b3 vs b4）

---

## 📊 实验结果

### Phase 2 结果

| 架构类型 | MSE | 说明 |
|---------|-----|------|
| 单体基线 | 0.15-0.25 | 使用全部状态块 |
| 正确架构 | 0.04-0.06 | Ground Truth |
| 错误架构 | 0.30-0.50 | 错误的状态块组合 |

### Phase 3 结果（100次随机搜索）

```
最佳架构: task1:b1,b2 | task2:b3 | task3:b1,b3
最佳MSE: 0.053227

Ground Truth: task1:b1,b2 | task2:b3 | task3:b1,b4
GT MSE: 0.044782

差距: 0.008 (19%)
```

**Top 5 架构**:
1. `task1:b1,b2 | task2:b3 | task3:b1,b3` - MSE=0.053 ⭐
2. `task1:b1,b2 | task2:b3,b4 | task3:b2,b4` - MSE=0.115
3. `task1:b2,b3 | task2:b3 | task3:b1,b4` - MSE=0.149
4. `task1:b1,b2 | task2:b3,b4 | task3:b3` - MSE=0.152
5. `task1:b2,b3 | task2:b3,b4 | task3:b1` - MSE=0.171

### 可视化

<img src="simulation code/phase3_search_results.png" width="800">

---

## 🔧 技术细节

### RL训练设置

```python
# 环境
- 类型: Contextual Bandit (单步MDP)
- 状态采样: 独立采样每个block
- 奖励: 确定性（无噪声）

# 模型
- 架构: 模块化Q网络
- 模块: MLP (input → 16 → 8)
- 头部: Linear (concat(modules) → 1)

# 训练
- 算法: TD Learning + Experience Replay
- Optimizer: Adam (lr=1e-3)
- Replay Buffer: 10000
- Batch Size: 64
- Episodes: 1000 per architecture
```

### 架构搜索设置

```python
# 搜索空间
- 状态块数: 4
- 每任务最多使用: 2个块
- 总架构数: 1000

# 搜索算法
- 方法: Random Search
- 采样数: 20-100
- 评估指标: MSE (预测Q vs 真实reward)
```

---

## 📈 未来工作

### 短期目标

1. **更高级的搜索算法**
   - [ ] 贝叶斯优化（Bayesian Optimization）
   - [ ] 进化算法（Evolutionary Algorithms）
   - [ ] 强化学习NAS（RL-based NAS）

2. **更复杂的MDP**
   - [ ] 多步MDP（非Bandit）
   - [ ] 状态转移依赖
   - [ ] 折扣因子γ < 1

3. **更大的搜索空间**
   - [ ] 更多状态块（8-16个）
   - [ ] 可变模块大小
   - [ ] 层次化架构

### 长期目标

1. **真实RL环境**
   - [ ] Meta-World（机器人操作）
   - [ ] DeepMind Control Suite
   - [ ] Atari游戏

2. **迁移学习**
   - [ ] 跨任务模块共享
   - [ ] 零样本泛化
   - [ ] 少样本适应

3. **理论分析**
   - [ ] 样本复杂度分析
   - [ ] 收敛性证明
   - [ ] 泛化界

---

## 📚 相关工作

### 神经架构搜索（NAS）
- **ENAS** (Pham et al., 2018): Efficient Neural Architecture Search
- **DARTS** (Liu et al., 2019): Differentiable Architecture Search
- **NAS-Bench** (Ying et al., 2019): Benchmarking NAS algorithms

### 多任务强化学习
- **MTRL** (Teh et al., 2017): Distral - Multi-task RL
- **Soft Modules** (Andreas et al., 2017): Modular Multitask RL
- **Meta-World** (Yu et al., 2020): Multi-task benchmark

### 模块化RL
- **Compositional RL** (Haarnoja et al., 2018)
- **Modular Networks** (Devin et al., 2017)
- **Option Discovery** (Bacon et al., 2017)

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---



## 🙏 致谢

- 感谢 **Claude Code** 在项目开发中的协助
- 感谢 **PyTorch** 团队提供优秀的深度学习框架
- 感谢开源社区的支持

---

**⭐ 如果觉得有用，请给个Star！**
