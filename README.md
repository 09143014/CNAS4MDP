# Compositional Neural Architecture Search for Multi-Task RL State Encoding

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Goal**: Automatically discover optimal modular architectures for multi-task state encoding in reinforcement learning.

---

## 📖 Project Overview

### Core Problem

In multi-task reinforcement learning, different tasks may rely on different subsets of the state space. For example:
- **Task A** may only need position information
- **Task B** may only need velocity information
- **Task C** may need position + angle information

**Key Challenge**: How to automatically discover which state features each task should use?

### Our Approach

This project implements a **Modular Neural Architecture Search (NAS) framework** that:

1. Decomposes the state space into independent **State Blocks**
2. Learns a dedicated **Encoder Module** for each state block
3. Uses **Architecture Search** to automatically discover the optimal module combination for each task
4. Trains and evaluates candidate architectures using **Reinforcement Learning**

### Current Version

⚠️ **Note**: This repository implements a **simplified version** to validate the core idea:
- Uses **Contextual Bandit** (single-step MDP) instead of a full MDP
- Uses **synthetic data** instead of real environments
- Uses **random search** instead of advanced NAS algorithms

**Ultimate Goal**: Extend to real multi-task RL environments (e.g., Meta-World, DeepMind Control Suite)

---

## 🎯 Core Concepts

### 1. State Blocks

The state is decomposed into independent blocks: `s = (b1, b2, b3, b4)`

| Block | Type   | Range     | Description             |
|-------|--------|-----------|-------------------------|
| b1    | Scalar | [-1, 1]   | Linear feature          |
| b2    | Scalar | [0, 2π]   | Periodic feature (angle)|
| b3    | Binary | {0, 1}    | Discrete feature        |
| b4    | 2D Vec | [-1, 1]²  | Vector feature          |

### 2. Encoder Modules

Each state block has its own independent neural network encoder:

```python
h1 = m1(b1; θ1)  # Encoder for Block 1
h2 = m2(b2; θ2)  # Encoder for Block 2
h3 = m3(b3; θ3)  # Encoder for Block 3
h4 = m4(b4; θ4)  # Encoder for Block 4
```

### 3. Architecture

Defines which modules each task uses:

```python
# Example architecture
architecture = {
    'task1': [0, 1],  # Task 1 uses b1 and b2
    'task2': [2],     # Task 2 uses b3
    'task3': [0, 3],  # Task 3 uses b1 and b4
}
```

### 4. Architecture Search

Find the optimal architecture among all possible ones:

```
Search space size = (C(4,1) + C(4,2))^3 = 10^3 = 1000 architectures
```

---

## 🚀 Quick Start

### Environment Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rl-architecture-search.git
cd rl-architecture-search
```

#### 2. Create a Virtual Environment (Recommended)
```bash
# Using conda
conda create -n rl-nas python=3.8
conda activate rl-nas

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

#### 3. Install Dependencies
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

#### 4. Launch Jupyter
```bash
jupyter notebook
```

---

## 📁 Project Structure

```
simulation/
├── Big_README.md                      # This file (project overview)
├── README.md                          # Brief description
├── CLAUDE.md                          # Project requirements specification
├── requirements.txt                   # Python dependencies
│
├── 📓 Core Experiment Notebooks
│   ├── phase1_synthetic_mdp.ipynb         # Phase 1: Synthetic MDP and Ground Truth
│   ├── phase2_fixed_architectures.ipynb   # Phase 2: Fixed Architecture Comparison
│   └── phase3_rl_training.ipynb           # Phase 3: Architecture Search
│
├── 📄 Documentation
│   ├── PROJECT_STRUCTURE.md           # Detailed project structure
│   ├── RL_VS_SUPERVISED.md            # RL vs Supervised Learning comparison
│   ├── RL_VERSION_SUMMARY.md          # RL version usage guide
│   ├── PHASE3_GUIDE.md                # Phase 3 detailed guide
│   ├── BUGFIX_LOG.md                  # Bug fix log
│   └── FILE_LIST.txt                  # File list
│
└── 📊 Experiment Results
    ├── phase1_reward_decomposition.png
    ├── phase2_training_curves.png
    ├── phase2_performance_comparison.png
    ├── phase3_rl_training_curves.png
    ├── phase3_search_progress.png
    └── phase3_performance_distribution.png
```

---

## 🔬 Experiment Description

### Phase 1: Synthetic MDP and Ground Truth

**Goal**: Understand the compositional structure of the problem

**File**: `phase1_synthetic_mdp.ipynb`

**Contents**:
1. Define the synthetic MDP environment
2. Implement ground truth reward functions for 3 tasks:
   - **Task 1**: `r1(s) = b1 + sin(b2)` (depends on b1, b2)
   - **Task 2**: `r2(s) = 2*b3 - 1` (depends on b3)
   - **Task 3**: `r3(s) = b1 + ||b4||` (depends on b1, b4)
3. Visualize reward decomposition

**Runtime**: ~1 minute

**Key Outputs**:
- Ground Truth architecture definition
- Reward function visualization

---

### Phase 2: Fixed Architecture Comparison

**Goal**: Validate the importance of architecture selection

**File**: `phase2_fixed_architectures.ipynb`

**Contents**:
1. **Task 2.1 - Monolithic Baseline**
   - All tasks use all state blocks
   - No modular structure

2. **Task 2.2 - Correct Modular Architecture**
   - Uses the Ground Truth architecture
   - Each task uses only its relevant state blocks

3. **Task 2.3 - Wrong Architecture**
   - Intentionally uses incorrect state block combinations
   - Observes performance degradation

**Training**: Reinforcement Learning (Experience Replay + TD Learning)

**Runtime**: ~5 minutes

**Key Findings**:
- ✅ Correct architecture significantly outperforms the monolithic baseline
- ✅ Wrong architecture shows clear performance degradation
- ✅ Architecture selection matters more than training time

---

### Phase 3: Architecture Search

**Goal**: Automatically discover the optimal architecture

**File**: `phase3_rl_training.ipynb`

**Contents**:

#### 3.1 Architecture Search Space
- Define all possible architectures (1000 total)
- Each task can use 1–2 state blocks

#### 3.2 Architecture Evaluation Function
```python
def evaluate_architecture_rl(architecture, env, n_episodes=1000):
    # 1. Create a new model
    model = ModularQNetwork(architecture)

    # 2. Train with RL
    train_with_rl(model, env, n_episodes=1000)

    # 3. Evaluate performance (MSE)
    return mse
```

#### 3.3 Search Algorithms

**Random Search**:
```python
random_results = random_search(
    arch_space, env,
    n_trials=100,              # Try 100 architectures
    n_episodes_per_arch=1000   # Train each for 1000 episodes
)
```

**Exhaustive Search** (optional):
```python
exhaustive_results = exhaustive_search(
    arch_space, env,
    n_episodes_per_arch=1000,
    max_architectures=50       # Evaluate top 50 architectures
)
```

**Runtimes**:
- Random search (20 architectures): ~40 minutes
- Random search (100 architectures): ~3.5 hours
- Exhaustive search (1000 architectures): ~35 hours

**Key Findings**:
- ✅ Random search finds near-optimal architecture at 10% sampling rate
- ✅ Best architecture MSE=0.053 vs Ground Truth MSE=0.045
- ✅ Only one block selection error in task3 (b3 vs b4)

---

## 📊 Experiment Results

### Phase 2 Results

| Architecture Type  | MSE       | Description                     |
|--------------------|-----------|---------------------------------|
| Monolithic Baseline| 0.15–0.25 | Uses all state blocks           |
| Correct Architecture| 0.04–0.06 | Ground Truth                   |
| Wrong Architecture | 0.30–0.50 | Incorrect state block combination|

### Phase 3 Results (100 Random Search Trials)

```
Best Architecture: task1:b1,b2 | task2:b3 | task3:b1,b3
Best MSE: 0.053227

Ground Truth: task1:b1,b2 | task2:b3 | task3:b1,b4
GT MSE: 0.044782

Gap: 0.008 (19%)
```

**Top 5 Architectures**:
1. `task1:b1,b2 | task2:b3 | task3:b1,b3` - MSE=0.053 ⭐
2. `task1:b1,b2 | task2:b3,b4 | task3:b2,b4` - MSE=0.115
3. `task1:b2,b3 | task2:b3 | task3:b1,b4` - MSE=0.149
4. `task1:b1,b2 | task2:b3,b4 | task3:b3` - MSE=0.152
5. `task1:b2,b3 | task2:b3,b4 | task3:b1` - MSE=0.171

### Visualization

<img src="simulation code/phase3_search_results.png" width="800">

---

## 🔧 Technical Details

### RL Training Configuration

```python
# Environment
- Type: Contextual Bandit (single-step MDP)
- State sampling: each block sampled independently
- Reward: deterministic (no noise)

# Model
- Architecture: Modular Q-Network
- Modules: MLP (input → 16 → 8)
- Head: Linear (concat(modules) → 1)

# Training
- Algorithm: TD Learning + Experience Replay
- Optimizer: Adam (lr=1e-3)
- Replay Buffer: 10000
- Batch Size: 64
- Episodes: 1000 per architecture
```

### Architecture Search Configuration

```python
# Search Space
- Number of state blocks: 4
- Max blocks per task: 2
- Total architectures: 1000

# Search Algorithm
- Method: Random Search
- Samples: 20–100
- Evaluation metric: MSE (predicted Q vs true reward)
```

---

## 📈 Future Work

### Short-Term Goals

1. **Advanced Search Algorithms**
   - [ ] Bayesian Optimization
   - [ ] Evolutionary Algorithms
   - [ ] RL-based NAS

2. **More Complex MDPs**
   - [ ] Multi-step MDP (non-Bandit)
   - [ ] State transition dependencies
   - [ ] Discount factor γ < 1

3. **Larger Search Spaces**
   - [ ] More state blocks (8–16)
   - [ ] Variable module sizes
   - [ ] Hierarchical architectures

### Long-Term Goals

1. **Real RL Environments**
   - [ ] Meta-World (robotic manipulation)
   - [ ] DeepMind Control Suite
   - [ ] Atari games

2. **Transfer Learning**
   - [ ] Cross-task module sharing
   - [ ] Zero-shot generalization
   - [ ] Few-shot adaptation

3. **Theoretical Analysis**
   - [ ] Sample complexity analysis
   - [ ] Convergence proofs
   - [ ] Generalization bounds

---

## 📚 Related Work

### Neural Architecture Search (NAS)
- **ENAS** (Pham et al., 2018): Efficient Neural Architecture Search
- **DARTS** (Liu et al., 2019): Differentiable Architecture Search
- **NAS-Bench** (Ying et al., 2019): Benchmarking NAS algorithms

### Multi-Task Reinforcement Learning
- **MTRL** (Teh et al., 2017): Distral - Multi-task RL
- **Soft Modules** (Andreas et al., 2017): Modular Multitask RL
- **Meta-World** (Yu et al., 2020): Multi-task benchmark

### Modular RL
- **Compositional RL** (Haarnoja et al., 2018)
- **Modular Networks** (Devin et al., 2017)
- **Option Discovery** (Bacon et al., 2017)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgements

- Thanks to **Claude Code** for assistance during project development
- Thanks to the **PyTorch** team for the excellent deep learning framework
- Thanks to the open-source community for their support

---

**⭐ If you find this useful, please give it a star!**
