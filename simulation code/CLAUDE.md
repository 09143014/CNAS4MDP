# Project: Random Search Driven Composite Neural Architecture Search for Multi-Source RL State Encoding

## Project Overview

This project implements a simulation framework for studying compositional neural architecture search in reinforcement learning. The goal is to understand how modular architectures can exploit compositional structure in multi-task RL problems.

**Deliverable**: One or more Jupyter notebooks (`.ipynb` files) implementing the three phases below.

---

## Core Concepts

### State Blocks (b)
The state is decomposed into independent blocks: `s = (b1, b2, b3, b4)`

Each block is a latent variable with its own type:
- **b1**: Real number in `[-1, 1]`
- **b2**: Real number in `[0, 2π]` (periodic/angular)
- **b3**: Binary value in `{0, 1}`
- **b4**: 2D vector `(u, v)` in `[-1, 1]²`

### Modules (m)
A module is a learnable function that transforms one block into a feature representation:
- `h1 = m1(b1; θ1)` - module for block 1
- `h2 = m2(b2; θ2)` - module for block 2
- `h3 = m3(b3; θ3)` - module for block 3
- `h4 = m4(b4; θ4)` - module for block 4

**Module Types** (from simple to complex):

**A. Linear Module**
```
m1(b1; θ1) = w1 * b1 + c1
θ1 = (w1, c1)
```

**B. Polynomial Module**
```
m2(b2; θ2) = a0 + a1*b2 + a2*b2² + a3*b2³
θ2 = (a0, a1, a2, a3)
```

**C. MLP Module (1 hidden layer)**
```
m2(b2; θ2) = W2 * relu(U2 * b2 + c2) + d2
θ2 = (U2, c2, W2, d2)
```

**D. Fourier Features Module** (for periodic data)
```
m2(b2; θ2) = [sin(ω1*b2), cos(ω1*b2), sin(ω2*b2), cos(ω2*b2), ...]
θ2 = (ω1, ω2, ...)
```

### Parameters (θ)
All learnable weights and biases inside modules.

### Architecture
Defines which modules each task uses. Represented as a binary vector.
- Example: `[1, 1, 0, 0]` means Task 1 uses blocks b1 and b2 only.

---

## Workflow Example

### Step 0: Environment samples a state
```
b1 = 0.3
b2 = 1.2
b3 = 1
b4 = (0.5, -0.2)
s = (0.3, 1.2, 1, (0.5, -0.2))
```

### Step 1: Pick a task
Task 1 with ground truth structure: `S1 = {1, 2}` (depends on b1 and b2)

### Step 2: Compute true reward (environment)
```
g1(x) = x
g2(y) = sin(y)
r1(s) = g1(b1) + g2(b2) = 0.3 + sin(1.2)
```

### Step 3: Agent computes module outputs
```
h1 = m1(b1; θ1)
h2 = m2(b2; θ2)
h3 = m3(b3; θ3)
h4 = m4(b4; θ4)
```

### Step 4: Architecture selects modules
For Task 1 with architecture `z1 = {1, 2}`:
```
h_task1 = concat(h1, h2)
```
(Ignore h3 and h4)

### Step 5: Head predicts reward
```
r_hat1 = Head1(h_task1; φ1)
```
where φ1 are head parameters (e.g., linear layer)

### Step 6: Learning updates parameters
```
Loss = (r_hat1 - r1)²
```
Update:
- θ1, θ2, θ3, θ4 (module parameters)
- φ1 (head parameters)
- Later: search over z1 (architecture choice)

---

## Phase 1: Synthetic MDPs and Ground Truth

**Purpose**: Understand compositional structure before architecture search.

### Task 1.1 — Implement a Simple Synthetic MDP

**Requirements**:
- Define state as `s = (b1, b2, b3, b4)`
- Sample states independently (contextual bandit setting)
- Define multiple tasks, each with its own reward function
- Each task's reward depends only on a subset of blocks
- Rewards are deterministic (or have negligible noise)

**Example Task Structure**:
- **Task 1**: reward depends on b1 and b2
- **Task 2**: reward depends only on b3
- **Task 3**: reward depends on b1 and b4

**Deliverables**:
- [ ] Simple environment class
- [ ] Script that prints rewards for different tasks and states

---

### Task 1.2 — Ground-Truth Q-Functions and Structure

**Purpose**: Make "ground truth" precise by writing explicit Q-functions.

**Requirements**:

1. **Fix a simple setting**:
   - Use contextual bandit or one-step MDP
   - Action does not affect state (or fix single action)
   - Q-function equals expected reward

2. **Write analytical Q-function** for each task:
   - Express `Q_i(s) = Q_i(b1, b2, b3, b4)` explicitly
   - Simplify to sum over relevant blocks only

   Example:
   - Task 1: `Q_1(s) = g1(b1) + g2(b2)`
   - Task 2: `Q_2(s) = g3(b3)`

3. **Identify compositional structure**:
   - Which blocks appear in Q_i(s)
   - Which blocks do not appear
   - How the function decomposes (additive, separable)

4. **Explain implications for representation learning**:
   - Why optimal representation doesn't need irrelevant blocks
   - Why modular representation (one module per block) is sufficient
   - What "correct" architecture looks like for each task

5. **(Optional) Empirical verification**:
   - Sample random states
   - Compute rewards numerically
   - Fit regression model to verify only relevant blocks have non-zero coefficients

**Deliverables**:
- [ ] Write-up (1-2 pages) with:
  - Explicit Q-function for each task
  - Table mapping tasks to relevant blocks
  - Explanation of ground-truth structure
- [ ] (Optional) Notebook verifying decomposition numerically

**Goal**: Understand what blocks, tasks, and "correct structure" mean. No neural networks yet.

---

## Phase 2: Fixed Architectures Before Search

**Purpose**: See why architecture matters before doing any search.

### Task 2.1 — Monolithic Baseline

**Requirements**:
- Implement neural model that:
  - Takes full state (b1, b2, b3, b4)
  - Predicts reward/value for each task
- Train using supervised learning or simple bandit loop

**Deliverables**:
- [ ] Training curves
- [ ] Final performance per task

---

### Task 2.2 — Hand-Designed Composite Architecture

**Requirements**:
- Implement modular model:
  - One small network per block (m1, m2, m3, m4)
  - Manually choose which modules each task uses
  - Combine selected modules with simple linear head
- Use correct structure first

**Deliverables**:
- [ ] Performance comparison with monolithic baseline
- [ ] Explanation: why does modular structure help?

---

### Task 2.3 — Wrong Structure Ablation

**Requirements**:
- Force incorrect structure (e.g., Task 1 uses b3 instead of b2)
- Observe what breaks

**Deliverables**:
- [ ] Plot showing performance degradation
- [ ] Explanation of why structure matters

**Goal**: Understand that architectures encode assumptions, compositional structure is meaningful, and there is "right" vs "wrong" architecture. Still no NAS.

---

## Phase 3: Architecture Choice as a Search Problem

**Purpose**: Introduce architecture choice as structured black-box optimization.

### Task 3.1 — Define Architecture Search Space

**Requirements**:
- Fix modules m1–m4
- Define architecture by which blocks each task uses (binary vector)
  - Example: `[1, 1, 0, 0]` means use b1 and b2

**Deliverables**:
- [ ] Clear definition of what an "architecture" is
- [ ] Count of how many possible architectures exist

---

### Task 3.2 — Architecture Evaluation Function

**Requirements**:
- Given an architecture:
  - Train corresponding model with fixed training budget
  - Evaluate performance using expected reward or MSE to ground-truth Q
- Treat as deterministic/low-variance function: `performance = f(architecture)`

**Deliverables**:
- [ ] Code that evaluates any architecture under fixed protocol
- [ ] Empirical measurement of evaluation variance

---

### Task 3.3 — Simple Search Methods

**Requirements**:
- Implement simple search:
  - Random search over architectures
  - Exhaustive search if space is small enough
- Explore:
  - How many evaluations needed to find correct structure?
  - Which architectures perform similarly?

**Deliverables**:
- [ ] Plot: performance vs number of tried architectures
- [ ] Best architecture found vs ground truth

---

## Implementation Notes

### Suggested Reward Functions

For the synthetic MDP, use these ground truth functions:

**Task 1** (depends on b1, b2):
```
g1(x) = x                    # linear
g2(y) = sin(y)              # periodic
r1(s) = g1(b1) + g2(b2)
```

**Task 2** (depends on b3):
```
g3(z) = 2*z - 1             # maps {0,1} to {-1,1}
r2(s) = g3(b3)
```

**Task 3** (depends on b1, b4):
```
g1(x) = x                    # same as Task 1
g4(u,v) = sqrt(u² + v²)     # Euclidean norm
r3(s) = g1(b1) + g4(b4)
```

### Module Architecture Recommendations

- Start with simple MLPs (1-2 hidden layers)
- Hidden dimension: 16-32 units
- Use ReLU activation
- For b2 (periodic), consider Fourier features

### Training Protocol

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Training samples: 1000-5000
- Validation samples: 500-1000

---

## Success Criteria

By the end of this project, you should be able to:

1. ✅ Explain what compositional structure means in RL
2. ✅ Show empirically that modular architectures outperform monolithic ones
3. ✅ Demonstrate that wrong architectures perform poorly
4. ✅ Implement a simple architecture search algorithm
5. ✅ Find the correct architecture through search

---

## File Structure

Suggested organization:
```
simulation/
├── CLAUDE.md                          # This file
├── phase1_synthetic_mdp.ipynb         # Phase 1 implementation
├── phase2_fixed_architectures.ipynb   # Phase 2 implementation
├── phase3_architecture_search.ipynb   # Phase 3 implementation
└── utils.py                           # (Optional) Shared utilities
```
