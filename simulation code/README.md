# Random Search Driven Composite Neural Architecture Search for Multi-Source RL State Encoding

## Project Overview

This project implements a complete simulation framework for studying compositional neural architecture search in reinforcement learning. The goal is to understand how modular architectures can exploit compositional structure in multi-task RL problems.

## Files

- **CLAUDE.md** - Detailed project specification and requirements
- **phase1_synthetic_mdp.ipynb** - Phase 1: Synthetic MDPs and Ground Truth
- **phase2_fixed_architectures.ipynb** - Phase 2: Fixed Architectures Before Search
- **phase3_architecture_search.ipynb** - Phase 3: Architecture Search
- **README.md** - This file

## Quick Start

### Prerequisites

```bash
pip install numpy matplotlib pandas torch scikit-learn seaborn tqdm
```

### Running the Notebooks

Execute the notebooks in order:

1. **Phase 1**: `phase1_synthetic_mdp.ipynb`
   - Implements synthetic MDP environment
   - Defines ground truth Q-functions
   - Verifies compositional structure

2. **Phase 2**: `phase2_fixed_architectures.ipynb`
   - Compares monolithic vs modular architectures
   - Shows why architecture matters
   - Tests wrong structure ablation

3. **Phase 3**: `phase3_architecture_search.ipynb`
   - Defines architecture search space
   - Implements evaluation function
   - Runs random and exhaustive search

## Project Structure

### Phase 1: Synthetic MDPs and Ground Truth

**Goal**: Understand compositional structure before architecture search.

**Key Concepts**:
- State blocks: `s = (b1, b2, b3, b4)`
- Tasks with different reward dependencies
- Ground truth Q-functions

**Deliverables**:
- ✅ Environment class
- ✅ Ground truth Q-functions
- ✅ Compositional structure table
- ✅ Empirical verification

### Phase 2: Fixed Architectures Before Search

**Goal**: Show why architecture matters.

**Experiments**:
1. Monolithic baseline (uses all blocks)
2. Modular architecture with correct structure
3. Modular architecture with wrong structure

**Deliverables**:
- ✅ Training curves
- ✅ Performance comparison
- ✅ Analysis of why structure matters

### Phase 3: Architecture Choice as a Search Problem

**Goal**: Automatically discover correct architecture.

**Methods**:
1. Random search
2. Exhaustive search

**Deliverables**:
- ✅ Architecture search space definition
- ✅ Architecture evaluation function
- ✅ Search algorithms
- ✅ Performance analysis

## Key Results

### Ground Truth Architecture

- **Task 1**: Uses blocks b1 and b2 (linear + periodic)
- **Task 2**: Uses block b3 only (binary)
- **Task 3**: Uses blocks b1 and b4 (linear + norm)

### Performance Comparison

| Model Type | Task 1 MSE | Task 2 MSE | Task 3 MSE |
|------------|------------|------------|------------|
| Monolithic | ~0.01-0.1 | ~0.01-0.1 | ~0.01-0.1 |
| Modular (Correct) | **~0.001** | **~0.001** | **~0.001** |
| Modular (Wrong) | ~0.1-1.0 | ~0.1-1.0 | ~0.1-1.0 |

*Note: Exact values depend on training and random seed*

### Search Results

- Random search finds good architectures within 20-50 evaluations
- Correct architecture consistently achieves lowest MSE
- Architecture choice has 10-100x impact on performance

## Understanding the Code

### State Representation

```python
state = {
    'b1': scalar in [-1, 1],           # linear block
    'b2': scalar in [0, 2π],           # periodic block
    'b3': binary in {0, 1},            # discrete block
    'b4': vector in [-1, 1]²           # 2D block
}
```

### Module Architecture

```python
# Each block has its own module
m1(b1) -> h1  # module for b1
m2(b2) -> h2  # module for b2
m3(b3) -> h3  # module for b3
m4(b4) -> h4  # module for b4

# Task selects relevant modules
task1: concat(h1, h2) -> reward_prediction
task2: h3 -> reward_prediction
task3: concat(h1, h4) -> reward_prediction
```

### Architecture Representation

```python
architecture = {
    'task1': [0, 1],  # uses blocks 0 (b1) and 1 (b2)
    'task2': [2],     # uses block 2 (b3)
    'task3': [0, 3],  # uses blocks 0 (b1) and 3 (b4)
}
```

## Extending the Project

### Ideas for Further Exploration

1. **More complex reward functions**
   - Non-additive dependencies
   - Interaction terms between blocks

2. **Larger state spaces**
   - More blocks (8-16)
   - Higher dimensional blocks

3. **More tasks**
   - 5-10 tasks with overlapping dependencies

4. **Advanced search methods**
   - Bayesian optimization
   - Evolutionary algorithms
   - Gradient-based NAS

5. **Real RL environments**
   - Temporal dynamics
   - Partial observability
   - Continuous control

## Troubleshooting

### Common Issues

**Issue**: Training loss not decreasing
- **Solution**: Increase learning rate or number of epochs

**Issue**: High variance in evaluation
- **Solution**: Increase training samples or use multiple seeds

**Issue**: Search not finding correct architecture
- **Solution**: Increase number of evaluations or reduce search space

## References

This project is inspired by research in:
- Neural Architecture Search (NAS)
- Multi-task Reinforcement Learning
- Compositional Representations
- Modular Neural Networks

## License

This is an educational project for research purposes.

## Contact

For questions or issues, please refer to the CLAUDE.md file for detailed specifications.
