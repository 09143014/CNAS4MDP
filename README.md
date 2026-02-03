Preliminary Readings
LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding (Poster) Section 2.1, 2.2, 3.1, Figures 3-4
- Understand key concepts like composite neural architecture search and RL state encoding
- No need to worry about the proposed LLM-based agent method and RL training details for now, but can think about what replaces them in our project

Reinforcement Learning by Richard Sutton Chapter 2-3
- Understand bandits and MDP

Project Goal

Understand how to design a principled composite neural architecture search method by first studying a simplified, synthetic MDP setting with stable, low-variance architecture evaluation. The long-term goal is to inform composite NAS for multi-source DeepRL, but this project focuses on structural identifiability and architecture choice rather than DeepRL optimization or exploration.

Phase 1: Synthetic MDPs and Ground Truth
Purpose
 Understand the problem setting clearly and build intuition for “compositional structure” before thinking about architecture search.
Task 1.1 — Implement a Simple Synthetic MDP (see Appendix for details)
- Define the state as
 s = (b1, b2, b3, b4)
- Each bi is a simple latent variable (e.g., real, binary).
- Sample states independently (contextual bandit is enough).
- Define several tasks i, each with its own reward function.
- Each task’s reward depends only on a subset of blocks.
- Rewards are deterministic functions of the state blocks (or have negligible noise).
Example:
- Task 1 reward depends on b1 and b2
- Task 2 reward depends only on b3
- Task 3 reward depends on b1 and b4
Deliverables
- A simple environment class
- A script that prints rewards for different tasks and states

---
Task 1.2 — Ground-Truth Q-Functions and Structure
Purpose
 Make the notion of “ground truth” precise by explicitly writing down the true value or Q-function for each task, and showing how it decomposes according to the block structure of the state.
 In this simplified setting, the Q-function is deterministic and fully determined by the reward function. This allows us to treat learning as supervised regression to the true Q-function, enabling precise and low-variance evaluation of architectural choices.
Task Description
Fix a simple setting
- Use a contextual bandit or a one-step MDP (no long horizons needed).
- Assume the action does not affect the state (or fix a single action).
- In this case, the optimal Q-function equals the expected reward.
2. Write down the analytical Q-function
 For each task i:
- Write Q_i(s) explicitly as a function of the state blocks:
 Q_i(s) = Q_i(b1, b2, b3, b4)
- Using the reward definition, simplify Q_i(s) into a sum over relevant blocks only.
Example:
- If Task 1 reward depends on b1 and b2:
 Q_1(s) = g1(b1) + g2(b2)
- If Task 2 reward depends only on b3:
 Q_2(s) = g3(b3)
3. Identify the compositional structure
 For each task, explicitly state:
- which blocks appear in Q_i(s),
- which blocks do not appear,
- how the function decomposes (additive, separable).
4. Explain the implication for representation learning
 In words, explain:
- why an optimal representation for Task i does not need information from irrelevant blocks,
- why a modular representation (one module per block) is sufficient,
- what a “correct” architecture should look like for each task.
5. (Optional) Empirical verification
- Sample random states.
- Compute rewards numerically.
- Fit a simple regression model to verify that:
  - only the relevant blocks have non-zero coefficients.
Deliverables
- A short write-up (1–2 pages) that includes:
  - the explicit Q-function for each task,
  - a table mapping tasks to relevant blocks,
  - a brief explanation of why this defines the ground-truth structure.
- (Optional) A small notebook verifying the decomposition numerically.
Goal of Phase 1
 The student should clearly understand:
- what blocks are,
- what tasks are,
- what it means for structure to be “correct”.
No neural networks yet.


---
Phase 2: Fixed Architectures Before Search
Purpose
 See why architecture matters at all, before doing any “search”.
Task 2.1 — Monolithic Baseline
- Implement a simple neural model that:
  - takes the full state (b1, b2, b3, b4),
  - predicts reward or value for each task.
- Train it using supervised learning or a simple bandit loop.
Deliverables
- Training curves
- Final performance per task

---
Task 2.2 — Hand-Designed Composite Architecture
- Implement a modular model:
  - one small network per block (m1, m2, m3, m4),
  - manually choose which modules each task uses,
  - combine selected modules with a simple linear head.
- Use the correct structure first.
Deliverables
- Performance comparison with the monolithic baseline
- Short explanation: why does modular structure help?

---
Task 2.3 — Wrong Structure Ablation
- Force an incorrect structure:
  - e.g., Task 1 uses b3 instead of b2.
- Observe what breaks.
Deliverables
- Plot showing performance degradation
- Short explanation of why structure matters
Goal of Phase 2
 The student should understand:
- architectures encode assumptions,
- compositional structure is meaningful,
- there is a “right” and “wrong” architecture in this synthetic world.
Still no NAS.


---
Phase 3: Architecture Choice as a Search Problem
Purpose
 Introduce architecture choice as a structured black-box optimization problem with a well-defined, low-noise objective, enabling principled comparison of different architecture search strategies.
Task 3.1 — Define a Simple Architecture Search Space
- Fix the modules m1–m4.
- Define an architecture by:
  - which blocks each task uses (a binary vector).
- For example, for Task 1:
  - [1, 1, 0, 0] means use b1 and b2.
Deliverables
- A clear definition of:
  - what an “architecture” is,
  - how many possible architectures exist.

---
Task 3.2 — Architecture Evaluation Function
- Given an architecture:
  - train the corresponding model with a fixed training budget,
  - evaluate its performance using:
    - expected reward, or
    - mean squared error to the ground-truth Q-function.
- Treat architecture performance as a deterministic or low-variance function:
 performance = f(architecture).
Deliverables
- Code that evaluates any given architecture under a fixed protocol
- Empirical measurement of evaluation variance

---
Task 3.3 — Simple Search Methods
 Start very simple:
- random search over architectures,
- exhaustive search if small enough.
Questions to explore:
- How many evaluations are needed to find the correct structure?
- Which architectures perform similarly?
Deliverables
- Plot: performance vs number of tried architectures
- Best architecture found vs ground truth

---
Appendix
| Original problem | This project |
|------------------|-------------|
| Source | Task |
| Source-specific encoder | Subset of neural modules |
| Fusion module | Learned module composition |
| Deep neural encoder | Modular function class |
| DeepRL training | Simple RL / bandit training |
                   ┌──────────────────────────────────────────────────┐
                   │                ENVIRONMENT (MDP)                 │
                   │          (Fixed structure, unknown to agent)     │
                   ├──────────────────────────────────────────────────┤
                   │ State:  s = (b¹, b², b³, b⁴)                     │
                   │                                                  │
                   │ Task-specific reward:                             │
                   │   rᵢ(s,a) = ∑_{j ∈ Sᵢ} gⱼ(bʲ)                     │
                   │   (each task i depends only on subset Sᵢ)        │
                   │                                                  │
                   │ Transition:  s' = T(s,a)                          │
                   │   (simple MDP or contextual bandit)               │
                   └──────────────────────────────────────────────────┘
                                        │
                                        ▼
                          (State sampled from the MDP)
                                        │
                                        ▼
     ┌──────────────────────────────────────────────────────────────────────────┐
     │                  COMPOSITE REPRESENTATION (Learnable)                   │
     │              (Modular function class / architecture space)               │
     ├──────────────────────────────────────────────────────────────────────────┤
     │  Neural modules (function approximators):                               │
     │     m₁(b¹) → h¹                                                         │
     │     m₂(b²) → h²                                                         │
     │     m₃(b³) → h³                                                         │
     │     m₄(b⁴) → h⁴                                                         │
     │                                                                          │
     │  Architecture variables (routing):                                      │
     │     zᵢ ⊂ {1,2,3,4}        # which modules task i uses                     │
     │     hᵢ = {hʲ : j ∈ zᵢ}                                                 │
     └──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────────────┐
                   │              TASK-SPECIFIC HEAD                  │
                   │        πᵢ(a | hᵢ)   or   Qᵢ(hᵢ, a)               │
                   │      (policy or value function)                  │
                   └──────────────────────────────────────────────────┘
                                        │
                                        ▼
                         Action a  (if applicable)
                                        │
                                        ▼
                     Reward rᵢ (and next state s')
                                        │
                                        ▼
     ┌──────────────────────────────────────────────────────────────────────────┐
     │               TRAINING / LEARNING LOOP (Tool)                            │
     ├──────────────────────────────────────────────────────────────────────────┤
     │  • Update task head parameters                                           │
     │  • Update module parameters θ₁…θ₄                                       │
     │  • Update routing / architecture parameters zᵢ                           │
     │                                                                          │
     │  (via contextual bandit, simple TD learning,                             │
     │   or PPO as an optional baseline)                                        │
     └──────────────────────────────────────────────────────────────────────────┘




──────────────────────────────────────────────────────────────────────────
                  GROUND-TRUTH SYNTHETIC MDP SETUP (CNAS4MDP)
──────────────────────────────────────────────────────────────────────────

ENVIRONMENT (MDP, Fixed, Unknown to Agent)
──────────────────────────────────────────────────────────────────────────
State:
    s = (b¹, b², b³, b⁴)          # independent latent blocks

Task-specific reward:
    rᵢ(s,a) = ∑_{j ∈ Sᵢ} gⱼ(bʲ)

Relevance sets (ground truth):
    S₁ = {1,2}
    S₂ = {3}
    S₃ = {1,4}

Examples of fixed environment mechanisms:
    g₁(x) = x
    g₂(y) = sin(y)
    g₃(z) = 2z − 1
    g₄(u,v) = u² + v

Transitions:
    s' = T(s,a)                  # simple (or bandit: no transition)


AGENT: COMPOSITE REPRESENTATION (Learnable)
──────────────────────────────────────────────────────────────────────────
Neural modules (function approximators):
    h¹ = m₁(b¹)     # linear feature module
    h² = m₂(b²)     # periodic feature module
    h³ = m₃(b³)     # logical / binary feature module
    h⁴ = m₄(b⁴)     # nonlinear feature module

Architecture variables (routing):
    zᵢ ⊂ {1,2,3,4}   # which modules task i uses

Ground-truth optimal routing:
    z₁* = {m₁, m₂}
    z₂* = {m₃}
    z₃* = {m₁, m₄}


TASK-SPECIFIC HEAD (Learnable)
──────────────────────────────────────────────────────────────────────────
For each task i:
    hᵢ = {hʲ : j ∈ zᵢ}
    πᵢ(a | hᵢ)      or      Qᵢ(hᵢ, a)


TRAINING / LEARNING LOOP (Tool, Not Focus)
──────────────────────────────────────────────────────────────────────────
Repeat:
1. Sample state s from MDP
2. Compute module outputs {h¹,h²,h³,h⁴}
3. Select modules via routing zᵢ
4. Compute reward rᵢ (and s' if MDP)
5. Update:
  - module parameters θ₁..θ₄
  - routing / architecture parameters zᵢ
  - task head parameters
    
Typical training choices:
    - contextual bandit
    - short-horizon MDP
    - supervised regression or contextual bandit updates


GOAL OF CNAS4MDP
──────────────────────────────────────────────────────────────────────────
• Recover ground-truth modular structure (blocks ↔ modules)
• Recover correct task–module wiring (zᵢ ≈ Sᵢ)
• Study identifiability and efficiency of composite architectures
• Isolate architecture discovery from DeepRL complexity
──────────────────────────────────────────────────────────────────────────

Notation Explanations
1. what b is
2. what θ is
3. what m is
4. how one full data point flows through the system

---
1. What is b?
In your synthetic world, the state is split into “blocks”:
s = (b1, b2, b3, b4)
Each bᵏ is just a part of the state — a latent variable (or a small vector) with its own meaning.
Examples of blocks b
- b1: a real number in [-1, 1]
- b2: a real number in [0, 2π]
- b3: a binary value in {0, 1}
- b4: a 2D vector (u, v) in [-1, 1]²
So:
- b = “a block of the state”
- the environment samples these blocks (or evolves them if you have transitions)

---
2. What is θ?
θ is just a standard symbol for learnable parameters.
When you see:
m(b; θ)
it means:
- m is a function (a model)
- it takes input b
- it has parameters θ that we learn from data
Example
If m is a tiny neural network, then θ includes:
- weight matrices
- bias vectors
If m is a polynomial model, θ includes:
- polynomial coefficients
So:
- θ = all trainable numbers inside the model

---
3. What is m?
m is a module: a learnable function that turns one block into a feature vector.
m1 takes b1 and outputs h1
 m2 takes b2 and outputs h2
 etc.
We write:
- h1 = m1(b1; θ1)
- h2 = m2(b2; θ2)
- h3 = m3(b3; θ3)
- h4 = m4(b4; θ4)
The output hᵏ is a “representation” (a feature embedding) for that block.
What kinds of functions can m be?
Here are very concrete choices (ranging from simplest to more neural):
Example A: Linear module
m1(b1; θ1) = w1 * b1 + c1
- θ1 = (w1, c1)
- output is a scalar or small vector
This is very simple, but may be too weak if g(b) is nonlinear.

---
Example B: Polynomial module (still simple)
m2(b2; θ2) = a0 + a1 b2 + a2 b2^2 + a3 b2^3
- θ2 = (a0, a1, a2, a3)
This can fit some nonlinear shapes, but the basis is fixed.

---
Example C: 1-hidden-layer MLP (typical “neural module”)
m2(b2; θ2) = W2 * relu(U2 * b2 + c2) + d2
- θ2 = (U2, c2, W2, d2)
Even though it’s small, this is a neural module:
- it learns a nonlinear representation.

---
Example D: Learned Fourier-features module (very relevant if b2 is periodic)
m2(b2; θ2) outputs:
 [sin(ω1 b2), cos(ω1 b2), sin(ω2 b2), cos(ω2 b2), ...]
 where ω1, ω2 are learnable parameters (part of θ2)
This is “neural” in the sense that the representation (frequencies) is learned.

---
4. Walk-through: one full workflow example
Let’s do a single step in a Level-2 setting (contextual bandit, deterministic reward).
Step 0: Environment samples a state
Suppose the environment samples:
b1 = 0.3
 b2 = 1.2
 b3 = 1
 b4 = (0.5, -0.2)
So the state is:
 s = (0.3, 1.2, 1, (0.5, -0.2))

---
Step 1: Pick a task
Say we are in Task 1, and the ground truth says:
 S1 = {1, 2}
Meaning: reward depends only on b1 and b2.

---
Step 2: Compute the true reward (environment side)
Environment has fixed functions g1, g2, g3, g4.
For example:
 g1(x) = x
 g2(y) = sin(y)
Then for Task 1:
 r1(s) = g1(b1) + g2(b2)
 = 0.3 + sin(1.2)
This reward is deterministic.

---
Step 3: Agent computes module outputs (agent side)
Agent computes features from each block:
h1 = m1(b1; θ1)
 h2 = m2(b2; θ2)
 h3 = m3(b3; θ3)
 h4 = m4(b4; θ4)
These are learned features.

---
Step 4: “Architecture” selects which modules to use
For Task 1, the architecture might choose:
 z1 = {1, 2}
So the representation used for Task 1 is:
 h_task1 = concat(h1, h2)
(We ignore h3 and h4.)

---
Step 5: The head predicts reward or Q
Head is a simple function like linear regression:
r_hat1 = Head1(h_task1; φ1)
where φ1 are head parameters.

---
Step 6: Learning updates θ and φ
We compute a loss, e.g. squared error:
Loss = (r_hat1 - r1)^2
Then we update:
- θ1..θ4 (module parameters)
- φ1 (head parameters)
- and later, we will search over z1 (architecture choice)
That’s the whole loop.

---
Key takeaway sentence
- b = parts of the state (blocks)
- θ = learnable parameters inside modules
- m = learnable module functions mapping blocks to features
- architecture = which m’s each task uses and how they’re composed

---
Clarifying the Role of Neural Architecture Search (NAS) in This Project
- In this project, the architecture is defined by the task-specific routing variables
- z = (z₁, z₂, z₃, …) where each zi specifies which neural modules mj are used by task i.
- Neural architecture search (NAS) refers to the process of selecting these routing variables z, i.e., deciding which modules are composed for each task. Different choices of z correspond to different hypothesis classes, since they determine which learned nonlinear functions are available to each task.
- For a fixed architecture z:
  - Module parameters θ1,…,θ4 and task-specific head parameters ϕi are learned using standard optimization (e.g., supervised regression or contextual bandit updates).
  - The resulting trained model is evaluated using a stable, low-variance metric such as mean squared error to the ground-truth Q-function or expected reward.
  - This defines a scalar performance value f(z), measuring the quality of architecture z.
- NAS methods (e.g., random search, exhaustive search, Bayesian optimization, or LLM-based agents) operate in an outer loop that:
  - proposes candidate architectures z,
  - observes their evaluated performance f(z),
  - and uses this information to guide the proposal of future architectures.
- Importantly, NAS methods do not directly learn the model parameters θ or ϕ. Instead, they treat the learning procedure as a black-box architecture evaluation function and focus solely on searching over the discrete, structured space of routing variables z.
- This separation between:
  - inner-loop learning (optimizing θ,ϕ for a fixed z), and
  - outer-loop architecture search (optimizing z based on f(z)),
 allows NAS to be studied in a principled manner, isolated from DeepRL-specific challenges such as exploration and high-variance returns.

Complete role break-down:
Environment
   ↓
Data (b, r)
   ↓
Inner loop (learning)
- Fix architecture z = {zᵢ}
- Learn θ (module parameters)
- Learn φ (head parameters)
- Measure performance f(z)
↓
Outer loop (NAS)
- BO / LLM / EA proposes next z

Routing as an abstraction of NAS
What routing is in our notation
We've defined
- Modules: m1,m2,m3,m4
- For each task i:
zi⊂{1,2,3,4}
Here:
- zi is the routing decision
- It specifies the path that information takes through the model
So routing = choosing the subset zi.

Think of the modules as components and the architecture as a wiring diagram.
- The full set of modules exists
- Routing decides:
  - which modules are active,
  - which ones are bypassed,
  - for which task
This is exactly analogous to:
- routing packets in a network,
- routing signals in a circuit,
- routing inputs to experts in a mixture-of-experts model.

Concrete example
Suppose:
- Task 1: z1={1,2}
- Task 2: z2={3}
Then:
- For Task 1:
  - Input b1 → m1
  - Input b2 → m2
  - Ignore m3,m4
- For Task 2:
  - Input b3 → m3
  - Ignore everything else
That selective activation is routing.

What routing is not
Routing is not:
- changing the internal computation of a module,
- learning module parameters θ,
- learning the task head parameters ϕ,
- dynamically changing per data point (in your base setup).
Routing is:
- a structural choice,
- fixed during evaluation of an architecture,
- changed only by the NAS outer loop.

Why use the term "routing"
Routing has three crucial properties:
1. Discrete and structural
  - Good for NAS / BO / search
2. Changes the hypothesis class
  - Different routes ⇒ different functions representable
3. Interpretable
  - You can say exactly which modules a task uses
This is why routing is the architecture variable in our project.

Where Bayesian Optimization Fits
Role of Bayesian Optimization (BO) in the simplified problem
- In this project, Bayesian optimization treats architecture search as a black-box optimization problem over a discrete, structured space of routing variables.
- An architecture is defined by
- z={zi}, zi⊂{1,2,3,4},
- where each zi specifies which neural modules are used by task i.
Concretely:
- BO proposes a candidate architecture z
 (e.g., Task 1 uses modules {1,2}, Task 2 uses {3}).
- The proposed architecture z is evaluated by:
  - fixing the routing variables z,
  - training the module parameters θ and task-specific head parameters ϕ,
  - computing a performance metric f(z) (e.g., MSE to the ground-truth Q-function or expected reward).
- BO updates its surrogate model using the observed pair (z,f(z)) and proposes the next architecture to evaluate.
- In this setup, BO acts solely as an architecture proposer; it does not directly update θ or ϕ.
Why BO is especially appropriate here
- In the Level-2 setting:
  - f(z) is deterministic or low-variance,
  - architecture evaluations are directly comparable,
  - the architecture space defined by routing variables is small and structured.
- As a result, BO provides a principled and sample-efficient method for searching over architectures rather than a heuristic one.

---
Where an LLM Agent Would Fit
An LLM-based agent plays the same functional role as BO, but uses a different mechanism for proposing architectures.
Role of an LLM agent
- Instead of maintaining an explicit surrogate model (e.g., a Gaussian process), the LLM:
  - observes a history of previously evaluated architectures z,
  - observes their corresponding performance values f(z),
  - reasons over this history to propose a new candidate architecture z.
- The proposed architecture is then evaluated using the same inner-loop procedure:
  - fix z,
  - train θ and ϕ,
  - measure performance f(z).
Functionally:
- LLM agent = architecture proposer
- It replaces the acquisition function used in BO, while leaving the architecture evaluation and learning process unchanged.

What is f(z)?
f(z) is computed offline, after training, and returned to the NAS outer loop.
For a fixed architecture z:
Fix routing zi for all tasks
1. Train:
  - module parameters θ1,…,θ4
  - head parameters ϕi
2. Use a fixed training protocol and budget
Only after this we evaluate z.
[图片]
f(z) measures how well the architecture allows the model to approximate the true Q-functions.
In our Level-2 setting:
- rewards are deterministic
- Q-functions are known
- state distribution is fixed
- training budget is fixed
Therefore:
f(z) is deterministic or very low-variance.

Concrete toy example
Suppose:
- 3 tasks
- true routing:
  - z1∗={1,2}
  - z2∗={3}
  - z3∗={1,4}
Now try two architectures:
Architecture A (correct)
- z=z∗
- Model fits Q-functions almost perfectly
- MSE ≈ 0.01
- f(z)=−0.01
Architecture B (wrong)
- Task 1 uses {1,3} instead of {1,2}
- Model cannot represent g2(b2)
- MSE ≈ 1.2
- f(z)=−1.2
NAS prefers Architecture A.
That’s all f(z) is doing.