# Phase 3 使用说明

## 🎯 Phase 3 是什么？

**架构搜索（Architecture Search）**：自动找到最优的模块化架构

### 核心流程

```
1. 定义搜索空间
   ↓
   有1000种可能的架构组合

2. 随机采样20个架构
   ↓
   对每个架构：
     - 用RL训练1000个episodes
     - 评估性能（MSE）

3. 找到最佳架构
   ↓
   比较所有架构的MSE，选择最小的
```

---

## 📊 架构搜索的详细过程

### 什么是"架构"？

架构定义了每个任务使用哪些状态块：

```python
# 例子1：Ground Truth架构（正确答案）
architecture = {
    'task1': [0, 1],  # Task 1 使用 b1 和 b2
    'task2': [2],     # Task 2 使用 b3
    'task3': [0, 3],  # Task 3 使用 b1 和 b4
}

# 例子2：随机架构
architecture = {
    'task1': [1, 3],  # Task 1 使用 b2 和 b4
    'task2': [0, 1],  # Task 2 使用 b1 和 b2
    'task3': [2],     # Task 3 使用 b3
}
```

### Random Search 流程

```python
for trial in range(20):  # 尝试20个架构
    # 1. 随机采样一个架构
    arch = sample_random_architecture()

    # 2. 用RL训练这个架构
    model = ModularQNetwork(arch)
    for episode in range(1000):  # 训练1000个episodes
        state = env.reset()
        reward = env.step(task)
        # ... RL训练 ...

    # 3. 评估性能
    mse = evaluate(model)

    # 4. 记录结果
    if mse < best_mse:
        best_arch = arch
        best_mse = mse
```

---

## 🚀 如何运行

### 方法1：在Jupyter中运行（推荐）

1. 打开notebook：
   ```bash
   jupyter notebook phase3_rl_training.ipynb
   ```

2. 按顺序运行单元格：
   - **单元格1-8**：定义所有函数和类
   - **单元格9**：测试Ground Truth架构（约2分钟）
   - **单元格10**：运行随机搜索（约40分钟，20个架构 × 2分钟）
   - **单元格11-12**：可视化和分析结果

### 方法2：快速测试（减少训练时间）

如果想快速测试，可以修改参数：

```python
# 在单元格10中修改：
random_results = random_search(
    arch_space, env,
    n_trials=5,  # 改为5个架构（原来20个）
    n_episodes_per_arch=500  # 改为500个episodes（原来1000个）
)
```

这样只需要约5分钟。

---

## 📈 进度条说明

### 你会看到的进度条

#### 1. RL训练进度条
```
RL Training: 100%|██████████| 1000/1000 [01:23<00:00, 12.05it/s, loss=0.0234]
```
- 显示当前训练的episode数
- 显示当前loss
- 预计剩余时间

#### 2. 架构搜索进度条
```
搜索进度: 45%|████▌     | 9/20 [18:23<20:15, 110.50s/it]
```
- 显示已评估的架构数
- 显示总架构数
- 预计剩余时间

#### 3. 找到更好架构的提示
```
  Trial 5: 找到更好的架构! MSE=0.012345
    task1:b1,b2 | task2:b3 | task3:b1,b4
```

---

## 📊 预期结果

### Ground Truth架构
```
Architecture: task1:b1,b2 | task2:b3 | task3:b1,b4
MSE: ~0.005
```

### 随机搜索结果
```
最佳架构: task1:b1,b2 | task2:b3 | task3:b1,b4
MSE: ~0.005-0.01
是否是Ground Truth: 可能是True
```

### 性能分布
- 最好的架构：MSE < 0.01
- 中等的架构：MSE ~ 0.1-0.3
- 最差的架构：MSE > 0.5

---

## 🔧 常见问题

### Q1: 为什么运行时间这么长？

**A**: 因为要训练多个架构：
- 每个架构需要训练1000个episodes（约2分钟）
- 随机搜索尝试20个架构
- 总时间：20 × 2分钟 = 40分钟

**解决方法**：
- 减少 `n_trials`（尝试更少的架构）
- 减少 `n_episodes_per_arch`（每个架构训练更少的episodes）

### Q2: 进度条卡住不动？

**A**: 可能是：
- 正在训练（等待一会儿）
- Jupyter kernel崩溃（重启kernel）

### Q3: 搜索没找到Ground Truth？

**A**: 这是正常的！原因：
- 随机搜索只尝试了20个架构（总共1000个）
- 训练episodes可能不够（增加到2000试试）
- RL训练有随机性

### Q4: 如何运行穷举搜索？

**A**: 添加一个新单元格：
```python
exhaustive_results = exhaustive_search(
    arch_space, env,
    n_episodes_per_arch=1000,
    max_architectures=50  # 只评估前50个架构
)
```

**注意**：穷举搜索会很慢（50个架构 × 2分钟 = 100分钟）

---

## 📝 代码结构

### 核心函数

| 函数 | 作用 | 运行时间 |
|------|------|---------|
| `train_with_rl()` | 用RL训练一个模型 | ~2分钟/1000 episodes |
| `evaluate_architecture_rl()` | 评估一个架构 | ~2分钟 |
| `random_search()` | 随机搜索最佳架构 | ~40分钟/20架构 |
| `exhaustive_search()` | 穷举搜索 | 很长时间 |

### 进度条位置

```python
# train_with_rl() 中
pbar = tqdm(range(n_episodes), desc="RL Training")

# random_search() 中
for trial in tqdm(range(n_trials), desc="搜索进度"):
    ...

# exhaustive_search() 中
for arch in tqdm(all_archs, desc="穷举搜索"):
    ...
```

---

## 🎯 学习要点

通过Phase 3，你应该理解：

1. **什么是架构搜索**
   - 在多个可能的架构中找到最优的
   - 类似于超参数搜索，但搜索的是模型结构

2. **如何评估架构**
   - 用RL训练模型
   - 在验证集上评估MSE
   - MSE越低越好

3. **搜索算法**
   - 随机搜索：简单但有效
   - 穷举搜索：保证找到最优，但很慢
   - 更高级的方法：贝叶斯优化、进化算法等

4. **RL在架构搜索中的作用**
   - 每个架构都需要训练才能评估
   - RL提供了一种训练方式
   - 架构搜索本身不是RL，而是黑盒优化

---

## 🚀 快速开始

```bash
# 1. 打开notebook
jupyter notebook phase3_rl_training.ipynb

# 2. 运行前8个单元格（定义函数）

# 3. 运行单元格9（测试Ground Truth）
#    预计时间：2分钟

# 4. 运行单元格10（随机搜索）
#    预计时间：40分钟
#    你会看到进度条！

# 5. 运行单元格11-12（可视化）
#    查看搜索结果
```

---

## 📚 相关文档

- `PROJECT_STRUCTURE.md`: 项目整体结构
- `RL_VS_SUPERVISED.md`: RL vs 监督学习
- `BUGFIX_LOG.md`: 问题修复记录

---

**现在你可以运行完整的架构搜索了！** 🎉
