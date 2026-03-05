# 训练步骤详解

以 `block_mask = [0, 1]`（b1+b2）架构，batch_size=2 为例，逐步说明。

---

## 网络结构（先明确参数）

```
state [5-dim]: [b1, b2, b3, b4x, b4y]

BlockModule_b1: Linear(1→16) → ReLU → Linear(16→8)
BlockModule_b2: Linear(1→16) → ReLU → Linear(16→8)
head:           Linear(16→1)       ← concat(8+8=16 dim)
```

---

## Step 1: `model.train()`

**作用：** 把模型切换到「训练模式」

这个模型里没有 Dropout / BatchNorm，所以实际上没有数值变化，但它会把一个内部标志位设为 True：

```python
model.training = True   # 内部状态切换
```

对比：
- `model.train()`  → `model.training = True`  → Dropout 生效、BN 用批统计
- `model.eval()`   → `model.training = False` → Dropout 关闭、BN 用历史均值

**本例影响：** 无直接数值影响，但是语义上告诉框架「现在在训练」。

---

## Step 2: `optimizer.zero_grad()`

**作用：** 把所有参数的梯度清零

PyTorch 的梯度是**累积的**，如果不清零，上一个 batch 的梯度会叠加进来。

```
假设上一次 backward 后：
  head.weight.grad = [0.3, -0.1, ...]   ← 上一个 batch 留下的

zero_grad() 后：
  head.weight.grad = [0.0,  0.0, ...]   ← 清空，准备接收新梯度
```

---

## Step 3: `pred_qs = model(batch_states)`（前向传播）

**作用：** 输入一批 state，算出预测 Q 值

假设 batch = 2 条样本：
```
batch_states = [
  [-0.25,  5.97,  0.0,  0.56,  0.19],   # sample 1
  [ 0.80,  1.23,  1.0, -0.33,  0.72],   # sample 2
]
```

前向计算流程：

```
Step 3.1  提取 block 输入
  b1_input = [[-0.25], [5.97]]         b2_input = [[0.80] ,[1.23]]
                                    

Step 3.2  各 BlockModule 独立计算（假设简化权重）
  b1_feat = BlockModule_b1(b1_input)  →  shape [2, 8]
  b2_feat = BlockModule_b2(b2_input)  →  shape [2, 8]

Step 3.3  拼接
  h = concat([b1_feat, b2_feat], dim=1)  →  shape [2, 16]

Step 3.4  线性头输出
  pred_qs = head(h)  →  shape [2]
           = [0.42, 0.78]    ← 预测出的 Q 值
```

---

## Step 4: `loss.backward()`（反向传播）

**作用：** 根据 loss 对所有参数求偏导数（梯度）

先算 loss：
```
true rewards  = [-0.56,  1.93]   ← 从 buffer 里采出来的真实 reward
pred_qs       = [ 0.42,  0.78]   ← Step 3 的输出

MSELoss = mean[(pred - true)²]
        = mean[(0.42-(-0.56))², (0.78-1.93)²]
        = mean[0.96,  1.32]
        = 1.14
```

然后 `loss.backward()` 用**链式法则**从 loss 一路往回求导：

```
∂loss/∂pred_qs  →  ∂loss/∂head.weight  →  ∂loss/∂b2_feat  →  ∂loss/∂BlockModule_b2.weight  →  ...
```

执行后每个参数都有了梯度：
```
head.weight.grad     = [...某个向量...]
BlockModule_b1 各层.grad = [...]
BlockModule_b2 各层.grad = [...]
```

**注意：** 梯度只是「应该往哪个方向更新多少」的信息，参数值本身**还没变**。

---

## Step 5: `optimizer.step()`（参数更新，Adam）

**作用：** 用梯度 + Adam 算法真正修改参数值

Adam 不是简单的 `w = w - lr * grad`，它维护两个动量（m, v）：

```
# 对每个参数 w：

m = β1 * m + (1 - β1) * grad          # 一阶动量（梯度的指数移动平均）
v = β2 * v + (1 - β2) * grad²         # 二阶动量（梯度平方的指数移动平均）

m_hat = m / (1 - β1^t)                # 偏差修正
v_hat = v / (1 - β2^t)

w = w - lr * m_hat / (√v_hat + ε)     # 参数更新
```

用本例的 `head.weight` 举例：
```
更新前: head.weight = [0.12, -0.05, 0.33, ...]

  grad  = [0.08, -0.02, 0.11, ...]   ← backward 算出来的
  m     = [0.00,  0.00, 0.00, ...] → 更新后 ≈ [0.008, -0.002, 0.011, ...]
  v     = [0.00,  0.00, 0.00, ...] → 更新后 ≈ [6.4e-5, 4e-6, 1.2e-4, ...]

  Δw ≈ -0.001 * (0.008 / √6.4e-5)
     ≈ -0.001 * 1.0
     ≈ -0.001

更新后: head.weight = [0.119, -0.049, 0.329, ...]   ← 微小变化
```

---

## 五步总结

```
model.train()          → 切换训练模式（标志位）
optimizer.zero_grad()  → 清空上一轮残留梯度
pred_qs = model(...)   → 前向传播，算预测值        【参数只读】
loss.backward()        → 反向传播，算每个参数的梯度 【参数只读，梯度写入】
optimizer.step()       → Adam 更新参数             【参数修改】
```

只有最后一步 `optimizer.step()` 才真正改变了网络权重。
