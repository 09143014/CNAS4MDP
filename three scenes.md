# 场景 1：多机器人协作 / 编队控制（Multi-Robot Coordination）

> **关键词**：多源状态、结构先验极强、传统 NAS 非常难、LLM 优势明显

---

## 1.1 场景设定（可直接写进 paper）

**任务**
多个机器人（UAV / AGV / mobile robots）在共享环境中完成协作任务，如：

* 编队飞行
* 协同导航避障
* 多机器人搬运 / 覆盖

每个智能体执行 **decentralized policy**，但 state encoding 包含 **自身 + 邻居 + 全局信息**。

---

## 1.2 多源状态拆解（这是关键）

| Source          | 数学形式        | 语义       | 合适 encoder        |
| --------------- | ----------- | -------- | ----------------- |
| 自身状态 (x^{self}) | 连续向量        | 位置、速度、电量 | FFN               |
| 邻居状态 (x^{nbr})  | 图结构         | 相对位置、速度  | GNN               |
| 环境感知 (x^{env})  | Image / BEV | 障碍物、地图   | CNN / ViT         |
| 通信历史 (x^{comm}) | 序列          | 消息、意图    | Transformer / GRU |

整体 state encoder：
[
s = g_\phi\big(
f^{self}*{\theta_1}(x^{self}),
f^{nbr}*{\theta_2}(x^{nbr}),
f^{env}*{\theta_3}(x^{env}),
f^{comm}*{\theta_4}(x^{comm})
\big)
]

👉 **这就是 LACER 的 canonical composite form**

---

## 1.3 NAS 搜索空间设计（可直接仿 Table 1）

### 模块级搜索空间示例

**Self encoder（FFN）**

* depth ∈ {1,2,3}
* hidden dim ∈ {64,128,256}
* activation ∈ {relu, gelu}

**Neighbor encoder（GNN）**

* GNN type ∈ {GCN, GAT}
* layers ∈ {1,2,3}
* heads ∈ {2,4,8}（for GAT）

**Environment encoder（CNN / ViT）**

* backbone ∈ {ResNet-like, shallow ViT}
* channels ∈ {32,64,128}
* depth ∈ {2,3,4}

**Communication encoder（Transformer）**

* layers ∈ {1,2,3}
* heads ∈ {2,4,8}
* embedding dim ∈ {64,128}

**Fusion module**

* merge ∈ {concat, gated, cross-attention}
* fusion depth ∈ {1,2}
* hidden dim ∈ {128,256}

👉 搜索空间规模 **指数级**，传统 NAS 很容易崩。

---

## 1.4 RL 目标与反馈信号（完全对齐 LACER）

### Task metric（用于 NAS 评价）

* 编队保持误差 ↓
* 协作成功率 ↑
* 碰撞率 ↓

### RL reward（训练时）

* 稳定性奖励
* 通信代价惩罚

### Representation side info（给 LLM）

* 邻居 encoder 输出之间的冗余度
* 邻居 → fusion 的 mutual information
* self / env 特征占比

👉 **LLM 非常擅长判断“是不是 neighbor encoder 太弱 / 太强”**

---

## 1.5 为什么 LLM 比 NAS 强（论文级解释）

* GNN / CNN / Transformer 的 **组合结构经验性极强**
* 搜索空间是 **组合 + 结构约束**，不是纯数值
* LLM 能利用：

  * “多机器人 → 图结构”
  * “通信 → attention”
  * “env → spatial encoding”

这正好满足你理论中：
[
p_\varepsilon^{LLM} \gg q_\varepsilon^{baseline}
]

---

# 场景 2：Embodied AI（视觉 + 语言 + 机器人控制）

> **关键词**：VLA、RoboTwin、ManiSkill、LLM 天生懂结构

---

## 2.1 场景设定

**任务**

* 操作任务（抓取、放置、组合）
* 目标由语言指定
* 状态同时包含 **视觉、语言、本体**

---

## 2.2 多源状态结构（标准 VLA）

| Source      | 数据          | Encoder     |
| ----------- | ----------- | ----------- |
| RGB / RGB-D | image       | CNN / ViT   |
| 语言指令        | text        | Transformer |
| 机器人状态       | vector      | FFN         |
| 触觉 / 力      | time-series | TCN / GRU   |

[
s = g_\phi(
f^{vision}(x^{img}),
f^{lang}(x^{text}),
f^{proprio}(x^{state}),
f^{tactile}(x^{force})
)
]

---

## 2.3 NAS 搜索空间（和 ManiSkill 非常像）

**Vision encoder**

* CNN vs ViT
* depth ∈ {2,3,4}
* patch size / kernel size

**Language encoder**

* frozen LM vs lightweight Transformer
* layers ∈ {1,2,3}
* dim ∈ {128,256}

**Proprio encoder**

* FFN depth ∈ {1,2,3}
* normalization ∈ {LN, none}

**Fusion**

* early fusion vs late fusion
* cross-attention heads
* gating vs concat

---

## 2.4 RL 目标与反馈

### Task metric

* success rate
* completion time

### RL reward

* dense shaping + sparse success

### Representation feedback（很重要）

* vision ↔ language MI
* tactile ↔ action MI
* fusion redundancy

👉 **这是 GENIUS 做不了的，而 LACER 能做的**

---

## 2.5 为什么这个场景特别适合写论文

* Reviewer 直觉：

  > “当然 encoder 结构很重要”
* LLM 的 **语言-视觉结构先验** 非常强
* 不和 “LLM 直接当 policy” 冲突（安全）

---

# 场景 3：智能电网 / 能源管理（Traffic 的“孪生场景”）

> **关键词**：时间序列 + 全局状态 + 决策稳定性

---

## 3.1 场景设定

**任务**

* 调度发电 / 储能
* 平衡供需
* 降低成本与峰值风险

---

## 3.2 多源状态设计（极其自然）

| Source  | 数据            | Encoder         |
| ------- | ------------- | --------------- |
| 历史负载    | time-series   | Transformer     |
| 设备状态    | vector        | FFN             |
| 天气预测    | time-series   | Transformer     |
| 电价 / 规则 | scalar / text | Embedding + FFN |

---

## 3.3 Encoder 搜索空间

**Load encoder**

* Transformer layers
* attention heads

**Weather encoder**

* shallow Transformer / TCN

**Device encoder**

* FFN depth / dim

**Fusion**

* gated fusion（非常重要）
* residual vs non-residual

---

## 3.4 RL 目标

### Task metric

* 总成本
* 峰值惩罚

### RL reward

* 稳态奖励
* 波动惩罚

### Representation signals

* load ↔ fusion MI
* redundancy between load & weather

---

## 3.5 为什么 reviewer 会接受

* 和 traffic 几乎同构
* 但 domain 不同，说明 **方法 general**
* 很好放在 **Future Work / Extension**

