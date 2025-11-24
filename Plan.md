# 项目技术规格书：VITA 框架实现方案

## 1. 核心目标
构建一个基于 **MAPPO** 的多智能体强化学习框架，通过引入 **自监督信任（Self-Supervised Trust）** 和 **变分信息瓶颈（Information Bottleneck）**，解决大规模动态环境下的协作鲁棒性问题。

**对比基线：** MAPPO (SOTA).

---

## 2. 系统架构总览

模型由四个核心子模块组成，数据流如下：

1.  **特征提取器 (Local Encoder):** 将原始观测映射为特征向量。
2.  **信任预测器 (Trust Predictor):** 自监督模块，预测邻居行为，生成信任分数。
3.  **变分通信层 (VIB-GAT):** 基于 GNN 和信息瓶颈，聚合邻居信息，自动过滤噪声。
4.  **残差决策层 (Residual Policy):** 融合自身特征与通信特征，输出动作。

---

## 3. 模块详细设计 (Implementation Details)

### 模块 A: 基础特征提取 (Local Encoder)
*   **输入:** 局部观测 $o_i$ (Shape: `[batch, n_agents, obs_dim]`)
*   **结构:**
    *   MLP (Layer 1) -> ReLU
    *   GRU (处理序列信息，捕捉时序特征)
*   **输出:** 自身隐状态 $h_i$ (Shape: `[batch, n_agents, hidden_dim]`)

### 模块 B: 自监督信任预测器 (Trust Predictor)
*   **原理:** 如果我能根据邻居的历史预测它的下一步动作，说明它行为可预测（可信）。
*   **输入:** 邻居 $j$ 的观测特征 $h_j$ (或观测历史)。
*   **结构:** 一个轻量级的 MLP 预测头。
*   **目标:** 预测邻居 $j$ 的真实动作 $a_j$。
*   **输出 & 计算逻辑:**
    1.  预测动作 $\hat{a}_j = \text{Predictor}(h_j)$
    2.  计算预测误差 $E_{ij} = ||\hat{a}_j - a_j||^2$ (MSE)
    3.  **信任分数 (Trust Score):**
        $$
        T_{ij} = \exp( - \gamma \cdot E_{ij} )
        $$
        (其中 $\gamma$ 是缩放系数，误差越大，信任越接近 0；误差为 0，信任为 1)。
    *   **代码关键:** `trust_mask` (Shape: `[batch, n_agents, k_neighbors, 1]`)

### 模块 C: 变分信息通信层 (VIB-GAT)
这是 A 类论文的核心创新点（信息瓶颈）。

*   **输入:** 自身特征 $h_i$，Top-K 邻居特征 $\{h_j\}_{j \in \mathcal{N}_i}$，以及信任分数 $T_{ij}$。
*   **结构 (变分编码):**
    *   对每个邻居的信息，不直接使用，而是先压缩。
    *   **Encoder:** $h_j \rightarrow \text{MLP} \rightarrow (\mu_{ij}, \sigma_{ij})$ (均值与方差)
    *   **Sampling (重参数化技巧):**
        $$
        Z_{ij} = \mu_{ij} + \epsilon \cdot \sigma_{ij}, \quad \text{where } \epsilon \sim \mathcal{N}(0,1)
        $$
*   **结构 (注意力聚合):**
    *   Query: $Q_i = W_q h_i$
    *   Key: $K_j = W_k Z_{ij}$
    *   Value: $V_j = W_v Z_{ij}$
    *   **Attention Score:**
        $$
        \alpha_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right)
        $$
    *   **Trust Gating (关键步骤):** 最终权重 $W_{ij} = \alpha_{ij} \cdot T_{ij}$ (注意力权重 $\times$ 信任分数)
*   **输出:** 通信特征 $h_{comm, i} = \sum_{j} W_{ij} V_j$

### 模块 D: 残差融合与决策 (Residual Fusion)
*   **输入:** 自身特征 $h_i$，通信特征 $h_{comm, i}$。
*   **结构:**
    *   门控网络: $g_i = \sigma(\text{Linear}([h_i, h_{comm, i}]))$
    *   融合公式:
        $$
        h_{final, i} = h_i + g_i \cdot h_{comm, i}
        $$
*   **输出:** $h_{final, i}$ 送入 Actor Head (输出动作概率) 和 Critic Head (输出价值 $V$)。

---

## 4. 损失函数设计 (Loss Functions)

总 Loss 由三部分组成，端到端优化：

$$
L_{Total} = L_{PPO} + \lambda_1 \cdot L_{Trust} + \lambda_2 \cdot L_{IB}
$$

1.  **PPO Loss ($L_{PPO}$):** 标准的强化学习损失（Clip loss + Entropy loss + Value loss），用于最大化游戏奖励。
2.  **信任预测 Loss ($L_{Trust}$):**
    $$
    L_{Trust} = \frac{1}{N \cdot K} \sum_{i,j} || \hat{a}_j - a_{j, \text{GT}} ||^2
    $$
    (监督信号：邻居的真实动作 `actions`)。
3.  **信息瓶颈 Loss ($L_{IB}$):**
    $$
    L_{IB} = D_{KL}( \mathcal{N}(\mu, \sigma) || \mathcal{N}(0, 1) )
    $$
    (KL 散度：强迫压缩后的信息分布接近标准正态分布，去除冗余信息，只保留最关键的特征)。

---

## 5. 训练流程 (Training Loop)

在每个训练 Step 中：

1.  **Rollout:** 智能体与环境交互，收集数据 `(obs, actions, rewards, ...)`。
2.  **Forward Pass (网络前向传播):**
    *   `Encoder` 提取特征。
    *   `Trust Predictor` 预测邻居动作，计算 $T_{ij}$。
    *   `VIB` 层计算 $\mu, \sigma$，并采样 $Z$。
    *   `GAT` 层利用 $T_{ij}$ 和 $Z$ 聚合出 $h_{comm}$。
    *   输出 `Action Logits` 和 `Value`。
3.  **Loss Calculation (计算损失):**
    *   计算 PPO 优势函数 (Advantage)。
    *   计算 $L_{PPO}$。
    *   使用存储的 `actions` 作为标签，计算 $L_{Trust}$。
    *   利用 $\mu, \sigma$ 计算 $L_{IB}$。
4.  **Backward Pass (反向传播):**
    *   `Optimizer.step()` 同时更新所有网络参数。

---

> 