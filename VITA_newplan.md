# VITA: 基于信息瓶颈与信任机制的高效多智能体通信框架
# (Efficient Multi-Agent Communication via Information Bottleneck and Trust Mechanism)

## 1. 核心叙事转型 (The Narrative Pivot)

**从“防御攻击 (Robustness)” ➡️ 转型为 ➡️ “通信效率 (Efficiency/Sparsity)”**

* **旧痛点**：通信信道中存在恶意攻击或噪声，现有算法很脆弱。
    * *难点*：难以构建公平的对比基准（MAPPO 本身不通信，不受攻击影响）。
* **新痛点**：现有的通信算法（如 GAT, TarMAC）存在严重的**“信息冗余” (Information Redundancy)**。智能体倾向于“全广播、全接收”，导致带宽浪费，且难以从嘈杂的队友信息中提取关键战术意图。
* **VITA 的解决方案**：
    * 利用 **VIB** 对发送的信息进行**压缩**（去除观测中的冗余背景）。
    * 利用 **Trust** 对接收的信息进行**门控**（只关注与当前决策高度相关的队友）。
    * **核心卖点**：VITA 实现了**“少即是多” (Less is More)** —— 用更少的有效连接数，达到了 SOTA 的协作性能。

---

## 2. 实验设置 (Experimental Setup)

**不再需要手动添加噪声环境！直接在 Clean 环境下决胜负。**

### 2.1 地图选择
* **首选**：`3s5z` (StarCraft II)
* **理由**：该地图兵种异构（Stalker/Zealot），战术分工明确，且存在局部视野遮挡，非常需要高效通信，是验证协作的绝佳场景。

### 2.2 关键参数配置 (Config)
请在 `src/config/algs/vita.yaml` 或 `default.yaml` 中确保以下设置：
* `t_max`: **5,000,000** (5M 步，大图收敛慢)
* `hidden_dim`: **128** (增加网络容量)
* `n_heads`: **4** (多头注意力捕捉不同战术意图)
* `vib_beta`: **0.001 ~ 0.01** (控制信息压缩率，越大压缩越狠)

### 2.3 对比基准 (Baselines) —— “三国杀”格局

| 算法名称 | 组件配置 | 角色定位 | 预期表现 |
| :--- | :--- | :--- | :--- |
| **MAPPO (No Comm)** | 无通信 | **基准线 (Baseline)** | 胜率及格，但缺乏精细配合。 |
| **MAPPO-Comm (GAT)** | 仅 GAT (无 VIB/Trust) | **反面教材 (Inefficient)** | 胜率可能最高，但通信开销巨大，关注了所有队友（全连接）。 |
| **VITA (Ours)** | **GAT + VIB + Trust** | **高效主角 (Efficient)** | **胜率与 GAT 持平或更高，但“有效连接数”显著降低。** |

---

## 3. 代码微调建议 (Action Items)

虽然核心架构不需要动，但建议微调 Loss 函数以强化“稀疏性”效果。

### 3.1 增加稀疏性正则化 (Sparsity Regularization)
在计算总 Loss 的地方（`src/learners/vita_learner.py` 或类似文件），加入 L1 正则化，强迫 Trust 权重趋向于 0：

```python
# 假设 trust_weights 是 Trust 模块输出的注意力权重 [batch, n_agents, n_neighbors]
# 我们希望它越稀疏越好（只保留关键连接）

sparsity_loss = torch.mean(torch.abs(trust_weights)) 
sparsity_coef = 0.01  # 系数不要太大，避免影响主任务

# 总 Loss
loss = ppo_loss + value_loss + vib_loss + trust_loss + (sparsity_coef * sparsity_loss)