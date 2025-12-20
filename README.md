# VITA：变分信息论可信多智能体框架

本仓库实现了 VITA（Variational Information-Theoretic Trustworthy Agents）以及 MAPPO 基线，可在 SMAC（StarCraft Multi-Agent Challenge）中检验多智能体鲁棒性。核心特性：

- 统一训练入口 `src/main.py`，通过 YAML 配置切换 MAPPO / VITA。
- `algorithm: mappo` 默认走 `external/on-policy` 的官方 RMAPPo 实现（用于论文 baseline）。
- `algorithm: vita` 默认走 `external/on-policy` 的 `rvita`（基于官方 RMAPPo runner/buffer）。
- `src/vita/` 下的模块化实现严格对应 Plan.md：`components/feature_encoder.py`（局部编码）、`trust_predictor.py`（自监督信任）、`vib_gat.py`（信息瓶颈 + 注意力）和 `residual_policy.py`（门控残差），最终由 `agent.py` 组装。
- VITA（`rvita`）在官方 PPO 损失上额外加入 KL/信任监督；信任与通信都支持“延迟+分段 warmup”调度。
- SMAC 封装支持高斯噪声、丢包、恶意队友以及按视距设置的通信掩码 (`comm_sight_range`)，方便构造压力测试。
- 自带日志、配置加载器与 `tools/plot_training.py` 可视化脚本。

## 环境准备

1. 安装依赖（示例使用 Conda）：
   ```bash
   conda create -n vita python=3.10 -y
   conda activate vita
   pip install -r requirements.txt
   ```
2. 按 SMAC 官方说明安装 StarCraft II 客户端与地图资源（默认路径 `~/.sc2`，或设置 `SC2PATH` 指向解压目录）。
3. 确认服务器上的两张 4090D GPU 可用：
   ```bash
   nvidia-smi
   ```

## 在同一服务器使用两张 4090D

默认单进程单卡。通过设置 `CUDA_VISIBLE_DEVICES` 可以在两个终端分别启动 MAPPO 与 VITA；如需单命令驱动两卡，可使用 `torchrun --nproc_per_node=2`。

### Windows PowerShell（单进程单卡）
```powershell
# GPU0：MAPPO 基线
$env:CUDA_VISIBLE_DEVICES="0"
python -m src.main --config configs/smac/3s5z/mappo_3s5z_clean.yaml --log-dir runs/gpu0

# GPU1：VITA 噪声实验
$env:CUDA_VISIBLE_DEVICES="1"
python -m src.main --config configs/smac/3s5z/vita_3s5z_noise.yaml --log-dir runs/gpu1
```

### Linux / WSL（单进程单卡）
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main --config configs/smac/3s5z/mappo_3s5z_clean.yaml --log-dir runs/gpu0
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/smac/3s5z/vita_3s5z_noise.yaml --log-dir runs/gpu1
```

### torchrun 并行示例
```bash
torchrun --nproc_per_node=2 --master_port=29501 \
    src/main.py --config configs/smac/3s5z/mappo_3s5z_clean.yaml --log-dir runs/mappo_dist

torchrun --nproc_per_node=2 --master_port=29502 \
    src/main.py --config configs/smac/3s5z/vita_3s5z_noise.yaml --log-dir runs/vita_dist
```
> 每个进程都会自行加载配置，建议在不同终端中指定 `--config` 与 `--log-dir`，即可在同一服务器上并行训练 MAPPO 与 VITA。

## 运行示例

1. MAPPO 基线（无噪声）：
   ```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main --config configs/smac/3s5z/mappo_3s5z_clean.yaml --log-dir runs/mappo_clean
   ```
2. VITA 噪声/恶意场景：
   ```bash
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/smac/3s5z/vita_3s5z_noise.yaml --log-dir runs/vita_noise
   ```

## 结果可视化

训练日志为 JSON 行格式，位于 `runs/<algo>/train.log`。新版脚本支持在同一图中对比多个算法的同一指标，例如：
```bash
python tools/plot_training.py \
    --log-files runs/mappo_dist/mappo/train.log runs/vita_dist/vita/train.log \
    --labels MAPPO VITA \
    --metrics episode_reward,policy_loss,value_loss,kl,trust_loss,eval_win_rate \
    --smooth 20 \
    --output-dir figures/compare \
    --winrate-style --timesteps-per-update 960
```
将分别生成 `figures/compare/episode_reward.png` 等文件；开启 `--winrate-style` 后 `eval_win_rate` 会以 “T (mil) vs Test Win Rate%” 的格式呈现，其中 `960` 对应 `episode_length × num_envs`，用于把 update 映射到总环境步数。若省略 `--output-dir`，则在屏幕上逐个弹出。

训练默认每 20 个 update 触发一次评估循环，使用当前策略在无探索噪声的环境里跑 32 局并记录 `eval_episode_reward`、`eval_win_rate`。这些字段在日志里和训练指标放在一起，可以直接用上面的脚本绘制更贴近 PyMARL 基准的胜率曲线。

## 配置说明

- `configs/smac/3s5z/mappo_3s5z_clean.yaml`：SMAC 3s5z 标准场景（官方 RMAPPo baseline）。
- `configs/smac/3s5z/vita_3s5z_noise.yaml`：同一地图但附加高斯噪声、丢包和恶意队友；`history_length` 用于构造观测序列，`comm_sight_range` + `max_neighbors` 控制通信范围与 Top-K 邻居。
- 训练配置新增 `eval_interval_updates`、`eval_episodes` 控制评估循环（默认每 20 次 update 暂停训练、贪心评估 32 局并记录 `eval_win_rate` / `eval_episode_reward`）。

可直接修改 YAML 中的噪声、`max_neighbors`、`history_length` 等字段，快速生成新的干扰组合。

## MAPPO vs. VITA

- **架构**：VITA 在局部编码后串联自监督信任预测、变分信息瓶颈 GAT 与门控残差，并结合视距掩码 + `trust_threshold` 做通信裁剪；MAPPO 仅使用常规集中式 actor-critic。
- **损失/调度**：VITA 的 PPO 目标额外包含 KL 与信任监督，可分别通过 `enable_kl`/`enable_trust` 与延迟-warmup 调度控制生效时机，对噪声/恶意代理更鲁棒。
- **数据处理**：VITA 的 rollout buffer 记录邻居观测序列、邻居动作序列以及 Top-K 邻居索引，供信任模块学习；MAPPO 只消费单步观测。

## 目录结构

```
configs/
  smac/
    mappo_3s5z.yaml
    vita_3s5z_noise.yaml
src/
  algorithms/
    onpolicy_baseline.py
  utils/
    config_loader.py
    seeding.py
  vita/
    components/
      feature_encoder.py
      trust_predictor.py
      vib_gat.py
      residual_policy.py
    agent.py
  main.py
external/
  on-policy/        # 官方 MAPPO/RMAPPO 基线 + rvita 扩展
tools/
  plot_training.py
Plan.md
requirements.txt
```

该结构便于扩展到 SMAC v2 或新的噪声类型，只需增加 YAML、组件或训练脚本即可。
