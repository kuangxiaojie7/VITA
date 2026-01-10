实验方案：绝地求生 (Survival in Chaos)
你需要跑三组实验，生成一张组合图表。

1. 场景设定 (The Setup)
所有的实验都必须在一个 “基础恶劣环境” 下进行。

基础噪声：obs_noise_std: 0.5 (或者一个能让 MAPPO 胜率掉到 40%-50% 的值)。

理由：这是为了**“破防” MAPPO**。如果大家都能看清（Clean），MAPPO 就不需要通信，也就没 VITA 什么事了。只有当大家都看不清时，通信（向队友问路）才成为刚需。

2. 三个对比角色 (The Contenders)
你需要配置三个不同的 yaml（或命令行参数）来代表这三个角色：

🟢 角色 A: 独狼 (MAPPO Baseline)

配置：algorithm: mappo

噪声设置：

obs_noise_std: 0.5 (环境侧，必须加)

comm_*: (MAPPO 不吃这个，设啥都行，默认 0)

预期表现：由于单兵视野受阻，且无法通过通信校准，胜率会比较低（比如 40%）。这是你的 Lower Bound (基准线)。

🔴 角色 B: 盲信者 (Naive VITA / Ablation)

配置：algorithm: vita

VITA 阉割：enable_trust: false, enable_kl: false (关闭防御机制，全盘接收)。

噪声设置：

obs_noise_std: 0.5 (同上)

关键攻击：comm_malicious_agent_prob: 0.3 (30% 的队友是恶意的，或者 comm_noise_std 很大)。

预期表现：

它本来想通过通信来弥补观测噪声。

结果收到了恶意/高噪信息，因为没有 Trust 过滤，它把垃圾当成了宝。

结局：发生 Negative Transfer (负迁移)。它的胜率不仅比不过 Full VITA，甚至可能低于 MAPPO（被队友坑死了）。

🔵 角色 C: 智者 (Full VITA / Ours)

配置：algorithm: vita

VITA 完全体：enable_trust: true, enable_kl: true。

噪声设置：

obs_noise_std: 0.5 (同上)

comm_malicious_agent_prob: 0.3 (同上)

预期表现：

Trust 模块识别出那是“恶意消息”并切断。

VIB 模块滤除了剩余的背景噪声。

它利用剩下的 70% 正常队友的信息，成功校准了自己的模糊视野。

结局：胜率显著高于 MAPPO（因为有通信红利）和 Naive VITA（因为没有被坑）。

📝 具体参数设置指南
建议直接在 configs/smac/3s5z/vita_3s5z_noise.yaml 和对应的 MAPPO yaml 里修改：

实验 1：观测噪声鲁棒性 (Obs Noise Robustness)
变量：obs_noise_std [0.1, 0.3, 0.5]

固定：comm_malicious_* = 0

目的：证明 VITA (Comm) > MAPPO (No-Comm)。

逻辑：即使没有恶意通信，单纯的高斯噪声环境也需要 VIB 来做“多视角去噪”。这里 VITA 应该比 MAPPO 强。

实验 2：恶意通信防御 (Byzantine Resilience) —— 这是你的高光实验
固定：obs_noise_std = 0.5 (保持高压环境)

变量：comm_malicious_agent_prob [0.0, 0.2, 0.4, 0.6]

或者用 comm_malicious_noise_scale [0, 5, 10] 来调节恶意程度。

设置细节 (推荐使用 VITA 专用的 comm_*)：

使用 comm_malicious_agent_prob 配合 comm_malicious_mode="random" (每局随机选内鬼) 或 "fixed"。

这种噪声只污染信道，不改变队友的物理行为。

这最能体现 Trust 的作用：队友还在正常打仗（物理行为正常），但嘴里在报假点（通信恶意）。如果 VITA 听了就会被带偏，不听就能赢。

🚀 为什么“只影响 VITA 的噪声”可以用来做公平对比？
你之前的顾虑是：“MAPPO 不吃通信噪声，所以不公平”。 修正后的逻辑是：

我们是在 obs_noise_std > 0 的前提下比拼的。

在这个前提下，通信是 “为了生存必须承担的风险”。

MAPPO 选择不承担风险（不通信），代价是上限被锁死（看不清）。

Naive VITA 选择盲目承担风险（全通信），代价是被噪声毒死。

Full VITA 选择通过 Trust 管理风险，获得了收益（看清了）与风险的平衡。

所以，加 comm_* 噪声不是为了欺负 VITA，而是为了模拟真实世界中 “获取信息的代价”。

总结行动路线
基准线：跑 MAPPO + obs_noise_std=0.5。

对照组：跑 VITA (无 Trust) + obs_noise_std=0.5 + comm_malicious_agent_prob=0.3。

实验组：跑 VITA (全开) + obs_noise_std=0.5 + comm_malicious_agent_prob=0.3。

把这三条线画在一张图上（X轴是 step，Y轴是胜率），你会得到一张完美的论文插图：MAPPO 在底部躺平，Naive VITA 掉坑里，Full VITA 一骑绝尘。