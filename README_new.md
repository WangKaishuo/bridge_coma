# Bridge-COMA: Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination

**MSc Research Project — Kaishuo Wang, 2026**

---

## 1. 研究问题

在合作-对抗混合多智能体环境中，agent 必须同时做到两件事：与队友有效传递信息，以及最小化对对手的信息泄露。这两个目标之间存在内在张力——在桥牌叫牌中，每一个叫品既是给 partner 的信号，也是对 opponent 的暴露。

**核心假设**：通过信息论 reward shaping，显式奖励"对 partner 有信息量的叫品"并惩罚"对 opponent 的信息泄露"，可以引导 agent 发展出更高效的叫牌协议，提升双桌 IMP 表现。

$$r_{\text{info}} = \underbrace{I(\text{bid};\,\text{hand} \mid \text{partner})}_{\text{partner推断增益}} - \beta \cdot \underbrace{I(\text{bid};\,\text{hand} \mid \text{opponent})}_{\text{opponent信息泄露}}$$

其中 $\beta \geq 0$ 控制通信清晰度与信息隐蔽性之间的权衡。

**三个研究问题**：

1. **有效性**：dual-information bonus 是否在 IMP 上显著优于 vanilla MAPPO？
2. **机制验证**：partner 与 opponent 的推断精度差异是否与叫牌效率相关？
3. **消融分析**：partner term 与 opponent penalty（β）各自贡献多少？

---

## 2. 实验平台：桥牌叫牌

桥牌叫牌是研究此问题的理想平台：

- **2v2 合作-对抗结构**：NS 合作，EW 为对手
- **受限通信信道**：仅 38 种叫品（Pass / X / XX / 1C–7NT）
- **信息不对称**：每人只能看到自己的手牌
- **全披露约束**（Alert Rule）：约定必须公开，信息压缩不能靠"暗语"
- **标准客观评估**：双桌 IMP 消除牌力运气

**奖励**：IMP regret = 实际 IMP − DDS 最优 IMP（≤ 0）。训练时叠加 $r_{\text{info}}$，评估时仅用 IMP。

---

## 3. 网络架构

### 3.1 Actor / Critic（MLP，对齐 Kita et al. 2024）

**输入**：301 维固定向量

```
vulnerability             :   4 维
当前玩家手牌 (one-hot)     :  52 维
每个 bid 谁叫 (35 × 4)   : 140 维   — 4维 one-hot，表示 N/E/S/W 谁叫了该 bid
每个 bid 的加倍状态 (35×3): 105 维   — 3维 one-hot，未加倍/加倍/再加倍
─────────────────────────────────
合计                      : 301 维
```

**设计说明**：

- 展开历史替代 LSTM 序列，每个 bid 的"谁叫"信息天然编码了叫牌顺序（位置蕴含轮次），无需时序建模
- 加倍状态用 35×3 而非 2 bit，因为每个实质叫品都可独立被加倍/再加倍，历史加倍信息传递防守实力
- 与 LSTM 相比：训练更快，批量 rollout 无需 padding，与 OpenSpiel 数据格式天然对齐

**Actor 网络**：4 层 MLP，每层 1024 神经元，ReLU，输出 38 维 logits + 合法动作掩码

**Critic 网络（CTDE）**：同 Actor 结构，额外接收 AllHandsEncoder（4×52 → 256），训练时可见所有手牌

**HAPPO 双独立架构**：`actor_n` / `actor_s` / `critic_n` / `critic_s` 四个完全独立网络，S-phase 只更新 actor_s + critic_s，N-phase 只更新 actor_n + critic_n，消除交叉污染。

### 3.2 Belief Network

**任务**：给定观察者手牌 + 叫牌历史 + 观察者位置 + 目标位置，预测目标玩家的手牌语义特征。

**输入**：

```
observer_hand   : (batch, 52)         — 观察者手牌 one-hot
history         : (batch, seq, 38)    — 叫牌历史序列（LSTM 处理）
observer_pos    : (batch,)            — Embedding(4, 32)
target_pos      : (batch,)            — Embedding(4, 32)
```

**输出**：(batch, 48) logits，未经 Sigmoid

```
[0 :16]  AKQJ 归属     — 16 维独立 binary（每门花色 × A/K/Q/J）
[16:48]  套长 one-hot  — 32 维（每门花色 × 8 档：0,1,2,3,4,5,6,7+）
```

**损失**：BCEWithLogitsLoss(pos_weight=3.0)，修正 75% 零类不平衡

**评估指标**：
- `honor_acc`：AKQJ 归属准确率（threshold=0.5）
- `length_acc`：套长 argmax 准确率（等价于 4 类分类准确率）
- `overall_acc`：全部 48 维准确率（随机基线 ≈ 0.25，目标 ≥ 0.40）

**设计说明**：Belief Network 保留 LSTM，因为推断任务中叫牌顺序本身有语义（先叫 1♥ 后叫 2♠ 与先叫 1♠ 后叫 2♥ 含义不同），LSTM 的序列建模在此有不可替代的价值。

### 3.3 r_info 计算

$$I(\text{bid};\,\text{hand} \mid \text{obs}) \approx \text{CE}(q_\phi(h_{t-1}),\, \text{hand}) - \text{CE}(q_\phi(h_t),\, \text{hand})$$

- 叫牌前后分别查询 Belief Network，计算交叉熵之差
- ReLU 截断（$\max(0, \cdot)$）：互信息 ≥ 0，负值仅反映 Belief Net 估计滞后
- β 在 reward 聚合时一次性应用：`r_final = normalize(IMP + β × (I_partner − I_opponent))`

---

## 4. 训练流程

### 4.1 SL 预训练（所有 agent 共享，一次性）

**数据**：OpenSpiel SAYC 数据集（WBridge5 生成，SAYC 系统）

- 训练集：1M 局，12.8M state-action 对
- 格式：每条 trajectory 是动作序列，前 52 步为发牌，之后为叫牌
- 数据地址：`https://console.cloud.google.com/storage/browser/openspiel-data/bridge`

**说明**：SL 预训练给 agent 一个合理的叫牌起点，将实验关注点聚焦在 r_info 机制本身，而非从零学习基础叫牌协议。这是标准做法（Kita et al. 2024, Lockhart et al. 2020 均采用）。

| 参数 | 值 |
|------|-----|
| epochs | 30（早停：acc ≥ 0.90 × patience=3） |
| lr | 1e-4 |
| batch size | 256 |
| 优化器 | Adam |

### 4.2 FSP + RL 精调

**FSP（Fictitious Self-Play）**：从历史 checkpoint pool 中随机采样对手，防止 policy cycling（Kita et al. 2024 消融实验证明其必要性）。Pool size = 10。

**训练结构**：IBR 交替训练

```
Round k:
  S-phase:  训练 actor_s + critic_s（actor_n 冻结）
  N-phase:  训练 actor_n + critic_n（actor_s 冻结）
            Agent B 在此阶段激活 r_info + JIT Belief Burn-in
```

**JIT Belief Burn-in**：每次 N-phase 开始前，用 1000 局 rollout 对 Belief Net 做 3 epoch 快速微调，同步到当前叫牌协议，避免 N 策略更新后 Belief Net OOD。

**KL Anchor**：将当前策略锚定到 SL checkpoint，防止 RL 破坏已学习的基础协议

```
loss += kl_lambda × KL(π_SL ∥ π_current)
kl_lambda: 0.5 → 0.1（跨轮线性退火）
```

| RL 超参数 | 值 | 来源 |
|----------|-----|------|
| lr | 1e-6 | Kita et al. 2024 |
| entropy coef | 1e-3 | Kita et al. 2024 |
| clip ratio | 0.2 | 标准 PPO |
| PPO epochs/update | 4 | — |
| batch size | 256 | — |
| γ | 0.99 | — |
| GAE λ | 0.95 | — |
| FSP pool size | 10 | — |
| β (Agent B) | 0.05 | — |

**DDS 数据**：200 万副完整叫牌手牌，预计算 DDS 结果存 npz，RL 训练时循环采样。

---

## 5. 实验设计

### 5.1 阶段结构

```
Phase 0: 工程改造（无计算消耗）
  ├── policy_net.py: MLP PolicyNetwork（301 维输入，4×1024）
  ├── openspiel_loader.py: SAYC 数据 → BC 格式转换
  ├── subgame_trainer.py: 批量并行 rollout（batch=32 环境池）
  └── FSP pool 功能确认

Phase 1: Competitive 子游戏排雷（2 Pro 账号，约 2 小时）
  ├── 目的: 验证 ir 信号、entropy、vl 健康，排查 bug
  ├── 前缀: 1H - 1S（对手全程参与，β term 激活）
  └── 判断标准: ir > 0，entropy 不坍塌，KL 可控

Phase 2: SL 预训练（Pro+ 账号，约 5 小时）
  └── 所有 agent 共享同一 SL checkpoint

Phase 3: 主实验（Pro+ 账号，每 agent 单次 ≤ 8 小时）
  ├── Agent A:    MAPPO, β=0               （控制组）
  ├── Agent B0:   MAPPO + r_info, β=0      （消融：仅 partner term）
  ├── Agent B05:  MAPPO + r_info, β=0.05   （主实验配置）
  ├── Agent B2:   MAPPO + r_info, β=0.2    （消融：高 β）
  └── IPPO baseline                         （独立学习对比）

Phase 4: 多 seed 验证（seed ∈ {42, 123, 456}，Agent A + B05）
  └── 建立 95% CI，Wilcoxon signed-rank test
```

### 5.2 评估方案

**RQ1（有效性）**：
- 双桌 IMP，1000 局 deterministic rollout
- B05 vs A，配对 Wilcoxon signed-rank test，p < 0.05 为显著

**RQ2（机制验证）**：
- 训练过程中记录每步的 `partner_acc` 和 `opponent_acc`
- 计算 `info_ratio = partner_gain / (opponent_leak + 1e-8)`
- 报告 info_ratio 与最终 IMP 的 Spearman 相关系数

**RQ3（消融）**：
- β ∈ {0, 0.05, 0.2} 的 IMP 曲线
- B05 vs B0：量化 opponent penalty 的边际贡献

### 5.3 算力预算（L4 高 RAM）

| 环节 | 时间 | 单元（≈5/小时） |
|------|------|----------------|
| Phase 1 排雷 | 2 小时 | 10 |
| Phase 2 SL 预训练 | 5 小时 | 25 |
| Phase 3 主实验（5 agent） | 36 小时 | 180 |
| Phase 4 多 seed（2×2） | 26 小时 | 130 |
| 评估 + 重跑 buffer | 50 小时 | 250 |
| **总计** | **119 小时** | **595 单元** |

总预算 2000 单元，**实际使用约 30%**，有充裕余量。

---

## 6. 统计分析

- **主要指标**：双桌 IMP（消除牌力运气）
- **检验**：Wilcoxon signed-rank test（per-deal IMP 为重尾分布，非参数检验更鲁棒）
- **置信区间**：bootstrap 95% CI，3 个种子
- **最小有效差异**：0.2 IMP（IMP 方差 ≈ 3.5，1000 局评估有足够统计功效）

---

## 7. 项目结构

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py       # 完整叫牌环境（38 动作，dealer 轮换）
│   └── dual_table_env.py           # 双桌 IMP 环境
├── networks/
│   ├── policy_net.py               # MLPPolicyNetwork（301 维），ValueNetwork
│   └── belief_net.py               # BeliefNetwork（LSTM），DualInfoComputer
├── utils/
│   ├── scoring.py
│   ├── imp.py
│   ├── dds_data.py                 # 200 万副 DDS 预计算数据加载
│   ├── running_stats.py
│   ├── hand_features.py            # hand_to_belief_target()，48 维
│   └── openspiel_loader.py         # ← Phase 0 新增：SAYC 数据 → BCDataset
├── algorithms/
│   ├── mappo.py                    # HAPPO：actor_n/s + critic_n/s 完全独立
│   ├── ippo.py
│   └── behavioral_cloning.py
├── subgames/
│   ├── competitive_env.py          # 1H-1S 竞叫子博弈（Phase 1 排雷用）
│   ├── subgame_trainer.py          # 批量 rollout，FSP，JIT burn-in，r_info
│   └── action_mask.py
├── experiments/
│   ├── train.py                    # 主实验入口
│   └── subgame_validation.py       # 子游戏验证（排雷用）
├── tests/
│   └── test_all.py
├── data/
│   ├── full_2m.npz                 # 200 万副完整叫牌 DDS 数据
│   └── sayc_train/                 # OpenSpiel SAYC 数据（下载后存放）
├── results/
├── setup_project.py
└── requirements.txt
```

---

## 8. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Actor 历史编码 | MLP + 301 维展开 | 批量 rollout 友好；与 OpenSpiel 格式对齐；加速 3-5× |
| Belief 历史编码 | LSTM | 推断任务需要叫牌顺序语义，LSTM 不可替代 |
| SL 数据来源 | OpenSpiel SAYC | 消除 BC 数据质量对 r_info 效果的干扰 |
| FSP | pool size=10 | 防止 policy cycling（Kita 消融证明必要） |
| DDS 数据规模 | 200 万副 | 已预计算，无需实时 DDS，RL 循环采样 |
| Critic 独立性 | N/S 各自独立 critic | 避免 phase 切换时灾难性遗忘（vl 爆炸问题） |
| RL lr | 1e-6 | 对齐 Kita；防止 RL 破坏 SL 学到的协议 |
| β 主配置 | 0.05 | "gentle breeze"：r_info 补充 IMP，不主导梯度 |

---

## 9. Stayman 子博弈（历史存档）

Stayman 子博弈（1NT-P-2C-P 固定前缀，EW 全 Pass）用于验证基础架构稳定性。

**结论**：BC 以 99.5% 准确率学会了 3-bit Stayman 协议（2♦/2♥/2♠），通信达到信息论上限。r_info 无法改进一个已经完美的发信机；EW 全 Pass 使 β term 结构性失活。这是**预期的 null result**，验证了基础架构的稳定性（无崩溃、Belief 正常、KL 有效）。

**详细记录**见 `README_stayman_archive.md`（含 P7–P52 全部 bug fix 历史）。

---

## 10. 快速开始

```bash
# 安装依赖
pip install torch numpy tqdm endplay pyyaml scipy

# 组装包结构
python setup_project.py

# 下载 SAYC 数据
gsutil -m cp -r gs://openspiel-data/bridge/train.txt data/sayc_train/

# Phase 1: Competitive 排雷（约 2 小时，L4）
python experiments/subgame_validation.py \
    --type competitive --seed 42 --quick

# Phase 2: SL 预训练（约 5 小时，L4 高 RAM）
python experiments/train.py \
    --mode sl --data data/sayc_train/train.txt \
    --epochs 30 --batch_size 256

# Phase 3: 主实验单个 agent（约 6-7 小时，L4 高 RAM）
python experiments/train.py \
    --mode rl --agent A \
    --data data/full_2m.npz \
    --sl_ckpt results/sl_base.pt \
    --seed 42

python experiments/train.py \
    --mode rl --agent B --beta 0.05 \
    --data data/full_2m.npz \
    --sl_ckpt results/sl_base.pt \
    --seed 42
```

---

## 11. 参考文献

1. Wei & Luke (2016). Lenient Learning in Independent-Learner Stochastic Cooperative Games. JMLR.
2. Hu et al. (2020). Other-Play for Zero-Shot Coordination. ICML.
3. Foerster et al. (2018). Counterfactual Multi-Agent Policy Gradients. AAAI.
4. Kita et al. (2024). A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI. IEEE CoG.
5. Lockhart et al. (2020). Human-Agent Cooperation in Bridge Bidding. NeurIPS.
6. Gong et al. (2024). Bridge Bidding via Deep RL and Belief MCTS. IEEE/CAA J. Autom. Sinica.
7. Elelimy et al. (2025). Rethinking the Foundations for Continual RL. RL Journal.

---

*README v1.0 — 2026年3月*
*上一版本研究方案：Bridge_COMA_Research_Proposal_v7.3.md*
*Stayman 实验历史：README_stayman_archive.md*
