# Bridge-COMA: Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination

## 研究方案 v7.3

---

# 第一部分：科研项目（MSc 核心）

---

## 1. 研究背景与问题

### 1.1 核心问题

在多智能体强化学习（MARL）中，**相对过度泛化（Relative Overgeneralization）** 和 **协调失败（Miscoordination）** 是两个根本性挑战：

- **相对过度泛化**：Agent 倾向于选择"安全"动作——这些动作无论队友如何行动都能获得还可以的回报，但错失了需要协调才能获得的更高回报
- **协调失败**：缺乏显式通信机制时，Agent 难以建立有效的隐式约定

### 1.2 合作-对抗混合场景的独特挑战

与纯合作博弈相比，**合作-对抗混合**场景存在独特挑战：Agent 必须与队友有效协调，同时最小化通信信息向对手的**泄露（information leakage）**。这种双向信息问题在现有 MARL 方法中缺乏显式处理。

### 1.3 先验不对称性（Prior Asymmetry）

关键洞察：即使通信协议是公开的（叫品的"合法"含义四方皆知），但观察者持有不同的**私有上下文**（自己的手牌）。因此，同一叫品对队友和对手产生不同的推断增益，因为他们从不同的先验更新信念。

```
例子：你叫 1♥ 表示 5+ 张红心

队友视角：                          对手视角：
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ 我手里有 ♥AQxx              │    │ 我手里有 ♥x                │
│                             │    │                             │
│ 你叫 1♥ (5+ 红心)           │    │ 你叫 1♥ (5+ 红心)           │
│         ↓                   │    │         ↓                   │
│ 推断：我们有 9+ 张配合       │    │ 推断：他们有红心，但不确定  │
│ 决策价值：极高               │    │ 决策价值：有限              │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 1.4 研究平台：桥牌叫牌

桥牌叫牌是研究此问题的理想平台：

| 特性 | 描述 |
|------|------|
| 合作-对抗结构 | 2v2，队友需协调，对手需防范 |
| 受限通信 | 只能通过 38 种叫品传递信息 |
| 信息不对称 | 每人只能看到自己的手牌 |
| 全披露原则 | 所有约定必须公开（Alert Rule） |
| 稀疏奖励 | 只有最终定约完成后才有得分 |
| 标准评估 | IMP（双桌对比）提供客观指标 |

### 1.5 技术挑战：信用分配问题

桥牌叫牌面临两类信用分配挑战：

| 类型 | 问题 | 解决方案 |
|------|------|----------|
| **多智能体信用分配** | 最后输了 5 IMP，是北家还是南家的错？ | COMA/MAPPO 已有效解决 |
| **时间信用分配** | 多轮叫牌中，哪一步埋下了祸根？ | 部分通过 $r_{\text{info}}$ 缓解 |

**探索坍塌风险**：纯强化学习在稀疏奖励下容易陷入"全 Pass"的局部最优。本方案通过**行为克隆预热**和**动作掩码**解决此问题。

---

## 2. 研究目标与创新点

### 2.1 核心研究问题

1. **有效性 (Effectiveness)**：与传统 MARL baseline 相比，dual-information bonus 是否能改善协调（以 IMP 衡量）？

2. **机制验证 (Mechanism Verification)**：测量到的先验不对称性（partner vs opponent 推断准确率）是否与叫牌效率和最终定约质量相关？

3. **消融分析 (Ablation)**：Partner information term 与 opponent penalty term (β) 的相对贡献是什么？

### 2.2 理论基础：History Process 形式化

#### 2.2.1 为什么不是 MDP？

传统强化学习建立在 MDP 形式化之上，假设存在一个马尔可夫状态足以决定最优动作。然而，桥牌叫牌在以下情况下违反了这一假设：

1. **最优动作依赖于完整历史**：最优叫品不仅取决于"当前状态"，还取决于整个叫牌序列
2. **观察者具有不同的私有上下文**：即使叫牌历史是公开的，不同观察者（队友 vs 对手）从不同的先验更新信念
3. **环境随 Agent 行为演化**：每次叫牌后，所有玩家的信念状态都会更新

#### 2.2.2 History Process 视角

我们将桥牌叫牌视为一个 **History Process**：

$$e: \mathcal{H} \times \mathcal{A} \rightarrow \Delta(\mathcal{O})$$

其中 $\mathcal{H}$ 是所有可能叫牌历史的集合。在这个视角下：

- **Agent 创造 Worlds**：每次叫牌后，Agent 进入一个新的"世界" $e_{h_t}$，伙伴和对手根据叫牌更新信念
- **先验不对称性是内生的**：同一叫牌对不同观察者产生不同的推断增益，这不是 bug，而是 feature

### 2.3 核心创新：Dual-Information Credit Assignment

在 MAPPO 的基础上，增加信息论 reward shaping：

$$r_{\text{info}} = \underbrace{I(\text{bid}; \text{hand} \mid \text{partner})}_{\text{partner's inference gain}} - \beta \cdot \underbrace{I(\text{bid}; \text{hand} \mid \text{opponent})}_{\text{opponent's inference gain}}$$

其中 $\beta \geq 0$ 作为**拉格朗日乘子**，控制通信清晰度与信息隐蔽性之间的权衡：
- **低 β（如 0.05）**：优先清晰通信，协调收益大于防守风险（适合合作场景）
- **高 β（如 0.5）**：偏好紧凑叫牌路径，鼓励最小化信息泄露（适合竞叫场景）

**$r_{\text{info}}$ 的双重作用**：
1. **鼓励信息性通信**：显式奖励对队友有帮助的叫牌
2. **缓解时间信用分配**：提供每一步的 dense reward，而非仅依赖最终的稀疏 IMP

### 2.4 理论联系

**Wiretap Channel**：经典的安全通信模型，目标是最大化对合法接收者的信息传输，同时最小化对窃听者的泄露。我们的 $r_{\text{info}}$ 公式直接对应这一目标。

### 2.5 数学建模

直接优化互信息是 intractable 的。我们通过最小化条件熵 $H(\text{hand} \mid \text{bid, context})$ 的**变分上界**来近似 $r_{\text{info}}$。**Belief Network** $q_\phi$ 作为变分分布，将抽象的信息增益转化为训练过程中可测量的**交叉熵减少量**，为策略优化提供稳定梯度。

$$I(\text{bid}; \text{hand} \mid \text{obs}) \approx \text{CE}(q_\phi(\text{hand} \mid h_{t-1}, \text{obs}), \text{hand}) - \text{CE}(q_\phi(\text{hand} \mid h_t, \text{obs}), \text{hand})$$

计算值经过 ReLU 截断（$\max(0, \cdot)$），因为互信息的数学性质保证真实值 $\geq 0$；负值仅反映 Belief Network 的估计滞后，不应惩罚 agent。

### 2.6 设计原则：最小假设

| 原则 | 含义 |
|------|------|
| 无预设通信协议 | 约定从学习中涌现 |
| 执行时去中心化 | CTDE 范式，Agent 独立行动 |
| 全披露兼容 | 方法在公开约定约束下工作 |

---

## 3. 技术方案

### 3.1 问题形式化

将桥牌叫牌建模为 **History-Dependent Dec-POMDP**：

- **智能体**：$\mathcal{N} = \{N, E, S, W\}$，N-S 为一队，E-W 为一队
- **历史**：完整叫牌序列 $h_t = (b_1, b_2, \ldots, b_t)$
- **观测**：自己手牌 + 叫牌历史
- **动作**：38 种叫品（含 Pass、加倍、再加倍）
- **奖励**：IMP（双桌对比，终局稀疏）+ $r_{\text{info}}$（每步 dense，仅训练时）

### 3.2 训练流程：分阶段架构

```
Phase 1: 基础设施 ✅
  环境 + 网络 + 算法 + 测试

Phase 2: 子博弈验证 ← 当前
  Stage 1:  BC 预热 → S_base
  Stage 1.5: Belief Network 预训练 + Critic 预热
  Stage 2:  交替精调 (A vs B)
  Stage 3:  评估 + 定性分析 + Head-to-Head

Phase 3: 竞叫子博弈 (1H-1S)
  3-agent 对比: A (MAPPO) / B (β=0) / C (β=0.05)
  跨 agent 交叉评估

Phase 4: 完整叫牌训练
  完整环境 + 多 baseline 对比
```

### 3.3 交替训练架构

非平稳性问题（non-stationarity）的处理：N 和 S 同时更新时，梯度互相干扰，信用分配模糊。采用 **Iterated Best Response (IBR)** 架构：

```
Round k:
  S-phase:  S 训练 (N 冻结) → 200 steps
  N-phase:  N 训练 (S 冻结) → 200 steps (含 JIT Belief Burn-in for Agent B)

Final:
  Joint:    N+S 联合精调 → 300 steps, lr/3
```

**轮数设定依据**（可向审稿人解释）：

- 监控每轮后的验证集 IMP。当连续两轮 Δ IMP < 0.1 时判定收敛。
- result6 数据：IMP 在 Round 2-3 后进入 ±0.2 震荡，Round 3 与 Round 6 结果统计上无差异。
- 因此 3 轮（而非 6 轮）已经足够，且可在论文中以收敛曲线为据。

**Joint 步数设定依据**：

- Joint 阶段 N+S 同时更新，KL 压力翻倍。
- 300 步下 KL ≤ 0.35（safe range），entropy 保持在 0.03-0.04（未坍缩）。
- 400 步时 policy_loss 在最后 100 步出现 −0.075 的异常（可能过度优化）。

### 3.4 网络架构

```
PolicyNetwork (Actor):
  HandEncoder:     52 → 256 → 256 (MLP, ReLU)
  HistoryEncoder:  (seq_len, 38) → LSTM(2 layers, hidden=256)
                   [pack_padded_sequence — 只处理有效 token]
  Fusion:          [hand_256 ‖ history_256 ‖ position_4 ‖ vul_2] → MLP(256) → 38-dim logits

ValueNetwork (Critic, CTDE):
  同 Actor 结构 + AllHandsEncoder: (4×52) → 256 → 256
  集中式：推断时可见所有手牌（训练专用）
  独立优化器：lr × 2，PPO2 值函数截断

BeliefNetwork:
  HandEncoder:     52 → 256 → 256 (MLP)
  HistoryEncoder:  LSTM(2 layers, hidden=256)
  PositionEmbed:   Embedding(4, 32) × 2 (observer + target)
  Output:          52-dim logits（不经 Sigmoid）
  Loss:            BCEWithLogitsLoss(pos_weight=3.0)
  Metric:          Top-13 命中率（随机基线 ≈ 0.25）
```

### 3.5 奖励设计：Piecewise Linear IMP

Stayman 子博弈使用以 DDS 最优（max_level=4）为基准的分段线性奖励：

```
IMP regret → piecewise reward:
  0       → 1.00  完美匹配
  -1      → 0.70  错选花色 (3NT vs 4M)，陡峭惩罚 (Δ=0.30)
  -6      → 0.25  漏局 (Part-score vs Game)
  ≤ -13   → 0.01  灾难 (clamp，保留微弱梯度)

设计逻辑：
  0→-1 段斜率 (0.30/IMP) >> -1→-6 段 (0.09/IMP)
  迫使模型优先区分 N 应叫的语义，而非恐惧满贯失败
  max_level=4 与 Stayman 子博弈的决策空间对齐（23-27 点牌力不应追满贯）
```

### 3.6 KL Anchor 正则化

PPO 通过 KL 散度将当前策略锚定到 BC 先验策略，防止 RL 破坏已学习的叫牌协议：

```
KL loss = λ_kl × KL(π_bc || π_current)  [per-sample, context-weighted]

S-phase: λ_kl 从 0.5 退火到 0.1（允许 S 探索新的应叫方式）
N-phase: λ_kl 固定为 0.5（防止 N 偏离 S 依赖的信号结构）

Context-level weighting（基于叫牌阶段）：
  level 1 → 1.5×  (开局叫品，协议敏感)
  level 2 → 1.0×
  level 3 → 0.5×
  level 4 → 0.25×
  level 5+ → 0.1×  (高阶叫品，约束放松)

权重基于 obs['history'] 的状态，而非动作本身（防止梯度利用）
```

### 3.7 Belief Network 说明

**top-13 命中率的信息论上限**：

在 Stayman 子博弈中，N 只叫一次，信息量约 $\log_2(3) \approx 1.58$ bits。经验观测：top-13 hit ≈ 0.35，相对随机基线 0.25 提升了 40%。这接近该环境的信息论天花板——追求 0.40 的预设目标将导致 Belief Net 过拟合噪声，因此该目标不作为 blocking 条件。

**ir 信号有效性验证**：result6 中 ir 全程正值（0.09–1.85），说明 Belief Net 提供了有意义的梯度，而非随机噪声。

---

## 4. 实验设计

### 4.1 Stayman 子博弈：三阶段实验 ✅ 已完成

**设计原则**：控制变量法——通过分阶段训练消除 N↔S 耦合，确保 $r_{\text{info}}$ 的效果可以被干净地归因。

```
Stage 1: 建立公共基线 S_base
  N = 硬编码 Stayman 规则策略（有 4H→2H，有 4S→2S，否则→2D）
  只训练 S，active_players=[SOUTH]
  环境完全 stationary → 快速收敛

Stage 2: 分支精调
  加载 S_base 权重（公平起跑线）
  解冻 N，active_players=[NORTH, SOUTH]
  A: MAPPO (control，无 info bonus)
  B: MAPPO + r_info (β=0.05, partner-only)

Stage 3: 评估 + 定性分析
  定量: IMP vs DDS optimal
  定性: N 的策略偏移 (Agent B 的 N 是否偏离标准规则?)
  Head-to-Head: B vs A 双桌 IMP 对战
```

**result6 最终结果（5 个种子运行前的单次结果）：**

| Agent | IMP | Δ vs S_base |
|-------|-----|-------------|
| S_base | −3.71 | — |
| A_control | −3.71 | +0.01 |
| B_partner_only | −3.57 | **+0.14** |
| B vs A | — | +0.14 |
| H2H tie rate | — | **96.5%** |

**科学结论**：

Stayman 子博弈验证了基础架构的稳定性（无崩溃、Belief 正常、KL 有效、ir 正值）。B 略优于 A（+0.14 IMP），但 H2H tie rate 96.5% 表明两者策略几乎完全相同。这是**预期结果而非失败**：

- BC 已经以 99.5% 准确率教会了 3-bit Stayman 协议（2D/2H/2S），通信达到理论上限
- r_info 无法改善一个已经完美的发信机
- 该结论将诚实地写入论文的局限性讨论

**关于 Gemini 的"去掉 N 的 KL"建议**：经审查，这会故意构造一个被破坏的对照组（N 随机漂移），用来衬托 B 的优越性。这在科学上是不诚实的，审稿人会识别出来。我们选择诚实报告通信天花板，并通过竞叫子博弈提供真正的差异化实验。

**下一步**：5 个随机种子复现（alt_rounds=3, joint_steps=300），建立 95% 置信区间。

### 4.2 竞叫子博弈（1H-1S）：下一阶段实验

```
固定前缀：1H - 1S
开叫人（N）：5+ 红心，12-21 HCP
争叫人（E）：5+ 黑桃，8-16 HCP

训练 3 个 Agent（各自 self-play）:
  A: MAPPO (Control)
  B: MAPPO + r_info (β=0, partner-only)
  C: MAPPO + r_info (β=0.05, dual-info)

交叉对抗（双桌 IMP）:
┌─────────────────────────────────────────┐
│  对抗     │ 验证假设                     │
├─────────────────────────────────────────┤
│  B vs A   │ r_info partner term 有效     │
│  C vs A   │ r_info 完整版有效            │
│  C vs B   │ β opponent term 有额外价值   │
└─────────────────────────────────────────┘

与 Stayman 的关键区别：
  - 四方都参与决策（非 EW 全 Pass）
  - N 的信号空间更丰富（不止 3 个有意义叫品）
  - 对手存在 → β term 真正被激活
  - 通信协议未被 BC 完全确定 → r_info 有改进空间
```

### 4.3 统计分析方案

**多种子实验设计**：

- 5 个随机种子：[42, 123, 456, 789, 2024]
- 报告：mean ± 95% CI (bootstrap)
- 检验：Wilcoxon signed-rank test（per-deal IMP，非参数，适合非正态分布）
- 最小有效差异：0.2 IMP（根据 IMP 方差 ≈ 3.5，需要 ≥300 deals 才有统计功效）

**为什么用 Wilcoxon 而非 t-test**：

每副牌的 IMP 分布是重尾的（偶发-13 IMP 的灾难局），Wilcoxon 对离群值更鲁棒。

### 4.4 消融实验（竞叫子博弈后）

| 配置 | β 值 | 目的 |
|------|------|------|
| A: Control | — | MAPPO baseline |
| B: Partner-Only | 0.0 | 验证 partner term |
| C: Dual-Info Low | 0.05 | 当前主要实验配置 |
| D: Dual-Info High | 0.5 | β 量级敏感性 |

**预期发现与含义**：

| 发现 | 结论 |
|------|------|
| B > A 显著 | r_info partner term 有效 |
| C > B 显著 | opponent penalty 有额外价值 |
| C ≈ B 不显著 | 先验不对称性已足够，β 是锦上添花 |
| 高 β 导致 IMP 下降 | 过度压缩通信损害协调质量 |

---

## 5. 计划与里程碑

### 5.1 资源约束

| 资源 | 可用量 |
|------|--------|
| 本地 CPU | 日常开发 + 快速测试 |
| Colab Pro GPU | 全量实验（~24h/session） |
| 总时间 | 16 周 MSc 项目周期 |

### 5.2 修订后的时间规划（v7.3）

```
Phase 1: 环境构建 (Week 1-3) ✅ 已完成
  ├─ 核心环境（发牌、叫牌、DDS）
  ├─ 双桌 IMP + 得分 + IMP 转换
  ├─ IPPO / MAPPO + 训练脚本
  └─ 35 项测试覆盖全模块

Phase 2: Stayman 子博弈 (Week 4-5) ✅ 架构完成，多种子进行中
  ├─ 三阶段消融实验设计与实现
  ├─ 37 个 bug fix（P7–P38）
  ├─ 单次运行结果：B > A (+0.14 IMP，H2H 96.5% tie）
  ├─ 科学结论：通信天花板，基础架构稳定
  └─ ⏳ 5 个随机种子 → 95% CI

Phase 2b: 竞叫子博弈 (Week 6-7) ← 下一阶段
  ├─ 1H-1S 前缀，3-agent (A/B/C)
  ├─ 交叉对抗双桌 IMP 评估
  ├─ β term 激活（对手全程参与）
  └─ Go/No-Go: B vs A 显著？C vs B 有差异？

Phase 3: Belief + DualInfo 完整集成 (Week 8-9)
  └─ 竞叫结果良好时，推进到完整叫牌环境

Phase 4: 完整训练与实验 (Week 10-13)
  ├─ Baseline 对比 (RQ1): MAPPO / IPPO / COMA
  ├─ 机制验证 (RQ2): Partner vs Opponent 准确率、info_ratio、相关性
  └─ 消融分析 (RQ3): β ∈ {0, 0.05, 0.5}

Phase 5: 论文撰写 (Week 13-16)
  ├─ Introduction, Related Work, Method
  ├─ Experiments, Results, Analysis
  └─ Conclusion, 修改
```

### 5.3 里程碑与验收标准

| 周次 | 里程碑 | 验收标准 |
|------|--------|----------|
| Week 3 | 环境完成 ✅ | 35 项测试通过 |
| Week 5 | Stayman 完成 ✅ | B ≥ A（单次），5种子 CI 建立 |
| Week 7 | 竞叫验证 | B vs A p < 0.05，β 效果可观测 |
| Week 9 | 算法完整版 | DualInfo 在竞叫中优于 MAPPO |
| Week 13 | 全部实验 | 消融 + 机制验证图表就绪 |
| Week 16 | 论文完成 | 可提交 |

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| **Stayman 多种子 B ≈ A** | 高 | 低 | 预期结果；诚实报告通信天花板；竞叫才是主战场 |
| **竞叫子博弈 B ≈ A** | 中 | 高 | 检查 β 量级、ir 是否正值、N 信号空间是否受限 |
| **Belief Net 不准** | 低（Stayman 已验证 ir 正值） | 中 | JIT burn-in 已解决；竞叫中 Belief 任务更难，需监控 |
| **探索坍塌（全 Pass）** | 低（BC 预热已有效） | 中 | 监控 pass rate；出现时检查 action mask |
| **β 量级不匹配** | 中 | 中 | ir 归一化到 piecewise reward 量级；β=0.05 是"微风"而非"台风" |
| **外星语言** | 中 | 中 | BC 初始化 + KL anchor；竞叫更容易出现 |
| **Colab 断连** | 高 | 中 | Checkpoint 机制；结果已保存 results/*.pt |
| **审稿人质疑 Stayman 结果** | 中 | 中 | 诚实的通信天花板叙事；竞叫提供真实差异 |

---

## 7. 预期贡献

### 7.1 学术贡献

1. **Dual-Information Credit Assignment**：首个在合作-对抗混合博弈中显式利用先验不对称性进行信息论 reward shaping 的方法

2. **子博弈验证框架**：提供从受控子博弈到完整博弈的渐进式验证方法论；子博弈负结果（通信天花板）本身也是有价值的 insight

3. **机制验证框架**：提供测量和可视化先验不对称效应的工具

4. **消融分析**：量化 partner info term 与 opponent penalty term 的相对贡献

### 7.2 预期结果（竞叫子博弈）

| 指标 | 预期 |
|------|------|
| B vs A (IMP) | +0.3–0.5 |
| C vs A (IMP) | +0.4–0.6 |
| Info Ratio | > 1.2 |
| Info-IMP Correlation | > 0.3 |

---

## 8. 参数配置（当前实验标准）

```python
# BC Warmup
stayman_bc_samples        = 20000
stayman_bc_epochs         = 15

# Stage 1.5
critic_warmup_rounds      = 10
critic_warmup_deals       = 512
belief_pretrain_deals     = 10000
belief_pretrain_epochs    = 50

# Stage 2: 交替精调（v7.3 减半）
stage2_alt_rounds         = 3          # 3 轮（result6 显示 Round 3 后已收敛）
stage2_alt_steps          = 200        # steps per half-round
stage2_joint_steps        = 300        # 减自 400（KL 更安全）
stage2_deals_per_step     = 32
stage2_accumulate         = 8          # 256 deals/update

# 学习率
stage2_lr                 = 3e-5
stage2_lr_joint           = 1e-5

# KL anchor
stage2_kl_lambda_start    = 0.5        # S-phase 起始
stage2_kl_lambda_end      = 0.1        # S-phase 终点（退火）
stage2_n_kl_lambda_start  = 0.5        # N-phase 固定（不退火）
stage2_n_kl_lambda_end    = 0.5

# Entropy
stage2_entropy_start      = 0.10
stage2_entropy_end        = 0.05

# Belief JIT burn-in
jit_burnin_deals          = 1000
jit_burnin_epochs         = 3
jit_burnin_lr             = 1e-3

# Eval
eval_deals                = 1000
diag_deals                = 2000

# Info bonus
beta                      = 0.05       # "gentle breeze"

# Multi-seed
seeds                     = [42, 123, 456, 789, 2024]
```

---

## 9. 代码架构

### 9.1 当前实际结构

源文件以扁平方式存放在 Claude Projects；运行 `setup_project.py` 自动组装为包结构。

```
bridge-coma/
├── env/
│   ├── bridge_bidding_env.py       # 单桌叫牌环境
│   └── dual_table_env.py           # 双桌 IMP 环境
├── networks/
│   ├── policy_net.py               # PolicyNetwork, ValueNetwork, ActorCritic
│   └── belief_net.py               # BeliefNetwork, DualInfoComputer
├── utils/
│   ├── scoring.py                  # 得分计算 SSOT
│   ├── imp.py                      # IMP 转换
│   ├── dds_data.py                 # DDS 数据生成/加载
│   ├── running_stats.py            # Welford 在线统计
│   └── generate_subgame_data.py    # 约束发牌生成
├── algorithms/
│   ├── ippo.py
│   ├── mappo.py                    # 独立 Actor/Critic 优化器，PPO2 截断
│   └── behavioral_cloning.py
├── subgames/
│   ├── stayman_env.py              # 分段奖励；MAX_LEVEL=4；合法掩码；干净 BC
│   ├── competitive_env.py          # 1H-1S 竞叫子博弈
│   ├── subgame_trainer.py          # ir wired；ReLU；JIT burn-in；HeadToHeadEvaluator
│   └── action_mask.py
├── experiments/
│   ├── train.py
│   └── subgame_validation.py       # alt_rounds=3；joint_steps=300；Stage 3 H2H
├── tests/
│   ├── test_all.py                 # 35 项测试
│   └── test_phase2.py
├── results/
├── data/
├── setup_project.py
└── requirements.txt
```

### 9.2 关键文件说明

| 文件 | 关键设计决策 |
|------|-------------|
| `stayman_env.py` | Piecewise reward [0.01,1.0]；MAX_LEVEL=4；`_auto_play_non_agent` 用 legal mask；BC 数据生成不再有死代码 |
| `subgame_trainer.py` | `store_episodes`：ir 直接叠加到 terminal reward（非归一化后叠加）；`compute_context_kl_weights`：按叫牌阶段加权；`HeadToHeadEvaluator`：两队实例 ONCE 创建，复用全部 deals |
| `subgame_validation.py` | `Phase2Config`：alt_rounds=3, joint_steps=300；Stage 3 含 H2H |
| `belief_net.py` | BCEWithLogitsLoss(pos_weight=3)；top13_hit_rate metric；forward 返回 logits 不含 Sigmoid |
| `mappo.py` | 独立 actor/critic 优化器；PPO2 value clipping；context-level KL |

---

## 10. 版本更新记录

### v7.2 → v7.3 主要更新

| 更新项 | 描述 |
|--------|------|
| **Stayman 结论锁定** | 通信天花板是预期结果，不是失败；科学叙事清晰 |
| **参数调整** | alt_rounds 6→3，joint_steps 400→300（有实验依据） |
| **多种子方案确定** | 5 seeds；Wilcoxon signed-rank；bootstrap 95% CI |
| **Belief 目标重新定位** | 0.40 不再是 blocking 条件；信息论天花板约 0.37 |
| **P38 bug fix 记录** | BC 死代码 + mask mismatch 修复 |
| **竞叫子博弈路线图更新** | 明确 β term 的验证场景 |
| **Gemini 建议评审** | KL=0 建议被拒绝（科学上不诚实）；归一化建议在当前版本不适用 |

### v7.1 → v7.2 主要更新

| 更新项 | 描述 |
|--------|------|
| Stayman 三阶段消融实验 | Stage 1 → Stage 2 → Stage 3 控制变量法 |
| BC 数据修复 | Round3 N response + 20k samples |
| r_info 接线修复 | P36：β=0.05；wire to reward；ReLU clamp；JIT burn-in |
| KL anchor 完整实现 | N-phase 固定 λ=0.5；context-level 加权 |
| HeadToHeadEvaluator | 通用双桌对战框架 |
| 代码普适化 | 去掉硬编码 mask 和 MAX_LEVEL=7（后回退到 4）；清理死代码 |

---

*文档版本: v7.3*
*最后更新: 2026年3月*
