# Bridge-COMA: Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination

## 研究方案 v6.3

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

---

## 2. 研究目标与创新点

### 2.1 核心研究问题

1. **有效性 (Effectiveness)**：与传统 MARL baseline 相比，dual-information bonus 是否能改善协调（以 IMP 衡量）？

2. **机制验证 (Mechanism Verification)**：测量到的先验不对称性（partner vs opponent 推断准确率）是否与叫牌效率和最终定约质量相关？

3. **消融分析 (Ablation)**：Partner information term 与 opponent penalty term (β) 的相对贡献是什么？Opponent penalty 是否必要，还是仅靠先验不对称性就足够？

### 2.2 理论基础：History Process 形式化

#### 2.2.1 为什么不是 MDP？

传统强化学习建立在 MDP 形式化之上，假设存在一个马尔可夫状态足以决定最优动作。然而，Elelimy et al. (2025) 在 *Rethinking the Foundations for Continual Reinforcement Learning* 中指出，这种形式化在以下情况下是不适当的：

1. **最优动作依赖于完整历史**：在桥牌叫牌中，最优叫品不仅取决于"当前状态"，还取决于整个叫牌序列
2. **观察者具有不同的私有上下文**：即使叫牌历史是公开的，不同观察者（队友 vs 对手）从不同的先验更新信念
3. **环境随 Agent 行为演化**：每次叫牌后，所有玩家的信念状态都会更新，创造出新的"世界"

#### 2.2.2 History Process 视角

我们将桥牌叫牌视为一个 **History Process**：

$$e: \mathcal{H} \times \mathcal{A} \rightarrow \Delta(\mathcal{O})$$

其中 $\mathcal{H}$ 是所有可能叫牌历史的集合。在这个视角下：

- **Agent 创造 Worlds**：每次叫牌后，Agent 进入一个新的"世界" $e_{h_t}$，伙伴和对手根据叫牌更新信念
- **先验不对称性是内生的**：同一叫牌对不同观察者产生不同的推断增益，这不是 bug，而是 feature

#### 2.2.3 Deviation Regret 的启示

Elelimy et al. 提出的 **Deviation Regret** 概念对我们的评估方法有重要启示：

$$\rho_T = \frac{1}{T}\sum_{t=1}^{T} \left[ \mathbb{E}[G_t | \phi(\sigma), H_{t-1}] - \mathbb{E}[G_t | \sigma, H_{t-1}] \right]$$

**核心思想**：评估 Agent 应该基于它所遇到的"世界"，而非某个理想化的最优策略。这启发我们：

- 单次对局的 IMP 结果混淆了**叫牌质量**和**运气**（如外面的极端分布）
- 更好的评估应该问："给定这个叫牌历史，是否存在系统性的偏离能带来更好的期望结果？"

### 2.3 核心创新：Dual-Information Credit Assignment

在 MAPPO/COMA 的基础上，增加信息论 reward shaping：

$$r_{\text{info}} = \underbrace{I(\text{bid}; \text{hand} \mid \text{partner})}_{\text{partner's inference gain}} - \beta \cdot \underbrace{I(\text{bid}; \text{hand} \mid \text{opponent})}_{\text{opponent's inference gain}}$$

其中 $\beta \geq 0$ 作为**拉格朗日乘子**，控制通信清晰度与信息隐蔽性之间的权衡：
- **低 β**：优先清晰通信，即使对手也能获得信息，协调收益大于防守风险
- **高 β**：偏好"高效"或"紧凑"的叫牌路径，鼓励用最少的信息交换达到最优定约

### 2.4 理论联系

**Wiretap Channel**：经典的安全通信模型，目标是最大化对合法接收者的信息传输，同时最小化对窃听者的泄露。我们的 $r_{\text{info}}$ 公式直接对应这一目标。

### 2.5 数学建模

直接优化互信息是 intractable 的。我们通过最小化条件熵 $H(\text{hand} \mid \text{bid, context})$ 的**变分上界**来近似 $r_{\text{info}}$。**Belief Network** $q_\phi$ 作为变分分布，将抽象的信息增益转化为训练过程中可测量的**交叉熵减少量**，为策略优化提供稳定梯度。

### 2.6 设计原则：最小假设

| 原则 | 含义 |
|------|------|
| 无预设通信协议 | 约定从学习中涌现 |
| 执行时去中心化 | CTDE 范式，Agent 独立行动 |
| 全披露兼容 | 方法在公开约定约束下工作 |

---

## 3. 技术方案

### 3.1 问题形式化

将桥牌叫牌建模为 **History-Dependent Dec-POMDP**（而非标准 Dec-POMDP，强调对完整历史的依赖）：

- **智能体**：$\mathcal{N} = \{N, E, S, W\}$，N-S 为一队，E-W 为一队
- **历史**：完整叫牌序列 $h_t = (b_1, b_2, \ldots, b_t)$
- **观测**：自己手牌 + 叫牌历史（不知道其他人的手牌）
- **动作**：38 种叫品
- **奖励**：IMP（双桌对比，只在叫牌结束后给出）

### 3.2 双桌 IMP 计算

```
┌─────────────────────────────────────────────────────────────────┐
│                        双桌 IMP 计算                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  同一副牌 Deal                                                   │
│                                                                 │
│  桌 1 (Open Room):                                              │
│  ┌─────────────────────────────────────────┐                   │
│  │  N-S: Agent    vs    E-W: Agent         │ → Score_1         │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  桌 2 (Closed Room):                                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  N-S: Agent    vs    E-W: Agent         │ → Score_2         │
│  │  (位置互换，同一副牌)                     │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  IMP_NS = score_to_imp(Score_1 - Score_2)                      │
│  IMP_EW = -IMP_NS                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
class DualTableEnv:
    """双桌环境，Self-play 计算 IMP"""
    
    def __init__(self, agent):
        self.agent = agent
        self.dds = DDSolver()
    
    def play_deal(self, deal):
        # 桌 1: 正常位置
        contract_1, declarer_1 = self.run_bidding(deal, swap=False)
        score_1 = self.dds.calculate_score(deal, contract_1, declarer_1)
        
        # 桌 2: N-S 与 E-W 互换
        contract_2, declarer_2 = self.run_bidding(deal, swap=True)
        score_2 = self.dds.calculate_score(deal, contract_2, declarer_2)
        
        # IMP 转换
        imp_ns = self.score_to_imp(score_1 - score_2)
        return imp_ns
    
    @staticmethod
    def score_to_imp(diff):
        """标准 IMP 转换表"""
        imp_table = [
            (20, 0), (50, 1), (90, 2), (130, 3), (170, 4),
            (220, 5), (270, 6), (320, 7), (370, 8), (430, 9),
            (500, 10), (600, 11), (750, 12), (900, 13),
            (1100, 14), (1300, 15), (1500, 16), (1750, 17),
            (2000, 18), (2250, 19), (2500, 20), (3000, 21),
            (3500, 22), (4000, 23), (float('inf'), 24)
        ]
        abs_diff = abs(diff)
        sign = 1 if diff >= 0 else -1
        
        for threshold, imp in imp_table:
            if abs_diff < threshold:
                return sign * imp
        return sign * 24
```

### 3.3 算法：Dual-Information Bonus

#### 3.3.1 核心算法框架

```python
class DualInfoAgent:
    """
    支持 MAPPO 或 COMA 的 Dual-Info 算法
    """
    def __init__(self, config):
        # 网络
        self.policy_net = PolicyNetwork(config)
        self.value_net = ValueNetwork(config)
        self.belief_net = BeliefNetwork(config)
        
        # 算法选择（主要使用 MAPPO，因为大动作空间下方差更低）
        self.base_algorithm = config.base_algorithm  # 'mappo' or 'coma'
        
        # Info bonus 参数
        self.lambda_init = config.lambda_info    # 初始权重
        self.lambda_min = config.lambda_min      # 最终权重 (退火)
        self.anneal_steps = config.anneal_steps  # 退火步数
        self.beta = config.beta                  # 对手惩罚系数
        
        # 归一化统计量
        self.partner_info_stats = RunningStats()
        self.opponent_info_stats = RunningStats()
    
    def get_lambda(self, step):
        """Lambda 退火：训练后期降低 info bonus 权重"""
        progress = min(1.0, step / self.anneal_steps)
        return self.lambda_init - progress * (self.lambda_init - self.lambda_min)
    
    def compute_advantage(self, batch, step):
        """计算带有 Dual-Info Bonus 的 Advantage"""
        # 1. 基础 Advantage (GAE for MAPPO)
        if self.base_algorithm == 'mappo':
            base_adv = self.compute_gae(batch)
        else:  # coma
            base_adv = self.compute_coma_advantage(batch)
        
        # 2. Dual-Info Bonus (归一化)
        info_bonus = self.compute_normalized_dual_info_bonus(batch)
        
        # 3. 组合 (带退火)
        current_lambda = self.get_lambda(step)
        total_adv = base_adv + current_lambda * info_bonus
        
        return total_adv, {
            'base_adv': base_adv.mean().item(),
            'info_bonus': info_bonus.mean().item(),
            'lambda': current_lambda
        }
```

#### 3.3.2 归一化的 Dual-Info Bonus

```python
def compute_normalized_dual_info_bonus(self, batch):
    """
    计算归一化的 Dual-Info Bonus
    
    归一化确保 partner_gain 和 opponent_leak 在同一量级
    使得 β 的物理含义更清晰
    """
    bonuses = []
    
    for agent_id in [0, 2]:  # N-S (对称处理 E-W)
        partner_id = (agent_id + 2) % 4
        opponent_id = (agent_id + 1) % 4
        
        # 计算原始信息增益
        partner_gain = self.compute_info_gain(
            observer=partner_id, target=agent_id, batch=batch
        )
        opponent_leak = self.compute_info_gain(
            observer=opponent_id, target=agent_id, batch=batch
        )
        
        # 更新运行统计量
        self.partner_info_stats.update(partner_gain)
        self.opponent_info_stats.update(opponent_leak)
        
        # 归一化到相同量级 (z-score)
        partner_normalized = (
            (partner_gain - self.partner_info_stats.mean) / 
            (self.partner_info_stats.std + 1e-8)
        )
        opponent_normalized = (
            (opponent_leak - self.opponent_info_stats.mean) / 
            (self.opponent_info_stats.std + 1e-8)
        )
        
        # Dual-Info Bonus
        bonus = partner_normalized - self.beta * opponent_normalized
        bonuses.append(bonus)
    
    return torch.stack(bonuses).mean(dim=0)

def compute_info_gain(self, observer, target, batch):
    """
    计算 observer 对 target 手牌的信息增益
    
    I(bid; hand) ≈ H(hand | history_before) - H(hand | history_after)
                 ≈ Uncertainty_before - Uncertainty_after
    """
    # 叫牌前后的 belief
    belief_before = self.belief_net(
        batch.hands[observer], batch.history_before
    )
    belief_after = self.belief_net(
        batch.hands[observer], batch.history_after
    )
    
    # target 的真实手牌
    target_hand = batch.hands[target]
    
    # 不确定性 = 交叉熵
    uncertainty_before = F.binary_cross_entropy(
        belief_before, target_hand, reduction='none'
    ).sum(dim=-1)
    
    uncertainty_after = F.binary_cross_entropy(
        belief_after, target_hand, reduction='none'
    ).sum(dim=-1)
    
    # 信息增益 = 不确定性减少量
    info_gain = uncertainty_before - uncertainty_after
    
    return info_gain
```

#### 3.3.3 GAE (用于 MAPPO)

```python
def compute_gae(self, batch, gamma=0.99, lam=0.95):
    """
    GAE (Generalized Advantage Estimation) for MAPPO
    
    比 COMA counterfactual 更稳定，适合大动作空间
    """
    values = self.value_net(batch.observations)
    
    advantages = []
    gae = 0
    
    for t in reversed(range(len(batch.rewards))):
        if t == len(batch.rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = batch.rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)
```

### 3.4 训练流程：分阶段训练

```python
def train_dual_info_agent(config):
    """
    分阶段训练流程
    
    Phase 1: Belief 预热 (冻结 Policy)
    Phase 2: 联合训练 (Info Bonus 退火)
    """
    agent = DualInfoAgent(config)
    env = DualTableEnv(agent)
    
    # ==================== Phase 1: Belief 预热 ====================
    print("=" * 50)
    print("Phase 1: Belief Network Warmup")
    print("=" * 50)
    
    for step in range(config.belief_warmup_steps):
        batch = collect_rollouts(env, config.batch_size)
        
        # 只更新 Belief Network
        belief_loss = compute_belief_loss(agent.belief_net, batch)
        agent.belief_optimizer.step(belief_loss)
        
        if step % config.log_interval == 0:
            belief_acc = evaluate_belief_accuracy(agent)
            print(f"Step {step}: Belief Accuracy = {belief_acc:.3f}")
            
            if belief_acc > config.belief_threshold:
                print(f"Belief accuracy reached {config.belief_threshold}, "
                      f"proceeding to Phase 2")
                break
    
    # ==================== Phase 2: 联合训练 ====================
    print("=" * 50)
    print("Phase 2: Joint Training with Annealing")
    print("=" * 50)
    
    for step in range(config.joint_training_steps):
        batch = collect_rollouts(env, config.batch_size)
        
        # 计算 Advantage (带退火的 Info Bonus)
        advantages, adv_info = agent.compute_advantage(batch, step)
        
        # Policy Loss
        policy_loss = compute_policy_loss(agent.policy_net, batch, advantages)
        
        # Value Loss
        value_loss = compute_value_loss(agent.value_net, batch)
        
        # Belief Loss (持续更新)
        belief_loss = compute_belief_loss(agent.belief_net, batch)
        
        # 总 Loss
        total_loss = (
            policy_loss + 
            config.value_coef * value_loss + 
            config.belief_coef * belief_loss
        )
        
        agent.optimizer.step(total_loss)
        
        # ==================== 监控 ====================
        if step % config.log_interval == 0:
            metrics = evaluate_agent(agent, env)
            
            log({
                # 训练状态
                'step': step,
                'lambda': adv_info['lambda'],
                'base_adv': adv_info['base_adv'],
                'info_bonus': adv_info['info_bonus'],
                
                # Belief 质量
                'belief_accuracy': metrics['belief_accuracy'],
                
                # 信息流分析
                'partner_info_mean': agent.partner_info_stats.mean,
                'opponent_info_mean': agent.opponent_info_stats.mean,
                'info_ratio': metrics['info_ratio'],
                
                # 性能
                'imp_self_play': metrics['imp'],
                'avg_bidding_rounds': metrics['avg_rounds'],
                
                # 健康检查
                'correlation_info_imp': metrics['correlation'],
            })
```

### 3.5 网络架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     网络架构                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入                                                           │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │  Hand (52d)  │     │History (seq) │                         │
│  │  one-hot     │     │  bid tokens  │                         │
│  └──────┬───────┘     └──────┬───────┘                         │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │  Hand MLP    │     │ History LSTM │  ← 完整历史，非马尔可夫   │
│  │  256 → 256   │     │   dim=256    │                         │
│  └──────┬───────┘     └──────┬───────┘                         │
│         │                    │                                  │
│         └────────┬───────────┘                                  │
│                  ▼                                              │
│         ┌──────────────┐                                        │
│         │    Concat    │                                        │
│         │    (512d)    │                                        │
│         └──────┬───────┘                                        │
│                │                                                │
│       ┌────────┼────────┐                                       │
│       ▼        ▼        ▼                                       │
│  ┌────────┐┌────────┐┌────────┐                                │
│  │ Policy ││ Value  ││ Belief │                                │
│  │ 512→38 ││512→1   ││512→52  │                                │
│  │softmax ││  -     ││sigmoid │                                │
│  └────────┘└────────┘└────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 限制说明

本项目使用 Double Dummy Solver (DDS) 评估定约得分，假设完美打牌。因此，信息泄露对防守决策的影响（如首攻选择）无法直接测量，这仍是未来工作的方向。

---

## 4. 实验设计

### 4.1 Baseline 对比（对应 RQ1）

| 方法 | 描述 |
|------|------|
| **MAPPO + Dual-Info (Ours)** | MAPPO + 归一化 Dual-Info Bonus + 退火 |
| MAPPO | 标准 MAPPO，无 info bonus |
| COMA | 标准 COMA，counterfactual baseline |
| IPPO | Independent PPO，无协调机制 |

**评估指标**：
- 双桌 Self-play IMP
- 收敛速度（达到目标 IMP 所需步数）

**预期结果**：

| 方法 | 预期 IMP 差距 |
|------|--------------|
| MAPPO + Dual-Info | 基准 |
| MAPPO | -0.3 ~ -0.5 |
| COMA | -0.2 ~ -0.4 |
| IPPO | -0.8 ~ -1.0 |

---

### 4.2 机制验证（对应 RQ2）

**目标**：验证 Dual-Info Bonus 确实产生了预期的先验不对称效应。

**指标**：

| 指标 | 定义 | 预期 |
|------|------|------|
| Partner Belief Accuracy | 队友预测我手牌的准确率 | 较高 |
| Opponent Belief Accuracy | 对手预测我手牌的准确率 | 较低 |
| Info Ratio | Partner Acc / Opponent Acc | > 1.2 |

**可视化**：

```python
def visualize_mechanism(agent, num_samples=10000):
    """机制验证可视化"""
    partner_accs = []
    opponent_accs = []
    info_bonuses = []
    imps = []
    
    for deal in generate_deals(num_samples):
        result = play_and_record(agent, deal)
        partner_accs.append(result['partner_belief_acc'])
        opponent_accs.append(result['opponent_belief_acc'])
        info_bonuses.append(result['info_bonus'])
        imps.append(result['imp'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. r_info 分布
    axes[0, 0].hist(info_bonuses, bins=50, alpha=0.7)
    axes[0, 0].set_xlabel('Info Bonus')
    axes[0, 0].set_title('Distribution of r_info')
    
    # 2. Partner vs Opponent 准确率
    axes[0, 1].scatter(partner_accs, opponent_accs, alpha=0.1, s=1)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 1].set_xlabel('Partner Belief Accuracy')
    axes[0, 1].set_ylabel('Opponent Belief Accuracy')
    axes[0, 1].set_title('Prior Asymmetry Effect')
    axes[0, 1].legend()
    
    # 3. Info Ratio 分布
    info_ratios = [p / (o + 1e-8) for p, o in zip(partner_accs, opponent_accs)]
    axes[1, 0].hist(info_ratios, bins=50, alpha=0.7)
    axes[1, 0].axvline(x=1.0, color='r', linestyle='--', label='Ratio=1')
    axes[1, 0].set_xlabel('Info Ratio (Partner/Opponent)')
    axes[1, 0].set_title(f'Mean Info Ratio: {np.mean(info_ratios):.3f}')
    axes[1, 0].legend()
    
    # 4. Info Ratio vs IMP 相关性
    axes[1, 1].scatter(info_ratios, imps, alpha=0.1, s=1)
    axes[1, 1].set_xlabel('Info Ratio')
    axes[1, 1].set_ylabel('IMP')
    corr = np.corrcoef(info_ratios, imps)[0, 1]
    axes[1, 1].set_title(f'Correlation: {corr:.3f}')
    
    plt.tight_layout()
    plt.savefig('mechanism_verification.png', dpi=150)
```

**分析**：
- 验证 Info Ratio > 1（先验不对称性存在）
- 分析 Info Ratio 与最终 IMP 的相关性

---

### 4.3 消融实验（对应 RQ3）

**目标**：量化 opponent penalty term (β) 的贡献。

**实验设置**：

| 配置 | β 值 | 含义 |
|------|------|------|
| Partner-Only | 0.0 | 只有 partner info gain |
| Low Penalty | 0.25 | 轻微 opponent penalty |
| Balanced | 0.5 | 平衡点 |
| High Penalty | 0.75 | 较强 opponent penalty |
| Max Penalty | 1.0 | 最大 opponent penalty |

**评估指标**：
- IMP
- Info Ratio
- 平均叫牌轮数

**预期结果**：

| β | IMP | Info Ratio | 叫牌轮数 | 解释 |
|---|-----|------------|----------|------|
| 0.0 | 基准 | ~1.2 | ~12 | 充分交流 |
| 0.5 | ≈基准 | ~1.3 | ~10 | 平衡 |
| 1.0 | ≈基准或略低 | ~1.4 | ~8 | 精简叫牌 |

**关键结论解释**：

| 发现 | 结论 |
|------|------|
| β=0 和 β=0.5 的 IMP 差异不显著 | Opponent penalty 非必要，先验不对称性已足够 |
| β=0.5 显著优于 β=0 | Opponent penalty 有额外价值 |
| 高 β 导致 IMP 下降 | 过度压缩通信损害协调质量 |

---

### 4.4 Benchmark 对比

| 对手 | 来源 | 评估方式 |
|------|------|----------|
| WBridge5 | 商业软件 | 1000+ 副牌，IMP |
| JPS | NeurIPS 2020 | 如能获取代码 |
| DRL+Belief MCTS | IEEE/CAA 2024 | 如能获取代码 |

---

### 4.5 Communication-Concealment Trade-off 分析（附加分析）

**目标**：绘制 β ∈ [0, 1] 的权衡曲线，展示通信-隐蔽性权衡。

**方法**：
1. 训练 β ∈ {0, 0.25, 0.5, 0.75, 1.0} 的 5 个 Agent
2. 测量每个 Agent 的 (IMP, Info Ratio, 叫牌轮数)
3. 绘制权衡曲线

```python
def plot_communication_concealment_tradeoff(results):
    """
    绘制 Communication-Concealment Trade-off
    
    results: dict mapping β -> (imp, info_ratio, avg_rounds)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    betas = sorted(results.keys())
    imps = [results[b]['imp'] for b in betas]
    ratios = [results[b]['info_ratio'] for b in betas]
    rounds = [results[b]['avg_rounds'] for b in betas]
    
    # IMP vs Info Ratio
    axes[0].plot(ratios, imps, 'o-')
    for b, r, i in zip(betas, ratios, imps):
        axes[0].annotate(f'β={b}', (r, i), textcoords="offset points", 
                         xytext=(5, 5), fontsize=9)
    axes[0].set_xlabel('Info Ratio (Concealment)')
    axes[0].set_ylabel('IMP (Performance)')
    axes[0].set_title('Performance vs Concealment')
    
    # IMP vs Bidding Rounds
    axes[1].plot(rounds, imps, 'o-')
    for b, rd, i in zip(betas, rounds, imps):
        axes[1].annotate(f'β={b}', (rd, i), textcoords="offset points", 
                         xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel('Avg Bidding Rounds (Communication)')
    axes[1].set_ylabel('IMP (Performance)')
    axes[1].set_title('Performance vs Communication')
    
    # All metrics vs β
    ax3 = axes[2]
    ax3.plot(betas, imps, 'o-', label='IMP', color='blue')
    ax3.set_xlabel('β')
    ax3.set_ylabel('IMP', color='blue')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(betas, ratios, 's--', label='Info Ratio', color='red')
    ax3_twin.set_ylabel('Info Ratio', color='red')
    
    ax3.set_title('Metrics vs β')
    
    plt.tight_layout()
    plt.savefig('communication_concealment_tradeoff.png', dpi=150)
```

---

### 4.6 健康监控指标

| 指标 | 健康范围 | 异常信号 | 处理方式 |
|------|----------|----------|----------|
| Belief Accuracy | > 0.6 | < 0.5 | 延长 warmup |
| Correlation(r_info, IMP) | > 0.2 | < 0.1 | 加速退火 |
| Avg Bidding Rounds | 8-15 | > 20 | 检查"话痨"行为 |
| Info Ratio | > 1.1 | < 1.0 | 检查 Belief Network |

---

## 5. 实现计划

### 5.1 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习 | PyTorch |
| 桥牌环境 | OpenSpiel / 自定义 |
| DDS | endplay |
| 实验追踪 | Weights & Biases |
| 可视化 | Matplotlib, Seaborn |

### 5.2 代码结构

```
bridge-coma/
├── core/
│   ├── env/
│   │   ├── bridge_bidding_env.py
│   │   └── dual_table_env.py
│   ├── networks/
│   │   ├── policy_net.py
│   │   ├── value_net.py
│   │   └── belief_net.py
│   └── utils/
│       ├── dds_wrapper.py
│       ├── imp_calculator.py
│       └── running_stats.py
│
├── algorithms/
│   ├── base_agent.py
│   ├── dual_info_agent.py
│   ├── mappo.py
│   ├── coma.py
│   └── ippo.py
│
├── experiments/
│   ├── configs/
│   │   ├── default.yaml
│   │   └── beta_sweep.yaml
│   ├── train.py
│   ├── evaluate.py
│   └── analysis/
│       ├── mechanism_verification.py
│       ├── ablation.py
│       └── tradeoff_analysis.py
│
└── scripts/
    ├── run_baseline_comparison.sh
    ├── run_beta_sweep.sh
    └── generate_figures.py
```

### 5.3 配置示例

```yaml
# configs/default.yaml

# 环境
env:
  type: dual_table
  num_deals_per_episode: 1

# 网络
network:
  hand_dim: 256
  history_dim: 256
  lstm_layers: 2

# 算法
algorithm:
  base: mappo
  gamma: 0.99
  gae_lambda: 0.95
  
  # Dual-Info 参数
  lambda_init: 0.5
  lambda_min: 0.1
  anneal_steps: 500000
  beta: 0.5

# 训练
training:
  belief_warmup_steps: 50000
  belief_threshold: 0.7
  joint_training_steps: 1000000
  batch_size: 256
  lr: 3e-4
  
  value_coef: 0.5
  belief_coef: 0.5
  entropy_coef: 0.01

# 日志
logging:
  log_interval: 100
  eval_interval: 10000
  save_interval: 50000
```

### 5.4 时间规划 (16 周)

```
┌─────────────────────────────────────────────────────────────────┐
│                   MSc 项目时间规划 (16 周)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 环境与基础设施 (Week 1-3)                             │
│  ├─ Week 1: 桥牌环境搭建，DDS 集成                              │
│  ├─ Week 2: 双桌 IMP 环境实现                                   │
│  ├─ Week 3: IPPO, MAPPO baseline 实现                          │
│  └─ 交付物: 可运行的训练框架                                    │
│                                                                 │
│  Phase 2: 核心算法实现 (Week 4-7)                               │
│  ├─ Week 4: Belief Network 实现                                │
│  ├─ Week 5: Belief 预热训练流程                                │
│  ├─ Week 6: Dual-Info Bonus (含归一化)                         │
│  ├─ Week 7: Lambda 退火 + 联合训练                             │
│  └─ 交付物: DualInfoAgent 可训练                               │
│                                                                 │
│  Phase 3: 实验 (Week 8-12)                                      │
│  ├─ Week 8-9: Baseline 对比 (RQ1)                              │
│  ├─ Week 10: 机制验证 + 可视化 (RQ2)                           │
│  ├─ Week 11: 消融实验 β ∈ [0,1] (RQ3)                          │
│  ├─ Week 12: Benchmark + Trade-off 分析                        │
│  └─ 交付物: 完整实验结果                                        │
│                                                                 │
│  Phase 4: 论文撰写 (Week 13-16)                                 │
│  ├─ Week 13: Introduction, Related Work                        │
│  ├─ Week 14: Method                                            │
│  ├─ Week 15: Experiments, Results                              │
│  ├─ Week 16: Analysis, Conclusion, 修改                        │
│  └─ 交付物: 论文 + 代码                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 里程碑与验收标准

| 周次 | 里程碑 | 验收标准 |
|------|--------|----------|
| Week 3 | 环境完成 | 双桌 IMP 可计算；IPPO 可训练 |
| Week 5 | Belief 完成 | Belief Accuracy > 0.7 |
| Week 7 | 算法完成 | DualInfo 收敛，IMP > MAPPO |
| Week 9 | RQ1 完成 | Baseline 对比表格就绪 |
| Week 10 | RQ2 完成 | 机制验证图表就绪 |
| Week 12 | 全部实验完成 | 消融 + Trade-off + Benchmark 就绪 |
| Week 16 | 论文完成 | 可提交 |

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| **Info Bonus 漂移** | 中 | 高 | Lambda 退火 + correlation 监控 |
| **Belief 不准** | 中 | 高 | 分阶段训练 + 预热 |
| **COMA 方差大** | 中 | 中 | 主要使用 MAPPO |
| **β 量级不匹配** | 中 | 中 | 归一化 + 分布可视化 |
| **β 影响不显著** | 中 | 低 | 重新定位：Partner Info 是核心贡献；报告"先验不对称性已足够"作为科学发现 |
| 收敛慢 | 中 | 中 | 简化网络 / 增加 batch size |
| DDS 计算慢 | 低 | 中 | 缓存 + 并行 |

---

## 7. 预期贡献

### 7.1 学术贡献

1. **Dual-Information Credit Assignment**：首个在合作-对抗混合博弈中显式利用先验不对称性进行信息论 reward shaping 的方法

2. **History Process 视角的应用**：将 continual RL 中的 history process 形式化应用于多智能体隐式通信场景

3. **机制验证框架**：提供测量和可视化先验不对称效应的工具

4. **消融分析**：量化 partner info term 与 opponent penalty term 的相对贡献，回答"opponent penalty 是否必要"


