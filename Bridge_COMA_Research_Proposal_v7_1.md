# Bridge-COMA: Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination

## 研究方案 v7.1

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
| **时间信用分配** | 12轮叫牌中，哪一步埋下了祸根？ | 部分通过 $r_{\text{info}}$ 缓解 |

**探索坍塌风险**：纯强化学习在稀疏奖励下容易陷入"全 Pass"的局部最优——因为瞎叫的期望是负数（被罚分），而 Pass 的期望最多是漏局。本方案通过**行为克隆预热**和**动作掩码**解决此问题（见 3.4 节）。

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

### 2.3 核心创新：Dual-Information Credit Assignment

在 MAPPO/COMA 的基础上，增加信息论 reward shaping：

$$r_{\text{info}} = \underbrace{I(\text{bid}; \text{hand} \mid \text{partner})}_{\text{partner's inference gain}} - \beta \cdot \underbrace{I(\text{bid}; \text{hand} \mid \text{opponent})}_{\text{opponent's inference gain}}$$

其中 $\beta \geq 0$ 作为**拉格朗日乘子**，控制通信清晰度与信息隐蔽性之间的权衡：
- **低 β**：优先清晰通信，即使对手也能获得信息，协调收益大于防守风险
- **高 β**：偏好"高效"或"紧凑"的叫牌路径，鼓励用最少的信息交换达到最优定约

**$r_{\text{info}}$ 的双重作用**：
1. **鼓励信息性通信**：显式奖励对队友有帮助的叫牌
2. **缓解时间信用分配**：提供每一步的 dense reward，而非仅依赖最终的稀疏 IMP

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
- **奖励**：IMP（双桌对比，只在叫牌结束后给出）+ $r_{\text{info}}$（每步）

### 3.2 双桌 IMP 计算

```
┌─────────────────────────────────────────────────────────────────┐
│                        双桌 IMP 计算                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  同一副牌 Deal                                                   │
│                                                                 │
│  桌 1 (Open Room):                                              │
│  ┌─────────────────────────────────────┐                       │
│  │  N-S: Agent    vs    E-W: Agent     │ → Score_1             │
│  └─────────────────────────────────────┘                       │
│                                                                 │
│  桌 2 (Closed Room):                                            │
│  ┌─────────────────────────────────────┐                       │
│  │  N-S: Agent    vs    E-W: Agent     │ → Score_2             │
│  │  (位置互换，同一副牌)                 │                       │
│  └─────────────────────────────────────┘                       │
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
    
    def get_action(self, hand, history, position):
        """带动作掩码的动作选择"""
        logits = self.policy_net(hand, history)
        mask = get_action_mask(hand, history, position)
        
        # 掩码后的 softmax
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(masked_logits, dim=-1)
        
        return Categorical(probs).sample()
    
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

#### 3.3.2 动作掩码（防止探索坍塌）

```python
def get_action_mask(hand, history, position):
    """
    返回合法且"不傻"的动作掩码
    
    Level 1: 硬性规则（必须）
    - 叫品必须高于当前最高叫品
    - 不能在同伴加倍后再加倍
    
    Level 2: 软性规则（加速收敛）
    - 极弱牌（<5 HCP）不开叫
    - 没有长套不跳叫高阶
    """
    mask = torch.ones(38)  # 38种叫品
    
    # Level 1: 合法性掩码
    current_level = get_current_level(history)
    for bid_idx in range(38):
        if not is_legal_bid(bid_idx, history):
            mask[bid_idx] = 0
    
    # Level 2: 合理性掩码
    hcp = count_hcp(hand)
    if is_opening_position(history):
        if hcp < 5:  # 极弱牌不开叫（但保留 Pass）
            for bid_idx in range(1, 38):
                mask[bid_idx] = 0
    
    return mask
```

#### 3.3.3 归一化的 Dual-Info Bonus

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

### 3.4 训练流程：四阶段训练（新增）

```python
def train_dual_info_agent(config):
    """
    四阶段训练流程
    
    Phase 0: 行为克隆预热（新增）—— 防止探索坍塌
    Phase 1: Belief 预热 —— 确保信息增益计算准确
    Phase 2: 子博弈验证（新增）—— 在受控环境验证假设
    Phase 3: 联合训练 —— 完整叫牌训练
    """
    agent = DualInfoAgent(config)
    
    # ==================== Phase 0: 行为克隆预热 ====================
    print("=" * 60)
    print("Phase 0: Behavioral Cloning Warmup")
    print("=" * 60)
    
    bc_data = create_base_policy_data(config.bc_samples)
    behavioral_cloning_warmup(agent, bc_data, epochs=config.bc_epochs)
    
    # 验证：确保不再是"全 Pass"
    pass_rate = evaluate_pass_rate(agent)
    print(f"Post-BC Pass Rate: {pass_rate:.2%}")
    assert pass_rate < 0.5, f"BC failed: pass_rate = {pass_rate}"
    
    # ==================== Phase 1: Belief 预热 ====================
    print("=" * 60)
    print("Phase 1: Belief Network Warmup")
    print("=" * 60)
    
    env = DualTableEnv(agent)
    
    for step in range(config.belief_warmup_steps):
        batch = collect_rollouts(env, config.batch_size)
        
        # 只更新 Belief Network
        belief_loss = compute_belief_loss(agent.belief_net, batch)
        agent.belief_optimizer.step(belief_loss)
        
        if step % config.log_interval == 0:
            belief_acc = evaluate_belief_accuracy(agent)
            print(f"Step {step}: Belief Accuracy = {belief_acc:.3f}")
            
            if belief_acc > config.belief_threshold:
                print(f"Belief accuracy reached {config.belief_threshold}")
                break
    
    # ==================== Phase 2: 子博弈验证 ====================
    print("=" * 60)
    print("Phase 2: Subgame Validation")
    print("=" * 60)
    
    # 2a: Stayman 子博弈（纯合作）
    stayman_results = train_subgame(
        agent, 
        StaymanSubgameEnv(),
        steps=config.subgame_steps,
        name="Stayman"
    )
    print(f"Stayman Results: Belief Acc = {stayman_results['belief_acc']:.3f}, "
          f"Info Ratio = {stayman_results['info_ratio']:.3f}")
    
    # 验证核心假设
    assert stayman_results['info_ratio'] > 1.0, \
        "Core hypothesis failed: Info Ratio <= 1 in cooperative subgame"
    
    # 2b: 竞叫子博弈（合作-对抗）
    competitive_results = train_subgame(
        agent,
        CompetitiveSubgameEnv(),
        steps=config.subgame_steps,
        name="Competitive"
    )
    print(f"Competitive Results: Info Ratio = {competitive_results['info_ratio']:.3f}")
    
    # ==================== Phase 3: 完整叫牌训练 ====================
    print("=" * 60)
    print("Phase 3: Full Bidding Training")
    print("=" * 60)
    
    full_env = DualTableEnv(agent)
    
    for step in range(config.joint_training_steps):
        batch = collect_rollouts(full_env, config.batch_size)
        
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
            metrics = evaluate_agent(agent, full_env)
            
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
                
                # 时间信用分配监控（新增）
                'info_imp_correlation': metrics['info_imp_correlation'],
                'value_error_early': metrics['value_error_early'],
                'value_error_late': metrics['value_error_late'],
                
                # 性能
                'imp_self_play': metrics['imp'],
                'avg_bidding_rounds': metrics['avg_rounds'],
                'pass_rate': metrics['pass_rate'],
            })
```

### 3.5 行为克隆数据生成（新增）

**用途**：仅用于 Competitive 子博弈（Stayman 不需要 BC 预热，因为固定前缀后动作空间小、探索不会坍塌）。

**均型定义**：无单缺（所有花色 ≥ 2 张），允许 5 张高花或 6 张低花（如 5332, 6322 等）。

```python
def create_base_policy_data(num_samples=100000):
    """
    生成规则驱动的"合理叫牌"数据
    
    注意：不需要是最优的，只需要是"不傻的"
    目标：防止探索坍塌，不是编码完整叫牌体系
    仅用于 Competitive 子博弈的 BC 预热
    """
    data = []
    
    for _ in range(num_samples):
        deal = generate_random_deal()
        
        for position in ['N', 'E', 'S', 'W']:
            hand = deal.hands[position]
            history = deal.get_history_at(position)
            
            hcp = count_hcp(hand)
            
            if is_opening_position(history):
                if hcp >= 12:
                    bid = select_simple_opening(hand)
                else:
                    bid = "Pass"
            else:
                bid = simple_response_logic(hand, history)
            
            data.append({
                'hand': encode_hand(hand),
                'history': encode_history(history),
                'bid': bid_to_index(bid)
            })
    
    return data

def select_simple_opening(hand):
    """简单开叫规则（不需要完整体系）"""
    hcp = count_hcp(hand)
    
    # 超强牌
    if hcp >= 22:
        return "2C"
    
    # 均型 15-17（无单缺，允许 5M/6m）
    if is_balanced(hand) and 15 <= hcp <= 17:
        return "1NT"
    
    # 有 5+ 张高花
    if len(hand.spades) >= 5:
        return "1S"
    if len(hand.hearts) >= 5:
        return "1H"
    
    # 有 4+ 张低花
    if len(hand.diamonds) >= len(hand.clubs):
        return "1D"
    return "1C"

def competitive_response_rules(hand, history):
    """
    竞叫应叫规则（1H-1S-? 进程）
    
    启发式规则，简单偏保守，防止 BC 让 agent 太"完美"
    
    进程 1H - 1S - ?（South 应叫）:
      - X (负加倍):    11+ HCP, 0-2 张 H
      - 1NT:           8-10 HCP, 3 张 H
      - 2C / 2D:       8-11 HCP, 5+ 张 C/D
      - 2H:            5-7 HCP, 3 张 H
      - 2S:            11+ HCP, 3 张 H（强争叫）
      - 2NT:           11+ HCP, 4 张 H
      - 3H:            12+ HCP, 4 张 H（限制性加叫）
      - 3C / 3D:       12+ HCP, 6+ 张 C/D（强争叫）
      - 3NT:           12-15 HCP, 均型, S 有止张
      - Pass:          0-7 HCP, < 3H, 无 5+ 低花
    
    进程 1H - 1S - (应叫人叫品低于 2S) - ?（后续 N 续叫）:
      - X (负加倍):    11+ HCP, 0-2 张 S
      - 2S:            6-10 HCP, 3 张 S
      - 2NT:           11+ HCP, 4 张 S
      - 3S:            6-10 HCP, 4 张 S
    """
    # 实现上述规则
    ...
```

### 3.6 子博弈环境（新增）

#### 3.6.1 Stayman 子博弈（纯合作验证）

```python
class StaymanSubgameEnv:
    """
    固定前缀：1NT - Pass - 2C - Pass
    
    学习目标：后续叫牌如何最优地找到 4-4 高花配合
    
    用途：在无竞叫的纯合作场景下验证 r_info 的 partner term
    
    特点：
    - 纯合作：EW 全部自动 Pass，只有 NS 需要决策
    - 无 BC 预热：固定前缀后动作空间小（~6 种合理选择），不会探索坍塌
    - 单桌评估：actual score vs DDS optimal score → IMP
    - 训练 2 个 agent 对比：MAPPO (control) vs MAPPO + r_info (β=0)
    """
    
    def __init__(self, data_path):
        self.fixed_prefix = ["1NT", "Pass", "2C", "Pass"]
        self.loader = create_loader(data_path)
        
        # 开叫人约束：均型（无单缺，允许 5M/6m），15-17 HCP
        self.opener_constraints = {
            'hcp': (15, 17),
            'shape': 'balanced'  # 无单缺
        }
        
        # 应叫人约束：8+ HCP，有 4 张高花
        self.responder_constraints = {
            'hcp': (8, None),
            'has_4card_major': True
        }
    
    def generate_deal(self):
        """只生成符合约束的牌"""
        while True:
            deal = random_deal()
            north = deal.hands['N']
            south = deal.hands['S']
            
            if (self.satisfies_opener_constraints(north) and
                self.satisfies_responder_constraints(south)):
                return deal
    
    def step(self, action):
        """NS 决策，EW 自动 Pass"""
        ...
    
    def get_action_mask(self, hand, history):
        """Stayman 后续的合理选择"""
        rounds_after_2c = len(history) - 4
        
        if rounds_after_2c == 0:  # 开叫人回应 2C
            return ["2D", "2H", "2S"]  # 无高花 / 有红心 / 有黑桃
        elif rounds_after_2c == 2:  # 应叫人再叫
            return ["2NT", "3NT", "3H", "3S", "4H", "4S"]
        # ...
    
    def compute_reward(self, contract, deal, dd_table):
        """
        单桌评估：actual score vs DDS optimal → IMP
        
        理想：4-4 配合时找到 4M，否则打 3NT
        """
        optimal = self.get_optimal_contract(deal, dd_table)
        actual_score = dds_score(deal, contract)
        optimal_score = dds_score(deal, optimal)
        
        return score_to_imp(actual_score - optimal_score)
```

#### 3.6.2 竞叫子博弈（合作-对抗验证）

```python
class CompetitiveSubgameEnv:
    """
    固定前缀：1H - 1S
    
    学习目标：在激烈竞叫中协调配合，同时防止对手找到配合
    
    用途：验证 r_info 的 opponent penalty term (β)
    
    特点：
    - 合作-对抗：四方都参与决策，竞争激烈
    - 1S 争叫（非跳叫 2S）：双方牌力接近，竞争更充分
    - 需要 BC 预热：动作空间大，需要防止探索坍塌
    - 双桌 IMP 交叉对抗评估：训练 3 个 agent，6 组交叉比较
    - 训练 3 个 agent:
        A: MAPPO (control)
        B: MAPPO + r_info (β=0, partner-only)
        C: MAPPO + r_info (β=0.5, dual-info)
    """
    
    def __init__(self, data_path):
        self.fixed_prefix = ["1H", "1S"]
        self.loader = create_loader(data_path)
        
        # 开叫人（N）：5+ 红心，12-21 HCP
        self.opener_constraints = {
            'hearts': (5, None),
            'hcp': (12, 21)
        }
        
        # 争叫人（E）：5+ 黑桃，8-16 HCP（1-level 争叫，非阻叫）
        self.overcaller_constraints = {
            'spades': (5, None),
            'hcp': (8, 16)
        }
    
    def step(self, action):
        """四方都参与决策"""
        ...
    
    def play_mixed(self, deal, ns_agent, ew_agent):
        """
        混合对抗：NS 和 EW 由不同 agent 控制
        用于交叉对抗评估
        """
        history = self.fixed_prefix.copy()
        
        while not bidding_complete(history):
            position = get_next_position(history)
            
            if position in ['N', 'S']:
                bid = ns_agent.get_action(deal, history, position)
            else:
                bid = ew_agent.get_action(deal, history, position)
            
            history.append(bid)
        
        return get_contract(history), history
    
    def play_mixed_with_metrics(self, deal, ns_agent, ew_agent):
        """
        混合对抗 + 记录详细信息论指标
        
        记录每方的 partner_info, opponent_leak, info_ratio
        用于验证 β 的效果
        """
        history, contract = self.play_mixed(deal, ns_agent, ew_agent)
        
        metrics = {
            # N-S 的协调质量
            'ns_partner_info': compute_info_gain('S', 'N', history, ns_agent.belief_net),
            'ns_opponent_leak': compute_info_gain('E', 'N', history, ew_agent.belief_net),
            
            # E-W 的协调质量
            'ew_partner_info': compute_info_gain('W', 'E', history, ew_agent.belief_net),
            'ew_opponent_leak': compute_info_gain('S', 'E', history, ns_agent.belief_net),
        }
        
        metrics['ns_info_ratio'] = metrics['ns_partner_info'] / (metrics['ns_opponent_leak'] + 1e-8)
        metrics['ew_info_ratio'] = metrics['ew_partner_info'] / (metrics['ew_opponent_leak'] + 1e-8)
        
        return history, contract, metrics

def cross_evaluate(agent_a, agent_b, env, num_deals):
    """
    交叉对抗评估（双桌 IMP）
    
    桌 1: A 打 N-S, B 打 E-W → score_1
    桌 2: B 打 N-S, A 打 E-W → score_2
    IMP = score_to_imp(score_1 - score_2)  (从 A 视角)
    """
    imps = []
    all_metrics = []
    
    for _ in range(num_deals):
        deal = env.generate_deal()
        
        # 桌 1: A=NS, B=EW
        history_1, contract_1, metrics_1 = env.play_mixed_with_metrics(
            deal, ns_agent=agent_a, ew_agent=agent_b
        )
        score_1 = dds_score(deal, contract_1)
        
        # 桌 2: B=NS, A=EW
        history_2, contract_2, metrics_2 = env.play_mixed_with_metrics(
            deal, ns_agent=agent_b, ew_agent=agent_a
        )
        score_2 = dds_score(deal, contract_2)
        
        imp = score_to_imp(score_1 - score_2)
        imps.append(imp)
        all_metrics.append({'table1': metrics_1, 'table2': metrics_2})
    
    return {
        'mean_imp': np.mean(imps),
        'std_imp': np.std(imps),
        'win_rate': np.mean([imp > 0 for imp in imps]),
        'significant': ttest_1samp(imps, 0).pvalue < 0.05,
        'p_value': ttest_1samp(imps, 0).pvalue,
        'metrics': all_metrics,
    }
```

### 3.7 网络架构

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

### 3.8 限制说明

1. **DDS 假设**：本项目使用 Double Dummy Solver (DDS) 评估定约得分，假设完美打牌。信息泄露对防守决策的影响（如首攻选择）无法直接测量。

2. **时间信用分配**：虽然 $r_{\text{info}}$ 提供了 dense reward 信号，但完美归因到每一步叫牌仍然困难。Value Network 在早期步骤的估计误差可能较大。

3. **外星语言风险**：从零学习可能收敛到人类难以理解的协议。本方案通过 BC 预热和语义监控缓解此问题，但不能完全消除。

---

## 4. 实验设计

### 4.1 阶段性验证计划（v7.1 修订）

```
┌─────────────────────────────────────────────────────────────────┐
│                    阶段性验证计划                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段 0: 环境与基础设施 (Phase 1 已完成 ✅)                     │
│  ├─ DDS 可以正确计算分数                                       │
│  ├─ Belief Network 架构可以学习                                │
│  ├─ IMP 双桌计算正确                                           │
│  └─ 35 项单元测试全部通过                                      │
│                                                                 │
│  阶段 1: Stayman 子博弈 (3-5 天)                                 │
│  ├─ 纯合作场景，EW 全部自动 Pass                                │
│  ├─ 无 BC 预热（固定前缀后动作空间小）                          │
│  ├─ 训练 2 个 agent: MAPPO vs MAPPO+r_info(β=0)                │
│  ├─ 单桌评估：actual vs DDS optimal → IMP                      │
│  ├─ 验收标准：Belief Accuracy > 0.8, Info Ratio > 1.0          │
│  └─ Go/No-Go: Info Ratio > 1.0?                                │
│                                                                 │
│  阶段 2: 竞叫子博弈 (5-7 天)                                    │
│  ├─ 1H - 1S 进程（激烈竞争，双方牌力接近）                      │
│  ├─ BC 预热（用竞叫应叫规则防止探索坍塌）                       │
│  ├─ 训练 3 个 agent:                                            │
│  │   ├─ A: MAPPO (control)                                      │
│  │   ├─ B: MAPPO + r_info (β=0, partner-only)                  │
│  │   └─ C: MAPPO + r_info (β=0.5, dual-info)                   │
│  ├─ 双桌 IMP 交叉对抗评估（6 组比较）                           │
│  ├─ 验收标准：B vs A > 0 IMP (p < 0.05)                        │
│  └─ Go/No-Go: β 有效果?                                        │
│                                                                 │
│  Go/No-Go 决策点：                                               │
│  ├─ 阶段 1 后：如果 Info Ratio < 1.0，重新审视核心假设           │
│  └─ 阶段 2 后：如果 β 无效果，调整为"Partner-Only"方向           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.1 Competitive 子博弈实验矩阵

```
┌─────────────────────────────────────────────────────────────┐
│  子博弈验证实验矩阵                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练 3 个 Agent（各自 self-play）:                          │
│  ├─ A: MAPPO (Control)                                      │
│  ├─ B: MAPPO + r_info (β=0, partner-only)                  │
│  └─ C: MAPPO + r_info (β=0.5, dual-info)                   │
│                                                             │
│  交叉对抗（双桌 IMP，3 组关键比较）:                          │
│  ┌─────────────┬─────────────┬─────────────┐               │
│  │  对抗       │ 预期 IMP    │ 验证假设    │               │
│  ├─────────────┼─────────────┼─────────────┤               │
│  │  B vs A     │  > 0        │ r_info 有效 │               │
│  │  C vs A     │  > 0        │ r_info 有效 │               │
│  │  C vs B     │  ≥ 0        │ β 有价值    │               │
│  └─────────────┴─────────────┴─────────────┘               │
│                                                             │
│  预期结果：                                                  │
│  ┌──────────┬─────────────┬─────────┬────────────────┐     │
│  │  对抗    │ IMP (A视角) │ p-value │ 结论           │     │
│  ├──────────┼─────────────┼─────────┼────────────────┤     │
│  │  B vs A  │ +0.3~0.5    │ < 0.05  │ r_info 有效 ✓  │     │
│  │  C vs A  │ +0.4~0.6    │ < 0.05  │ r_info 有效 ✓  │     │
│  │  C vs B  │ +0.1~0.2    │ < 0.10  │ β 有边际价值   │     │
│  └──────────┴─────────────┴─────────┴────────────────┘     │
│                                                             │
│  信息论指标（每组对抗同时记录）:                              │
│  ├─ ns_partner_info / ns_opponent_leak → ns_info_ratio      │
│  └─ ew_partner_info / ew_opponent_leak → ew_info_ratio      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1.2 Fallback 策略

| 情况 | 动作 |
|------|------|
| Stayman Info Ratio < 1.0 | 重新审视 Belief Net 架构或训练方式 |
| B vs A 不显著 | 检查 r_info 量级、退火速度、Belief 质量 |
| C vs B 无差异 | 调整为 Partner-Only (β=0)，论文叙事改写 |
| 全部失败 | 回退到纯 MAPPO baseline，focus on BC + Action Mask 的贡献 |

### 4.2 Baseline 对比（对应 RQ1）

| 方法 | 描述 | 初始化 |
|------|------|--------|
| **MAPPO + Dual-Info (Ours)** | MAPPO + 归一化 Dual-Info Bonus + 退火 | BC + Mask |
| MAPPO | 标准 MAPPO，无 info bonus | BC + Mask |
| COMA | 标准 COMA，counterfactual baseline | BC + Mask |
| IPPO | Independent PPO，无协调机制 | BC + Mask |

**关键控制**：所有 baseline 使用**相同的 BC 预热和 Action Mask**，确保差异只来自 $r_{\text{info}}$。

**评估指标**：
- 双桌 Self-play IMP
- 收敛速度（达到目标 IMP 所需步数）
- Info Ratio

**预期结果**：

| 方法 | 预期 IMP 差距 |
|------|--------------|
| MAPPO + Dual-Info | 基准 |
| MAPPO | -0.3 ~ -0.5 |
| COMA | -0.2 ~ -0.4 |
| IPPO | -0.8 ~ -1.0 |

### 4.3 机制验证（对应 RQ2）

**目标**：验证 Dual-Info Bonus 确实产生了预期的先验不对称效应。

**指标**：

| 指标 | 定义 | 预期 |
|------|------|------|
| Partner Belief Accuracy | 队友预测我手牌的准确率 | 较高 |
| Opponent Belief Accuracy | 对手预测我手牌的准确率 | 较低 |
| Info Ratio | Partner Acc / Opponent Acc | > 1.2 |
| Info-IMP Correlation | $r_{\text{info}}$ 与最终 IMP 的相关性 | > 0.3 |

**可视化**：

```python
def visualize_mechanism(agent, num_samples=10000):
    """机制验证可视化"""
    # 1. r_info 分布
    # 2. Partner vs Opponent 准确率散点图
    # 3. Info Ratio 分布
    # 4. Info Ratio vs IMP 相关性
    # 5. Value Error by Timestep（时间信用分配诊断）
```

### 4.4 消融实验（对应 RQ3）

**目标**：量化 opponent penalty term (β) 的贡献。

**实验设置**：

| 配置 | β 值 | 含义 |
|------|------|------|
| Partner-Only | 0.0 | 只有 partner info gain |
| Low Penalty | 0.25 | 轻微 opponent penalty |
| Balanced | 0.5 | 平衡点 |
| High Penalty | 0.75 | 较强 opponent penalty |
| Max Penalty | 1.0 | 最大 opponent penalty |

**预期结果与解释**：

| 发现 | 结论 |
|------|------|
| β=0 和 β=0.5 的 IMP 差异不显著 | Opponent penalty 非必要，先验不对称性已足够 |
| β=0.5 显著优于 β=0 | Opponent penalty 有额外价值 |
| 高 β 导致 IMP 下降 | 过度压缩通信损害协调质量 |

### 4.5 健康监控指标

```python
MONITORING_CONFIG = {
    # 训练健康度
    'pass_rate': {'warning': 0.4, 'critical': 0.6},
    'policy_entropy': {'warning_low': 0.5, 'warning_high': 3.0},
    
    # Belief 质量
    'belief_accuracy': {'warning': 0.6, 'target': 0.75},
    
    # 核心假设验证
    'info_ratio': {'warning': 1.0, 'target': 1.2},
    'info_imp_correlation': {'warning': 0.1, 'target': 0.3},
    
    # 时间信用分配诊断
    'value_error_ratio': {'warning': 2.0},  # early/late error ratio
}
```

---

## 5. 计划与里程碑

### 5.1 资源约束

| 资源 | 可用量 | 限制 |
|------|--------|------|
| Colab Pro GPU | ~24h/session | 需要 checkpoint 机制 |
| 总时间 | 16 周 | MSc 项目周期 |

### 5.2 修订后的时间规划

```
┌─────────────────────────────────────────────────────────────────┐
│                      时间规划 (16 周)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 环境构建 (Week 1-3) ✅ 已完成                         │
│  ├─ 核心环境（发牌、叫牌、DDS）                                  │
│  ├─ 双桌 IMP + 得分计算 + IMP 转换                              │
│  ├─ IPPO / MAPPO 算法 + 训练脚本                                │
│  ├─ 35 项测试覆盖全模块                                         │
│  └─ 交付物: 可运行的训练环境 ✅                                  │
│                                                                 │
│  Phase 2: 子博弈验证 (Week 4-5) ← 当前阶段                      │
│  ├─ Week 4: Stayman 子博弈                                      │
│  │   ├─ 纯合作，EW 自动 Pass，无 BC                             │
│  │   ├─ 单桌评估 (actual vs DDS optimal → IMP)                  │
│  │   └─ Go/No-Go: Info Ratio > 1.0?                            │
│  ├─ Week 5: 竞叫子博弈 (1H-1S)                                  │
│  │   ├─ BC 预热 + 3 agent 训练 (A/B/C)                         │
│  │   ├─ 双桌 IMP 交叉对抗评估                                   │
│  │   └─ Go/No-Go: B vs A 显著? β 有效果?                       │
│  └─ 交付物: 子博弈结果 + 超参数 + Go/No-Go 报告                 │
│                                                                 │
│  Phase 3: Belief + DualInfo (Week 6-7)                          │
│  ├─ Week 6: Belief Network 完整实现                             │
│  ├─ Week 7: Dual-Info Bonus 集成                                │
│  └─ 交付物: 完整 DualInfo Agent                                 │
│                                                                 │
│  Phase 4: 完整训练与实验 (Week 8-12)                             │
│  ├─ Week 8-9: Baseline 对比 (RQ1)                               │
│  ├─ Week 10: 机制验证 + 可视化 (RQ2)                            │
│  ├─ Week 11: 消融实验 β ∈ [0,1] (RQ3)                           │
│  ├─ Week 12: Benchmark + Trade-off 分析                         │
│  └─ 交付物: 完整实验结果                                         │
│                                                                 │
│  Phase 5: 论文撰写 (Week 13-16)                                  │
│  ├─ Week 13: Introduction, Related Work                         │
│  ├─ Week 14: Method                                             │
│  ├─ Week 15: Experiments, Results                               │
│  ├─ Week 16: Analysis, Conclusion, 修改                         │
│  └─ 交付物: 论文 + 代码                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 里程碑与验收标准

| 周次 | 里程碑 | 验收标准 |
|------|--------|----------|
| Week 3 | 环境完成 ✅ | 双桌 IMP 可计算；35 项测试通过 |
| Week 4 | Stayman 验证 | Info Ratio > 1.0；r_info agent 单桌 IMP > control（Go/No-Go） |
| Week 5 | 竞叫验证 | B vs A > 0 IMP, p < 0.05；β 效果可观测（Go/No-Go） |
| Week 7 | 算法完成 | DualInfo 在子博弈中优于 MAPPO |
| Week 9 | RQ1 完成 | 完整叫牌 Baseline 对比表格就绪 |
| Week 10 | RQ2 完成 | 机制验证图表就绪 |
| Week 12 | 全部实验完成 | 消融 + Trade-off 就绪 |
| Week 16 | 论文完成 | 可提交 |

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| **探索坍塌（全 Pass）** | 高 | 致命 | BC 预热 + Action Mask（已在方案中） |
| **Info Bonus 与 IMP 不相关** | 中 | 高 | 监控 correlation；如 <0.1 则重新设计 $r_{\text{info}}$ |
| **子博弈验证失败** | 中 | 高 | 提前暴露问题；调整核心假设或方向 |
| **Belief 不准** | 中 | 高 | 分阶段训练 + 预热 |
| **β 量级不匹配** | 中 | 中 | 归一化 + 分布可视化 |
| **β 影响不显著** | 中 | 低 | 重新定位：Partner Info 是核心贡献 |
| **外星语言** | 中 | 中 | BC 初始化 + 语义监控 |
| **Colab 断连** | 高 | 中 | Checkpoint 机制 + Google Drive |
| 收敛慢 | 中 | 中 | 简化网络 / 增加 batch size |
| DDS 计算慢 | 低 | 中 | 缓存 + 并行 |

---

## 7. 预期贡献

### 7.1 学术贡献

1. **Dual-Information Credit Assignment**：首个在合作-对抗混合博弈中显式利用先验不对称性进行信息论 reward shaping 的方法

2. **子博弈验证框架**：提供从受控子博弈到完整博弈的渐进式验证方法论

3. **机制验证框架**：提供测量和可视化先验不对称效应的工具

4. **消融分析**：量化 partner info term 与 opponent penalty term 的相对贡献

### 7.2 预期结果

| 指标 | 预期 |
|------|------|
| vs MAPPO | +0.3~0.5 IMP |
| vs IPPO | +0.8~1.0 IMP |
| Info Ratio | > 1.2 |
| Info-IMP Correlation | > 0.3 |

---

# 第二部分：工程扩展（项目外）

---

## 8. 工程目标

在科研项目完成后，利用已有代码框架，构建**最强性能的桥牌 AI** 和**实用的叫牌指导工具**。

### 8.1 与科研的关系

| 维度 | 科研项目 | 工程扩展 |
|------|----------|----------|
| 目标 | 学术贡献 | 最强性能 + 实用工具 |
| 约束 | 公平比较，统一 baseline | 无约束 |
| DSL | ❌ 不使用 | ✅ 精细 5542 |
| 训练规模 | 适中 (Colab Pro) | 大规模 |
| 搜索 | ❌ 不使用 | ✅ MCTS |
| 评估 | DDS (单次结果) | Monte Carlo (期望值) |

### 8.2 性能提升手段

| 手段 | 描述 | 预期收益 |
|------|------|----------|
| 精细 DSL 初始化 | 5542 体系 (~200 规则) | +0.3~0.5 IMP |
| Vugraph SL 预训练 | 顶级比赛记录 | +0.2~0.3 IMP |
| 大规模训练 | 10x 训练步数 | +0.2 IMP |
| Test-time MCTS | 推理时搜索 | +0.3~0.5 IMP |
| League Training | 多 Agent 联赛 | 更鲁棒 |

### 8.3 应用系统

#### Bridge Advisor

```
用户输入: 手牌 + 叫牌进程
           ↓
┌─────────────────────────────┐
│  Trained Agent (强化版)     │
│  + 蒙特卡洛模拟             │
│  + 解释生成                 │
└─────────────────────────────┘
           ↓
输出: 推荐叫品 + 概率 + 期望IMP + 理由
```

---

## 9. 代码架构

### 9.1 当前实际结构（Phase 1 完成后）

源文件以扁平方式存放，运行 `setup_project.py` 自动组装为包结构：

```
bridge-coma/
├── env/                          # 环境
│   ├── bridge_bidding_env.py     # 单桌叫牌环境 (Dec-POMDP)
│   └── dual_table_env.py         # 双桌 IMP 环境
├── networks/                     # 神经网络
│   ├── policy_net.py             # PolicyNetwork, ValueNetwork, ActorCritic
│   └── belief_net.py             # BeliefNetwork, DualInfoComputer
├── utils/                        # 工具
│   ├── scoring.py                # 得分计算 (SSOT)
│   ├── imp.py                    # IMP 转换
│   ├── dds_data.py               # DDS 数据生成/加载
│   └── running_stats.py          # Welford / EMA 在线统计
├── algorithms/                   # 算法
│   ├── ippo.py                   # Independent PPO
│   └── mappo.py                  # Multi-Agent PPO (CTDE)
├── experiments/
│   └── train.py                  # 训练入口
├── tests/
│   └── test_all.py               # 35 项测试
├── setup_project.py
└── requirements.txt
```

### 9.2 Phase 2 新增文件

```
bridge-coma/
├── ... (Phase 1 已有)
│
├── subgames/                     # 【Phase 2 新增】
│   ├── stayman_env.py            # Stayman 子博弈环境（纯合作，单桌评估）
│   ├── competitive_env.py        # 竞叫子博弈环境（1H-1S，双桌交叉对抗）
│   ├── subgame_trainer.py        # 通用子博弈训练器
│   └── action_mask.py            # 动作掩码（合法性 + 合理性）
│
├── algorithms/
│   ├── ... (已有)
│   └── behavioral_cloning.py     # 【Phase 2 新增】BC 预热（仅 Competitive 使用）
│
├── experiments/
│   ├── train.py                  # (已有)
│   └── subgame_validation.py     # 【Phase 2 新增】子博弈验证实验入口
│
├── tests/
│   ├── test_all.py               # (已有)
│   └── test_phase2.py            # 【Phase 2 新增】
│
└── results/                      # 【Phase 2 新增】
    └── phase2_report.json        # 子博弈验证报告
```

### 9.3 未来完整目标架构

```
bridge-coma/
│
├── core/                        # 【共用】
│   ├── env/
│   ├── networks/
│   └── utils/
│
├── algorithms/                  # 【共用】
│   ├── dual_info_agent.py
│   ├── mappo.py
│   └── behavioral_cloning.py
│
├── subgames/                    # 【子博弈验证】
│   ├── stayman_env.py
│   ├── competitive_env.py
│   └── subgame_trainer.py
│
├── research/                    # 【科研专用】
│   ├── experiments/
│   │   ├── subgame_validation.py
│   │   ├── baseline_comparison.py
│   │   ├── mechanism_verification.py
│   │   └── ablation_beta.py
│   ├── analysis/
│   │   ├── info_ratio_analysis.py
│   │   ├── temporal_credit_analysis.py
│   │   └── visualizations.py
│   └── paper/
│
├── engineering/                 # 【工程专用】
│   ├── dsl/
│   ├── search/
│   └── large_scale_training/
│
└── applications/                # 【应用】
    ├── advisor/
    └── web/
```

---

## 10. 总结

### 科研核心 (MSc)

| 维度 | 内容 |
|------|------|
| 核心创新 | Dual-Information Credit Assignment |
| 理论基础 | Wiretap Channel, History Process |
| 关键技术 | BC 预热 + Action Mask + 归一化 + 退火 + 分阶段训练 |
| 验证策略 | 子博弈验证 → 完整叫牌（渐进式） |
| 研究问题 | 有效性 → 机制验证 → 消融分析 |
| 评估方法 | DDS + 双桌 IMP（Competitive）/ 单桌 vs DDS（Stayman） |
| 时间 | 16 周 |

### v7.0 → v7.1 主要更新

| 更新项 | 描述 | 原因 |
|--------|------|------|
| **竞叫前缀改为 1H-1S** | 争叫（非跳叫阻叫 2S） | 竞争更激烈，双方牌力接近，更能发挥 r_info 优势 |
| **Stayman 去除 BC** | 纯合作场景不需要 BC 预热 | 固定前缀后动作空间小（~6 种），不会探索坍塌 |
| **3-Agent 实验矩阵** | A(control)/B(β=0)/C(β=0.5) 交叉对抗 | 更严格的消融设计，同时验证 r_info 和 β |
| **双桌交叉对抗评估** | Competitive 用双桌 IMP 交叉比较 | 消除单桌偶然性，标准化评估 |
| **单桌评估 Stayman** | actual vs DDS optimal → IMP | 纯合作场景有明确最优解 |
| **详细 BC 规则** | 1H-1S 竞叫应叫规则 | 基于桥牌领域知识的启发式规则 |
| **均型定义修正** | 无单缺（允许 5M/6m） | 更符合实际桥牌约定 |
| **play_mixed_with_metrics** | 交叉对抗时记录信息论指标 | 精确测量 partner_info / opponent_leak / info_ratio |
| **Phase 1 标记完成** | 环境与基础设施 ✅ | 35 项测试通过，训练脚本可运行 |

### v6.3 → v7.0 主要更新

| 更新项 | 描述 | 原因 |
|--------|------|------|
| **新增 Phase 0: BC 预热** | 行为克隆防止探索坍塌 | 解决"全 Pass"问题 |
| **新增 Action Mask** | 领域知识注入 | 加速收敛，减少无意义探索 |
| **新增 Phase 2: 子博弈验证** | Stayman + 竞叫子博弈 | 在受控环境验证核心假设 |
| **新增 Go/No-Go 决策点** | 明确的验证标准 | 提前发现问题，避免浪费算力 |
| **新增时间信用分配讨论** | $r_{\text{info}}$ 的双重作用 | 明确方法的能力边界 |
| **调整时间规划** | Week 4-5 用于子博弈验证 | 适应 Colab Pro 算力约束 |
| **新增监控指标** | info-IMP correlation, value error | 诊断训练健康度 |

---

*文档版本: v7.1*  
*最后更新: 2025年3月*
