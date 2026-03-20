# Bridge-COMA: Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination

**MSc Research Project — Kaishuo Wang, 2026**

---

## 1. 研究问题

在合作-对抗混合多智能体环境中，agent 必须同时做到两件事：与队友有效传递信息，以及最小化对对手的信息泄露。

$$r_{\text{info}} = \underbrace{I(\text{bid};\,\text{hand} \mid \text{partner})}_{\text{partner推断增益}} - \beta \cdot \underbrace{I(\text{bid};\,\text{hand} \mid \text{opponent})}_{\text{opponent信息泄露}}$$

**三个研究问题**：
1. **有效性**：dual-information bonus 是否在 IMP 上显著优于 vanilla MAPPO？
2. **机制验证**：partner 与 opponent 的推断精度差异是否与叫牌效率相关？
3. **消融分析**：partner term 与 opponent penalty（β）各自贡献多少？

---

## 2. 网络架构

### 2.1 Actor（301维MLP，对齐 Kita et al. 2024）

**输入**：301 维固定向量（无LSTM，批量rollout友好）
```
vulnerability             :   4 维
当前玩家手牌 (one-hot)     :  52 维
每个 bid 谁叫 (35 × 4)   : 140 维
每个 bid 加倍状态 (35×3)  : 105 维
─────────────────────────────────
合计                      : 301 维
```

**Actor**：4层MLP，每层1024，ReLU，输出38维logits + action mask
**Critic**：同Actor结构，额外接收AllHandsEncoder（4×52→256），CTDE

**HAPPO 八独立网络架构**：`actor_n/s/e/w` + `critic_n/s/e/w`，NS和EW完全独立，消除cross-contamination。

### 2.2 Belief Network

**输入**：observer_hand(52) + history_flat(NUM_BIDS×NUM_PLAYERS=152) + pos_embed×2(64) = 268维
**输出**：(batch, 48) logits
```
[0:16]  AKQJ归属   — 16维独立binary
[16:48] 套长one-hot — 32维（每门×8档）
```
**损失**：BCEWithLogitsLoss(pos_weight=3.0)

### 2.3 r_info 计算

$$I(\text{bid};\,\text{hand} \mid \text{obs}) \approx \text{CE}(q_\phi(h_{t-1}),\, \text{hand}) - \text{CE}(q_\phi(h_t),\, \text{hand})$$

---

## 3. 训练流程

### 3.1 SL 预训练（Phase 2）

**数据**：OpenSpiel SAYC 数据集（WBridge5生成）
- 路径：`/content/drive/MyDrive/bridge_data/sayc_train.txt`（已下载到Google Drive）
- 格式：每行一局，前52个整数为发牌（deck[card]=player），之后为叫牌动作序列
- 动作映射：OpenSpiel 52→Pass(0), 53-87→1C-7NT(3-37), 88→X(1), 89→XX(2)
- 训练集：9,984,884 state-action pairs；验证集：86,288 pairs

**SL训练参数**（`utils/sl_pretrain.py`）：

| 参数 | 值 |
|------|-----|
| epochs | 10（acc在epoch3后基本收敛） |
| batch_size | 2048 |
| lr | 3e-4 |
| hidden_dim | 1024 |
| class_weight[Pass] | 0.1（修正57% Pass的class imbalance） |
| early_stop target | non_pass_acc ≥ 0.36 |

**运行命令**：
```bash
python utils/sl_pretrain.py \
    --train /content/drive/MyDrive/bridge_data/sayc_train.txt \
    --valid /content/drive/MyDrive/bridge_data/sayc_valid.txt \
    --out results/sl_base.pt \
    --batch_size 2048 \
    --lr 3e-4 \
    --epochs 10 \
    --device cuda
```

**预期结果**：non_pass_acc ≈ 0.35-0.40，整体acc ≈ 0.29（Pass占57%压低整体acc，non_pass_acc才是真实指标）

**输出格式**（对齐MAPPOAgent.save()）：
```python
{
    'actor_n': state_dict,  # 四方初始相同，RL后分化
    'actor_s': state_dict,
    'actor_e': state_dict,
    'actor_w': state_dict,
    'val_acc': float,
    'obs_dim': 301,
    'hidden_dim': 1024,
}
```

### 3.2 RL 微调（双桌IBR）

**结构**：每轮用同一批牌跑两桌
- 桌1（NS训练）：NS用当前agent，EW用frozen FSP
- 桌2（EW训练）：EW用当前agent，NS用frozen FSP

**FSP**：pool_size=10，每2轮存一次snapshot，防止policy cycling

**KL Anchor**：SL结束后设为anchor，RL训练中 `loss += kl_lambda × KL(π_current ∥ π_SL)`，λ从0.5退火到0.1

**JIT Belief Burn-in**：每轮桌1结束后，用1000局rollout对Belief Net做3 epoch快速微调

**RL参数**（`subgames/subgame_trainer.py` SubgameConfig默认值）：

| 参数 | 值 |
|------|-----|
| num_rounds | 20 |
| steps_per_phase | 500 |
| deals_per_step | 32 |
| lr | 1e-6 |
| critic_lr_ratio | 5.0 |
| batch_size | 256 |
| entropy_coef | 1e-3 |
| kl_lambda_start | 0.5 |
| kl_lambda_end | 0.1 |
| fsp_pool_size | 10 |
| bc_warmup_samples | 5000（SL后不再用rule-based BC，此参数弃用） |

### 3.3 下一步：在RL中加载SL Checkpoint

**待实现**：`subgame_validation.py` 需要加一个 `--sl_ckpt` 参数，训练开始前用SL权重初始化所有actor，跳过rule-based BC warmup。

具体修改：
1. `subgame_validation.py`：加 `--sl_ckpt` CLI参数
2. `subgame_trainer.py`：加 `load_sl_checkpoint(path)` 方法，把四个actor都初始化为SL权重
3. `subgame_validation.py`：如果传了`--sl_ckpt`，跳过`run_bc_warmup()`，直接`load_sl_checkpoint()`

---

## 4. 实验设计

### 4.1 当前进度

- ✅ Phase 0：工程框架（301维MLP，双桌rollout，FSP pool，KL anchor）
- ✅ Stayman子博弈验证（预期null result，基础架构稳定）
- ✅ Competitive子博弈环境（1H-1S前缀，四方独立actor，批量rollout）
- 🔄 **Phase 2 SL预训练**（当前位置：已跑10 epoch，non_pass_acc≈0.35）
- ⬜ Phase 3 主实验（Agent A vs B vs B0 vs B2 vs IPPO）
- ⬜ Phase 4 多seed验证

### 4.2 主实验配置

| Agent | 配置 | 说明 |
|-------|------|------|
| A | MAPPO, β=0 | 控制组 |
| B0 | MAPPO + r_info, β=0 | 消融：仅partner term |
| B05 | MAPPO + r_info, β=0.05 | 主实验配置 |
| B2 | MAPPO + r_info, β=0.2 | 消融：高β |
| IPPO | 独立学习 | baseline |

### 4.3 评估方案

**主要指标**：DDS oracle IMP regret（绝对基准，与对手无关）
**统计检验**：Wilcoxon signed-rank test，p < 0.05
**置信区间**：bootstrap 95% CI，3个seed

---

## 5. 关键代码文件

```
bridge-coma/
├── experiments/
│   ├── subgame_validation.py    # competitive子博弈验证入口
│   └── train.py
├── subgames/
│   ├── competitive_env.py       # 1H-1S环境，DDS oracle reward
│   ├── subgame_trainer.py       # 双桌IBR，FSP，KL anchor，批量rollout
│   └── action_mask.py
├── networks/
│   ├── policy_net.py            # 301维MLP，encode_obs_flat，encode_history_flat
│   └── belief_net.py            # BeliefNetwork，DualInfoComputer
├── algorithms/
│   ├── mappo.py                 # HAPPO：actor_n/s/e/w + critic_n/s/e/w
│   └── ippo.py
└── utils/
    ├── sl_pretrain.py           # SL预训练脚本（新增）
    ├── fsp_pool.py              # FSP checkpoint pool
    ├── dds_data.py
    ├── scoring.py
    ├── imp.py
    ├── hand_features.py
    └── running_stats.py
```

---

## 6. 重要Bug修复历史（P系列）

### Stayman阶段（P1-P52，详见README_stayman_archive.md）
- BC策略学到99.5%准确率 → Stayman是预期null result（EW全Pass，β term失活）

### Competitive阶段（P53-P72）

**P53**：新架构重构——301维MLP替换LSTM Actor，HAPPO四独立网络

**P54**：`DualInfoComputer.compute()`不存在 → 改用`compute_info_gain()`+`compute_dual_info_bonus()`

**P55**：`Contract(redoubled=0)`字段不存在 → 删除`redoubled`参数

**P56**：`encode_history_flat`从`policy_net`导入但不存在 → 添加该函数

**P57**：`BASE_INPUT_DIM`从`policy_net`导入但不存在 → 添加别名

**P58**：`subgames`包路径问题 → `subgame_validation.py`加`sys.path.insert`

**P59**：reward=0（批量rollout用裸`BridgeBiddingEnv`绕过reward计算）→ 改用`CompetitiveSubgameEnv`实例

**P60**：`_collect_episodes_batch`缩进错误（方法被嵌套在另一个方法里）→ 修复缩进

**P61**：`_print_log`同样缩进错误 → 修复

**P62**：`critic_warmup`只收集NS数据，EW Critic冷启动vl爆炸 → 改为NS/EW各收一半

**P63**：`set_bc_anchor`只设置NS，EW无KL anchor → 改为四方都设置

**P64**：`_compute_info_bonus`过滤掉EW步骤 → 改为四方都计算r_info

**P65**：`_print_diagnostics`读`n_metrics`但日志用`ew_metrics` → 修复key名称

**P66**：`actor_n/s`映射到EW → 扩展为独立的`actor_e/w/critic_e/w`八网络

**P67**：BC warmup只训NS → 改为四方各自独立训练

**P68**：rule-based BC过拟合（loss=0.24），entropy collapse（ent≈0.2）→ 改用SAYC数据SL预训练

**P69**：SL训练Pass class imbalance（Pass占57%，模型全猜Pass）→ 加class_weight[Pass]=0.1

**P70**：SL val_acc波动（valid_loader shuffle=False）→ 改为shuffle=True

**P71**：SL acc瓶颈在epoch3后卡住0.35 → epochs改为10，加即时保存best checkpoint

---

## 7. 当前状态与下一步

### 当前状态
- SL预训练：`results/sl_base.pt`，non_pass_acc≈0.35
- Competitive环境：完全就绪，批量rollout，双桌训练，四方独立网络
- 所有已知bug已修复

### 下一步（新对话开始时）

**Step 1**：在`subgame_validation.py`和`subgame_trainer.py`中实现SL checkpoint加载

```python
# subgame_trainer.py 新增方法
def load_sl_checkpoint(self, path: str):
    """加载SL预训练权重到四个actor，跳过rule-based BC warmup."""
    ckpt = torch.load(path, map_location=self.device)
    for role in ('actor_n', 'actor_s', 'actor_e', 'actor_w'):
        if role in ckpt:
            actor = getattr(self.agent.model, role)
            actor.load_state_dict(ckpt[role])
    print(f"[SL Checkpoint] Loaded from {path}, val_acc={ckpt.get('val_acc', 'N/A'):.4f}")
```

**Step 2**：运行competitive子博弈验证（quick模式确认entropy正常）

```bash
python experiments/subgame_validation.py \
    --competitive_data data/competitive_100k.npz \
    --sl_ckpt results/sl_base.pt \
    --seed 42 --beta 0.05 --rounds 10 \
    --device cuda --quick
```

**Step 3**：确认entropy > 1.0（SL后应该在1.5-2.5之间），vl正常收敛，ir > 0

**Step 4**：跑正式实验（20轮，1000局评估）

---

## 8. 算力预算（L4 高RAM）

| 环节 | 时间 |
|------|------|
| SL预训练（10 epoch，全量） | ~20分钟 |
| Competitive quick验证 | ~10分钟 |
| 正式实验单agent（20轮） | ~3小时 |
| Agent A + B总计 | ~6小时 |
| 多seed（3×2） | ~18小时（分批） |

---

## 9. 参考文献

1. Kita et al. (2024). A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI. IEEE CoG.
2. Lockhart et al. (2020). Human-Agent Cooperation in Bridge Bidding. NeurIPS.
3. Gong et al. (2024). Bridge Bidding via Deep RL and Belief MCTS. IEEE/CAA J. Autom. Sinica.
4. Foerster et al. (2018). Counterfactual Multi-Agent Policy Gradients. AAAI.

---

*README v2.0 — 2026年3月*
*上一版本：README_new.md（v1.0）*
*Stayman实验历史：README_stayman_archive.md*
