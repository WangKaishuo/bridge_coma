"""
Policy and Value Networks — MLP Architecture
=====================================================

P101: Belief-Conditioned Actor (BCA) — Partner + RHO

支持两种输入模式:
  1. Base mode (301 维) — 向后兼容旧实验
  2. Belief-conditioned mode (397 维) — P101 新框架
     = base obs (301) + partner belief (48) + RHO belief (48)
     每组 belief features = BeliefNetwork.get_probs() 输出
       [0:16]  honor probs (sigmoid)
       [16:48] length probs (softmax per suit)

设计理由 (参考 PBL, Tian et al. AAAI 2020):
  - Belief 输出作为 Actor 直接输入，使 agent 能"听懂"叫品含义
  - Partner belief: "partner 大概有什么牌" → 配合叫牌
  - RHO belief: "右手对手大概有什么牌" → 竞叫决策
  - r_info 的信息增益闭环: sender 传递更多信息 → receiver 的 belief
    更准确 → receiver 做出更好的决策 → 更好的 IMP
  - 去除对 KL 约束的依赖: Full Disclosure 由评估时交换 BeliefNet 实现

为什么是 RHO 而不是 LHO:
  - RHO 在你之前叫牌 → 信息相关性最高 (你需要理解 RHO 的叫品来决策)
  - LHO 在你之后叫牌 → 预推断价值有限
  - 信息论完备: 自己(13) + partner(48) + RHO(48) → LHO 可精确推算
    P(card∈LHO) = 1 - P(self) - P(partner) - P(RHO)
  - 自由度: 4人手牌只有2个自由度 (给定自己的牌), partner+RHO 恰好用尽

原始 301 维输入向量:
    vulnerability             :   4 维 (one-hot，4 种局况组合)
    当前玩家手牌 (one-hot)     :  52 维
    每个 bid 谁叫 (35 × 4)   : 140 维 — 4 维 one-hot，表示 N/E/S/W 谁叫
    每个 bid 的加倍状态 (35×3): 105 维 — 3 维 one-hot，未加倍/加倍/再加倍
    ─────────────────────────────────
    合计                      : 301 维

P101 额外 96 维 belief features:
    [Partner belief — 48 维]
      honor probs (16 dim)    :  16 维 — partner 持有 AKQJ 各花色的概率
      length probs (32 dim)   :  32 维 — partner 各花色套长的分布 (8 bins × 4)
    [RHO belief — 48 维]
      honor probs (16 dim)    :  16 维 — RHO 持有 AKQJ 各花色的概率
      length probs (32 dim)   :  32 维 — RHO 各花色套长的分布 (8 bins × 4)
    ─────────────────────────────────
    合计                      : 397 维
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from env import NUM_BIDS, NUM_PLAYERS

# ── 常量 ──────────────────────────────────────────────────────────────────────
NUM_REAL_BIDS  = 35      # 1C–7NT (bid index 3–37)
OBS_DIM        = 301     # 4 + 52 + 35×4 + 35×3
BASE_INPUT_DIM = OBS_DIM  # 向后兼容别名

# P98: Belief-conditioned Actor
# P101: Actor input = base obs (301) + partner belief (48) + RHO belief (48)
BELIEF_FEAT_DIM = 96     # 48 (partner) + 48 (RHO) — information-theoretically complete
BELIEF_OBS_DIM  = OBS_DIM + BELIEF_FEAT_DIM  # 301 + 96 = 397


# ==============================================================================
# 观测编码工具
# ==============================================================================

def encode_obs_flat(obs: Dict[str, np.ndarray], dealer: int, history_int: list) -> np.ndarray:
    """
    将环境 obs 字典转换为 301 维固定向量.

    此函数在 rollout 时调用，由 SubgameTrainer 负责传入 dealer 和 history_int.

    Args:
        obs        : BridgeBiddingEnv._get_observation() 返回的字典
        dealer     : 发牌人 (0=N, 1=E, 2=S, 3=W)，用于推算每步叫牌人
        history_int: 当前完整叫牌历史（整数列表，包含 Pass/X/XX/实质叫品）

    Returns:
        flat: (301,) float32

    注: 此函数在 Python 层逐样本调用，效率够用 (batch rollout 按环境并行，
        而非按时间步并行)。如果需要批量编码，直接 stack 结果即可。
    """
    vul  = obs['vulnerability']       # (2,) → 扩展为 (4,)
    hand = obs['hand']                # (52,)

    # vulnerability: 4 维 one-hot (4 种局况组合)
    # 索引: 0=无, 1=NS有利, 2=EW有利, 3=双方有利
    ns_vul, ew_vul = bool(vul[0] > 0.5), bool(vul[1] > 0.5)
    vul4 = np.zeros(4, dtype=np.float32)
    vul4[int(ns_vul) * 2 + int(ew_vul)] = 1.0

    # 只考虑 35 个实质叫品 (index 3–37 → 0–34)
    who_called   = np.zeros((NUM_REAL_BIDS, 4),  dtype=np.float32)  # (35, 4)
    double_state = np.zeros((NUM_REAL_BIDS, 3),  dtype=np.float32)  # (35, 3)

    # 追踪每个实质叫品的加倍状态
    # double_state[i]: one-hot [未加倍, 已加倍, 已再加倍]
    real_bid_indices = {}   # bid_int → real_idx (0-34)

    last_real_bid_real_idx = -1

    for step_idx, bid in enumerate(history_int):
        caller = (dealer + step_idx) % NUM_PLAYERS

        if bid >= 3:  # 实质叫品 (1C=3 … 7NT=37)
            real_idx = bid - 3  # 0–34
            real_bid_indices[real_idx] = step_idx
            who_called[real_idx, caller] = 1.0
            double_state[real_idx, 0]   = 1.0   # 默认：未加倍
            last_real_bid_real_idx = real_idx

        elif bid == 1 and last_real_bid_real_idx >= 0:  # Double
            ri = last_real_bid_real_idx
            double_state[ri, 0] = 0.0
            double_state[ri, 1] = 1.0

        elif bid == 2 and last_real_bid_real_idx >= 0:  # Redouble
            ri = last_real_bid_real_idx
            double_state[ri, 1] = 0.0
            double_state[ri, 2] = 1.0

    flat = np.concatenate([
        vul4,                           # 4
        hand,                           # 52
        who_called.flatten(),           # 140
        double_state.flatten(),         # 105
    ])
    assert flat.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {flat.shape}"
    return flat


def batch_encode_obs(obs_list, dealers, history_ints):
    """批量编码: 返回 (B, 301) float32 ndarray."""
    return np.stack([
        encode_obs_flat(o, d, h)
        for o, d, h in zip(obs_list, dealers, history_ints)
    ])


def make_belief_features_prior() -> np.ndarray:
    """
    P101: 返回 belief 先验特征向量 (96,) = partner (48) + RHO (48).

    在没有叫牌信息时的 belief 输出:
      - Honor (16 dim): 每张荣誉牌有 0.25 概率在目标玩家手中
      - Length (32 dim): 均匀分布 [0.125] × 8 per suit

    两组先验完全相同（partner 和 RHO 的先验分布一致）。

    用于:
      1. BC pretrain 时 actor 的 belief 输入（尚无 belief net）
      2. 开叫前第一步的 belief 输入（无历史信息）
      3. FSP pool 对手的 belief 输入（不主动推断我方叫品）
    """
    honor_prior = np.full(16, 0.25, dtype=np.float32)
    length_prior = np.full(32, 0.125, dtype=np.float32)
    single_prior = np.concatenate([honor_prior, length_prior])  # (48,)
    return np.concatenate([single_prior, single_prior])          # (96,)


def append_belief_features(flat_obs: np.ndarray,
                           belief_feats: np.ndarray) -> np.ndarray:
    """
    P101: 拼接 base obs (301) + belief features (96) → (397,).

    Args:
        flat_obs:      (301,) from encode_obs_flat
        belief_feats:  (96,)  = partner (48) + RHO (48) from belief queries or prior

    Returns:
        combined: (397,) float32
    """
    return np.concatenate([flat_obs, belief_feats])


def encode_history_flat(history: torch.Tensor) -> torch.Tensor:
    """
    将叫牌历史序列压缩为固定维度向量，供 BeliefNetwork 使用.

    Args:
        history: (B, max_len, NUM_BIDS) float32 one-hot tensor

    Returns:
        flat: (B, NUM_BIDS * NUM_PLAYERS) float32

    实现：对时间轴做 max-pool（判断每个 bid 是否出现过），
    然后 tile NUM_PLAYERS 次构成固定向量。
    BeliefNetwork input_dim = 52 + NUM_BIDS*NUM_PLAYERS + 32 + 32 = 268。
    """
    bid_presence = history.max(dim=1).values   # (B, NUM_BIDS)
    return bid_presence.repeat(1, NUM_PLAYERS)  # (B, NUM_BIDS * NUM_PLAYERS)


# ==============================================================================
# AllHandsEncoder  (仅 Critic 使用)
# ==============================================================================

class AllHandsEncoder(nn.Module):
    """
    集中式 Critic 专用: 将全局 4 张手牌编码为 256 维向量.

    输入:  (batch, 4, 52) float32
    输出:  (batch, 256)
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 52, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, all_hands: torch.Tensor) -> torch.Tensor:
        # all_hands: (batch, 4, 52)
        return self.net(all_hands.view(all_hands.size(0), -1))


# ==============================================================================
# MLPPolicyNetwork  (Actor)
# ==============================================================================

class MLPPolicyNetwork(nn.Module):
    """
    Actor 网络: 4 × 1024 MLP, 输入 301 维固定向量.

    输入: flat_obs (batch, 301)  +  legal_actions (batch, 38)
    输出: logits (batch, 38)，已 mask 非法动作

    get_action / evaluate_actions 接口与旧 PolicyNetwork 兼容.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024,
                 num_actions: int = NUM_BIDS):
        super().__init__()
        self.obs_dim     = obs_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def _masked_logits(self, flat_obs: torch.Tensor,
                       legal_actions: torch.Tensor) -> torch.Tensor:
        logits = self.net(flat_obs)
        return logits - 1e9 * (1.0 - legal_actions)

    def forward(self, flat_obs: torch.Tensor,
                legal_actions: torch.Tensor) -> torch.Tensor:
        """返回 masked logits."""
        return self._masked_logits(flat_obs, legal_actions)

    def get_action(
        self,
        flat_obs: torch.Tensor,
        legal_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作.

        Returns:
            action   : (batch,) int64
            log_prob : (batch,) float32
            entropy  : (batch,) float32
        """
        logits = self._masked_logits(flat_obs, legal_actions)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        flat_obs: torch.Tensor,
        legal_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估已有动作的 log_prob 和 entropy (PPO update 用).

        Returns:
            log_prob : (batch,)
            entropy  : (batch,)
        """
        logits = self._masked_logits(flat_obs, legal_actions)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy()


# ==============================================================================
# MLPValueNetwork  (Critic)
# ==============================================================================

class MLPValueNetwork(nn.Module):
    """
    Critic 网络: 4 × 1024 MLP + 可选 AllHandsEncoder (CTDE).

    输入:
        flat_obs  : (batch, 301)
        all_hands : (batch, 4, 52)  — 仅 centralized=True 时需要
    输出:
        value     : (batch,)
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 1024,
                 centralized: bool = True):
        super().__init__()
        self.centralized = centralized

        if centralized:
            self.all_hands_encoder = AllHandsEncoder(output_dim=256)
            input_dim = obs_dim + 256
        else:
            input_dim = obs_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, flat_obs: torch.Tensor,
                all_hands: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.centralized and all_hands is not None:
            hand_feat = self.all_hands_encoder(all_hands)
            x = torch.cat([flat_obs, hand_feat], dim=-1)
        else:
            x = flat_obs

        return self.net(x).squeeze(-1)


# ==============================================================================
# 向后兼容别名 (旧代码使用 PolicyNetwork / ValueNetwork / ActorCritic)
# ==============================================================================

PolicyNetwork = MLPPolicyNetwork
ValueNetwork  = MLPValueNetwork


class ActorCritic(nn.Module):
    """
    向后兼容容器.

    新代码应直接使用 MLPPolicyNetwork + MLPValueNetwork.
    此类仅供不想改调用方的旧接口使用.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 1024,
        centralized_critic: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.actor  = MLPPolicyNetwork(obs_dim, hidden_dim)
        self.critic = MLPValueNetwork(obs_dim, hidden_dim, centralized_critic)

    def get_action_and_value(
        self,
        flat_obs: torch.Tensor,
        legal_actions: torch.Tensor,
        all_hands: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy = self.actor.get_action(
            flat_obs, legal_actions, deterministic)
        value = self.critic(flat_obs, all_hands)
        return action, log_prob, entropy, value
