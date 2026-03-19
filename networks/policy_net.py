"""
Policy and Value Networks
=========================

Actor-Critic 网络架构 (P52 重构)

P52 变更:
- 删除 HandEncoder / HistoryEncoder (LSTM)
- 改用 MLP + 全局输入拼接 (Kita et al. 2024 风格)
- history 编码: per-bid who-made-it binary (38×4=152 dims)
  "哪个 player 叫了叫品 b" 对桥牌叫牌是充分且无损的编码,
  因为叫牌严格单调递增且每个实质叫品在序列中最多出现一次.
- 整体输入: 52 (hand) + 152 (history) + 4 (position) + 4 (vuln) = 212 dims
  + belief_dim (可选, 默认0)
- 网络深度: 4层 MLP × 1024 units, ReLU

设计理由:
  LSTM 在 Stayman/Competitive 子博弈中无额外价值:
  - 序列极短 (≤12 token), 无长程依赖
  - LSTM 两层占原 actor 74% 参数, 训练不稳定
  - 全局拼接输入下 MLP 更快收敛, value loss 更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from env import NUM_BIDS, NUM_PLAYERS


# ── 常量 ──────────────────────────────────────────────────────────────────────
HAND_DIM     = 52           # one-hot 手牌
HISTORY_DIM  = NUM_BIDS * NUM_PLAYERS   # 38×4 = 152, per-bid who-made-it
POSITION_DIM = 4
VULN_DIM     = 2
BASE_INPUT_DIM = HAND_DIM + HISTORY_DIM + POSITION_DIM + VULN_DIM  # 210


def encode_history_flat(history_obs: torch.Tensor) -> torch.Tensor:
    """
    将 (B, max_len, NUM_BIDS) one-hot 历史序列转换为
    (B, NUM_BIDS × NUM_PLAYERS) who-made-it 二值向量.

    对每个叫品 b: 找到历史中第一次叫 b 的位置 t,
    则 player = (dealer + t) % 4.
    由于叫牌单调递增, 每个实质叫品最多出现一次.
    Pass (bid=0) 可被多家叫, 取最后一次 Pass 的 player (近似).

    输出 shape: (B, NUM_BIDS * NUM_PLAYERS), 值为 0/1.
    """
    B, T, _ = history_obs.shape
    device = history_obs.device

    # (B, T) — 每步叫的是哪个 bid
    bid_indices = history_obs.argmax(dim=-1)   # 0 = 全零padding (当作 Pass)

    # 有效步 mask: history_obs.sum(dim=-1) > 0
    valid = (history_obs.sum(dim=-1) > 0)      # (B, T)

    # 为每个时间步分配 player id (假设 dealer=NORTH=0, 轮流叫牌)
    # player_at_step[t] = t % NUM_PLAYERS
    step_ids = torch.arange(T, device=device) % NUM_PLAYERS  # (T,)
    step_ids = step_ids.unsqueeze(0).expand(B, -1)            # (B, T)

    # 结果: (B, NUM_BIDS, NUM_PLAYERS)
    result = torch.zeros(B, NUM_BIDS, NUM_PLAYERS, device=device)

    for t in range(T):
        mask = valid[:, t]                  # (B,)
        bids = bid_indices[:, t]            # (B,)
        players = step_ids[:, t]            # (B,) — 标量 player id

        # scatter: result[b, bids[b], players[b]] = 1
        # 用 scatter_ 实现批量赋值
        b_idx = torch.arange(B, device=device)[mask]
        if b_idx.numel() == 0:
            continue
        b_bids    = bids[mask]
        b_players = players[mask]
        result[b_idx, b_bids, b_players] = 1.0

    return result.view(B, -1)   # (B, NUM_BIDS * NUM_PLAYERS)


class PolicyNetwork(nn.Module):
    """
    Actor 策略网络 (P52).

    输入 (全局拼接):
      hand (52) + history_flat (152) + position (4) + vulnerability (2)
      [+ belief (可选, 默认0)]
    输出: (B, 38) logits

    belief_dim > 0 时接受 obs['belief'], stop-gradient.
    架构支持任意 player 的双向 belief (N↔S, E↔W 等).
    """

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 4,
                 belief_dim: int = 0):
        super().__init__()
        self.belief_dim = belief_dim
        input_dim = BASE_INPUT_DIM + belief_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, NUM_BIDS))
        self.net = nn.Sequential(*layers)

    def _build_input(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hist_flat = encode_history_flat(obs['history'])
        parts = [obs['hand'], hist_flat, obs['position'], obs['vulnerability']]
        if self.belief_dim > 0:
            if 'belief' in obs:
                parts.append(obs['belief'].detach())
            else:
                parts.append(torch.zeros(
                    obs['hand'].shape[0], self.belief_dim,
                    device=obs['hand'].device, dtype=obs['hand'].dtype))
        return torch.cat(parts, dim=-1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(self._build_input(obs))

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        mask   = obs['legal_actions']
        logits = logits - 1e9 * (1 - mask)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, obs, actions):
        logits = self.forward(obs)
        mask   = obs['legal_actions']
        logits = logits - 1e9 * (1 - mask)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy()



class PopArtLayer(nn.Module):
    """
    PopArt 自适应 value 归一化层 (Yu et al. 2021, MAPPO Suggestion #1).

    原理:
      - 维护 returns 的 running mean μ 和 std σ
      - critic 在归一化空间 [-3,3] 内学习, 避免 value scale 随 reward 分布漂移
      - 每次更新 μ/σ 时同步调整线性层权重, 保证去归一化输出不变 (Art步骤)
      - forward 返回去归一化值 (供 GAE 使用); normalized_forward 返回归一化值 (供 loss 使用)

    Phase 切换时 value landscape 剧变 → 原始 critic 的 vl 会从 0.07 跳到 2.8.
    PopArt 自动吸收这个 scale 变化, vl 始终在 [0, 1] 附近.
    """

    def __init__(self, in_features: int, beta: float = 3e-4, epsilon: float = 1e-5):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.beta    = beta       # running stats 更新速率
        self.epsilon = epsilon

        # running stats (non-trainable buffers)
        self.register_buffer('mu',    torch.zeros(1))
        self.register_buffer('nu',    torch.ones(1))   # E[x²], 用于计算 σ
        self.register_buffer('sigma', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回去归一化 value (用于 GAE bootstrap)."""
        return self.linear(x).squeeze(-1) * self.sigma + self.mu

    def normalized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回归一化 value (用于 loss 计算)."""
        return self.linear(x).squeeze(-1)

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        """将 returns 归一化为训练目标."""
        return (targets - self.mu) / (self.sigma + self.epsilon)

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        """
        用新一批 returns 更新 μ/σ, 并同步调整线性层权重 (Art步骤).

        调用时机: 每次 critic 更新前 (在 critic_warmup_step 和 safe_update 里).
        """
        batch_mean = targets.mean()
        batch_sq   = (targets ** 2).mean()

        old_sigma = self.sigma.clone()
        old_mu    = self.mu.clone()

        # EMA 更新
        self.mu.copy_((1 - self.beta) * self.mu + self.beta * batch_mean)
        self.nu.copy_((1 - self.beta) * self.nu + self.beta * batch_sq)
        self.sigma.copy_(
            torch.clamp((self.nu - self.mu ** 2).sqrt(), min=self.epsilon)
        )

        # Art: 同步调整权重, 保证去归一化输出在更新前后连续
        # w_new = w_old * old_sigma / new_sigma
        # b_new = (b_old * old_sigma + old_mu - new_mu) / new_sigma
        if old_sigma.item() > self.epsilon:
            scale = old_sigma / self.sigma
            self.linear.weight.data.mul_(scale)
            self.linear.bias.data.mul_(scale)
            self.linear.bias.data.add_((old_mu - self.mu) / self.sigma)


class ValueNetwork(nn.Module):
    """
    Critic 价值网络 (P52).

    集中式 (CTDE): 接收 all_hands (4×52) 展平后拼入输入.
    输入: hand(52) + history_flat(152) + position(4) + vuln(2) + all_hands(208) = 418
    """

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 4,
                 centralized: bool = True):
        super().__init__()
        self.centralized = centralized
        input_dim = BASE_INPUT_DIM
        if centralized:
            input_dim += 4 * HAND_DIM   # 208

        # 主干: (num_layers-1) 层 MLP, 最后一层用 PopArtLayer 替代普通 Linear
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.popart = PopArtLayer(in_dim)   # 替代原来的 nn.Linear(in_dim, 1)

    def _featurize(self, obs: Dict[str, torch.Tensor],
                   all_hands: Optional[torch.Tensor] = None) -> torch.Tensor:
        hist_flat = encode_history_flat(obs['history'])
        parts = [obs['hand'], hist_flat, obs['position'], obs['vulnerability']]
        if self.centralized and all_hands is not None:
            B = all_hands.shape[0]
            parts.append(all_hands.view(B, -1))
        return self.trunk(torch.cat(parts, dim=-1))

    def forward(self, obs: Dict[str, torch.Tensor],
                all_hands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回去归一化 value (供 GAE bootstrap / collect_episodes 使用)."""
        return self.popart(self._featurize(obs, all_hands))

    def normalized_forward(self, obs: Dict[str, torch.Tensor],
                           all_hands: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回归一化 value (供 loss 计算使用)."""
        return self.popart.normalized_forward(self._featurize(obs, all_hands))

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        return self.popart.normalize_target(targets)

    def update_stats(self, targets: torch.Tensor):
        self.popart.update_stats(targets)
