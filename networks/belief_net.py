"""
Belief Network (P105 rewrite)
==============================

P105: Uses OpenSpiel 571-dim observation as input instead of the previous
(observer_hand + encode_history_flat + position_embeds) = 268-dim encoding.

The old encoding had a fatal flaw: encode_history_flat collapsed all bidding
history into a 38-dim "which bids appeared" bitmap, discarding WHO made each bid.
A 1NT opening by partner vs 1NT opening by opponent looked identical. This made
honor prediction impossible (stuck at 0.750 = all-zero baseline).

The OpenSpiel 571-dim observation already contains:
  - Vulnerability (4 dim)
  - Pass-before-opening per relative player (4 dim)
  - Bidding history: 35 bids x 12 bits (who bid/doubled/redoubled) = 420 dim
  - Observer's hand (52 dim)
  - Additional features (91 dim)

Target position is passed separately as an embedding since the obs
doesn't encode which target we're predicting for.

Architecture:
  Input: 571 (OpenSpiel obs) + 32 (target_pos_embed) = 603 dim
  Trunk: 603 -> hidden -> hidden (2 layers)
  Honor head: hidden -> 16 (binary, sigmoid)
  Length head: hidden -> 32 (4 suits x 8 bins, softmax per suit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from env import NUM_BIDS, NUM_PLAYERS
from utils.hand_features import (
    BELIEF_DIM, HONOR_DIM, LENGTH_DIM, LENGTH_BINS, NUM_SUITS,
    HONOR_BIAS_INIT, LENGTH_BIAS_INIT,
    belief_accuracy,
)

# Try to get OBS_DIM from policy_net; fall back to 571 if not available
try:
    from networks.policy_net import OBS_DIM
except ImportError:
    OBS_DIM = 571


class BeliefNetwork(nn.Module):
    """
    Belief Network (P105): predicts target player's hand features from
    OpenSpiel 571-dim observation + target position.

    Input:
        obs_571: (batch, 571) OpenSpiel observation tensor
        target_pos: (batch,) int64, target player index (0-3)

    Output:
        honor_logits: (batch, 16) - sigmoid -> P(target holds AKQJ)
        length_logits: (batch, 4, 8) - softmax(dim=-1) -> suit length distribution
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 512):
        super().__init__()

        self.obs_dim = obs_dim
        self.target_embed = nn.Embedding(4, 32)

        # Trunk: obs + target_pos_embed -> hidden
        input_dim = obs_dim + 32
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Honor head: 16 independent binary logits
        self.honor_head = nn.Linear(hidden_dim, HONOR_DIM)
        # Length head: 4 suits x 8 bins = 32 logits
        self.length_head = nn.Linear(hidden_dim, LENGTH_DIM)

        # Bias initialization (P86)
        self._init_bias()

        # Legacy buffer for backward compat
        from utils.hand_features import build_pos_weight
        self.register_buffer('pos_weight', build_pos_weight())

    def _init_bias(self):
        """Prior bias initialization: honor -> 0.25, length -> prior distribution.
        
        Only bias is initialized to prior values. Weights use default init
        (Kaiming/Xavier) - mul_(0.01) was removed because it kills gradient
        flow from the heads back through the trunk, making the trunk unable
        to learn from honor/length signals.
        """
        with torch.no_grad():
            self.honor_head.bias.fill_(HONOR_BIAS_INIT)

            length_bias = torch.tensor(LENGTH_BIAS_INIT * NUM_SUITS, dtype=torch.float32)
            self.length_head.bias.copy_(length_bias)

    def forward(
        self,
        obs: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: (batch, 571) OpenSpiel observation
            target_pos: (batch,) int64 target player

        Returns: (batch, 48) raw logits
            [0:16] = honor logits (for BCE / sigmoid)
            [16:48] = length logits (for CE / softmax per suit)
        """
        tgt_embed = self.target_embed(target_pos)
        x = torch.cat([obs, tgt_embed], dim=-1)
        h = self.trunk(x)
        honor_logits = self.honor_head(h)
        length_logits = self.length_head(h)
        return torch.cat([honor_logits, length_logits], dim=-1)

    def get_probs(
        self,
        obs: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns calibrated probabilities (batch, 48):
        [0:16]  = sigmoid(honor_logits)
        [16:48] = softmax(length_logits per suit)
        """
        logits = self.forward(obs, target_pos)
        return self._logits_to_probs(logits)

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """logits (B, 48) -> calibrated probs (B, 48)."""
        honor_probs = torch.sigmoid(logits[:, :HONOR_DIM])
        length_logits = logits[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_probs = F.softmax(length_logits, dim=-1).view(-1, LENGTH_DIM)
        return torch.cat([honor_probs, length_probs], dim=-1)

    def compute_loss(
        self,
        obs: torch.Tensor,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mixed loss:
        - Honor [0:16]: BCEWithLogitsLoss
        - Length [16:48]: CrossEntropyLoss (4 suits x 8 classes)
        """
        logits = self.forward(obs, target_pos)

        honor_loss = F.binary_cross_entropy_with_logits(
            logits[:, :HONOR_DIM],
            target_features[:, :HONOR_DIM],
            reduction='mean',
        )

        length_logits = logits[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_target = target_features[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_labels = length_target.argmax(dim=-1)

        length_loss = F.cross_entropy(
            length_logits.reshape(-1, LENGTH_BINS),
            length_labels.reshape(-1),
            reduction='mean',
        )

        return honor_loss + length_loss

    @staticmethod
    def evaluate_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> dict:
        return belief_accuracy(probs, targets)

    # ==================================================================
    # EWC (Elastic Weight Consolidation) - P97
    # ==================================================================

    def compute_fisher(
        self,
        obs: torch.Tensor,
        tp: torch.Tensor,
        tgt: torch.Tensor,
        num_samples: int = 5000,
    ):
        """Compute diagonal Fisher Information Matrix on pretrain data."""
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()
                  if p.requires_grad}

        device = next(self.parameters()).device
        N = min(num_samples, obs.size(0))
        perm = torch.randperm(obs.size(0))[:N]

        CHUNK = 256
        count = 0
        for start in range(0, N, CHUNK):
            idx = perm[start:start + CHUNK]
            self.zero_grad()
            loss = self.compute_loss(
                obs[idx].to(device),
                tp[idx].to(device),
                tgt[idx].to(device),
            )
            loss.backward()
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) * len(idx)
            count += len(idx)

        for n in fisher:
            fisher[n] /= max(count, 1)

        # Normalize per-parameter
        for n in fisher:
            f_mean = fisher[n].mean()
            if f_mean > 1e-12:
                fisher[n] = fisher[n] / f_mean

        self._ewc_fisher = {n: f.detach().clone() for n, f in fisher.items()}
        self._ewc_star = {n: p.data.detach().clone()
                          for n, p in self.named_parameters() if p.requires_grad}

        n_params = sum(f.numel() for f in self._ewc_fisher.values())
        total_sum = sum(f.sum().item() for f in self._ewc_fisher.values())
        print(f"  [EWC] Fisher computed on {count} samples, {n_params} params. "
              f"Normalized. Total sum: {total_sum:.1f}")
        self.train()

    def ewc_penalty(self) -> torch.Tensor:
        """EWC penalty: mean_i [ F_i * (theta_i - theta*_i)^2 ]"""
        if not hasattr(self, '_ewc_fisher') or not self._ewc_fisher:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        n_params = 0
        for n, p in self.named_parameters():
            if n in self._ewc_fisher:
                loss += (self._ewc_fisher[n] * (p - self._ewc_star[n]) ** 2).sum()
                n_params += p.numel()
        return loss / max(n_params, 1)

    @property
    def has_ewc(self) -> bool:
        return hasattr(self, '_ewc_fisher') and bool(self._ewc_fisher)


class DualInfoComputer:
    """
    Dual-Info Bonus (P86)
    r_info = I(bid; hand | partner) - beta * I(bid; hand | opponent)
    """

    def __init__(self, belief_net: BeliefNetwork, beta: float = 0.5):
        self.belief_net = belief_net
        self.beta = beta

    def compute_info_gain(
        self,
        belief_before: torch.Tensor,
        belief_after: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Signed probe gain = CE(before, target) - CE(after, target).

        Negative values are retained: a bid that makes the frozen Judge less
        accurate must not be silently converted to zero reward.  Applying a
        ReLU here creates a positive noise bias that grows with auction length.
        Honor: BCE, Length: CE (per suit), equal weight average.
        """
        h_before = belief_before[:, :HONOR_DIM].clamp(1e-7, 1-1e-7)
        h_after  = belief_after[:, :HONOR_DIM].clamp(1e-7, 1-1e-7)
        h_target = target_features[:, :HONOR_DIM]

        honor_ce_before = F.binary_cross_entropy(
            h_before, h_target, reduction='none').mean(dim=-1)
        honor_ce_after = F.binary_cross_entropy(
            h_after, h_target, reduction='none').mean(dim=-1)

        l_before = belief_before[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        l_after  = belief_after[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        l_target = target_features[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        l_labels = l_target.argmax(dim=-1)

        l_ce_before = -torch.log(
            l_before.clamp(1e-7, 1.0).gather(2, l_labels.unsqueeze(-1)).squeeze(-1)
        ).mean(dim=-1)
        l_ce_after = -torch.log(
            l_after.clamp(1e-7, 1.0).gather(2, l_labels.unsqueeze(-1)).squeeze(-1)
        ).mean(dim=-1)

        ce_before = (honor_ce_before + l_ce_before) / 2.0
        ce_after  = (honor_ce_after  + l_ce_after)  / 2.0

        return ce_before - ce_after

    @staticmethod
    def compute_cross_entropy(
        belief: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the frozen-Judge hand cross entropy used by information gain."""
        honor = belief[:, :HONOR_DIM].clamp(1e-7, 1 - 1e-7)
        honor_target = target_features[:, :HONOR_DIM]
        honor_ce = F.binary_cross_entropy(
            honor, honor_target, reduction="none"
        ).mean(dim=-1)

        lengths = belief[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)
        length_target = target_features[:, HONOR_DIM:].view(
            -1, NUM_SUITS, LENGTH_BINS
        )
        length_labels = length_target.argmax(dim=-1)
        length_ce = -torch.log(
            lengths.clamp(1e-7, 1.0).gather(
                2, length_labels.unsqueeze(-1)
            ).squeeze(-1)
        ).mean(dim=-1)
        return (honor_ce + length_ce) / 2.0

    def compute_dual_info_bonus(
        self,
        partner_gain: torch.Tensor,
        opponent_leak: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        bonus = partner_gain - self.beta * opponent_leak
        metrics = {
            'partner_gain':    partner_gain.mean().item(),
            'opponent_leak':   opponent_leak.mean().item(),
            'info_ratio':      (partner_gain.mean() / (opponent_leak.mean() + 1e-8)).item(),
            'dual_info_bonus': bonus.mean().item(),
        }
        return bonus, metrics
