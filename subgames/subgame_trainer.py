"""MAPPO/FSP trainer for the competitive bridge-bidding subgame.

Actors expose the standard 571-dimensional OpenSpiel observation API and build
partner/RHO belief features internally.  A separate frozen BeliefNet is the
training-only Judge.  For a bid by player ``i``, both Judge receiver views
predict ``i``'s hand: the partner view conditions on the partner's private hand
and the opponent view conditions on the next opponent's private hand.
"""

from __future__ import annotations

import numpy as np
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from env import NUM_PLAYERS, NUM_BIDS, BID_PASS, NORTH, EAST, SOUTH, WEST
from networks.belief_net import BeliefNetwork, DualInfoComputer
from networks.policy_net import (
    MLPPolicyNetwork, MLPValueNetwork, OBS_DIM,
    convert_hands_suit_to_rank, hands_to_openspiel_state,
    get_openspiel_obs, ours_to_openspiel_raw,
    physical_to_openspiel_player,
)
from algorithms.mappo import MAPPOAgent, MAPPOConfig
from utils.running_stats import RunningStats
from utils.hand_features import (
    hand_to_belief_target, batch_hand_to_belief_target,
    belief_accuracy, BELIEF_DIM,
)
from utils.fsp_pool import FSPPool
from utils.imp import score_to_imp

# P109: max entries for _encode_for_actor obs/state caches per trainer instance.
# A rollout batch of 512 deals × ~8 steps × 4 players ≈ 16k unique (hands, hist) keys.
# 32k gives comfortable headroom while staying well under 1 GB memory.
_OBS_CACHE_MAX = 32_768


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class SubgameConfig:
    """Training configuration for the formal A/B/C experiment."""
    # Rollout scale
    num_rounds:       int   = 20
    steps_per_phase:  int   = 500
    deals_per_step:   int   = 32
    accumulate_steps: int   = 4

    # Learning rates
    lr:              float  = 1e-6
    critic_lr_ratio: float  = 10.0
    belief_lr:       float  = 1e-4

    # PPO
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_ratio:       float = 0.2
    num_epochs:       int   = 4
    batch_size:       int   = 256
    entropy_coef:     float = 1e-3
    value_coef:       float = 0.5
    max_grad_norm:    float = 0.5

    # Optional KL baseline
    kl_lambda_start:  float = 0.0
    kl_lambda_end:    float = 0.0
    kl_anneal_frac:   float = 0.0

    # ── KL Early Stopping ───────────────────────────────────────────────────
    kl_early_stop_threshold: float = 0.015

    # Dual-information reward
    use_info_bonus:      bool  = False
    beta:                float = 0.05
    info_reward_weight:  float = 0.05
    info_scale_calibration_deals: int = 2048

    # Deployed actor: 571 external observation -> internal 96-dim belief head.
    belief_conditioned: bool = True
    actor_belief_coef:  float = 0.1

    # Fictitious self-play
    fsp_pool_size:    int   = 0
    fsp_add_interval: int   = 2
    self_play:        bool  = False

    # ── P126: FSP Quality Gate + Weighted SL Sampling ──────────────────────
    fsp_quality_gate:          bool  = True   # Enable quality gate before pool insertion
    fsp_gate_eval_deals:       int   = 200    # Deals to play vs SL for quality evaluation
    fsp_gate_max_auction_len:  int   = 7      # Reject if median auction length > this (SL median=6)
    fsp_gate_max_double_rate:  float = 0.60   # Reject if doubled contract rate > this (SL ≈ 53%)
    fsp_sl_sample_prob:        float = 0.30   # Minimum probability of sampling SL permanent member

    # ── Belief Net Update ────────────────────────────────────────────────
    # P93: on-policy update (epochs=8, lr=5e-5) — caused catastrophic
    #   forgetting of pretrain foundation (val_loss 1.76→2.19 in Round 1).
    # P96: freeze_belief=True by default. With strong KL (λ=0.5),
    #   policy stays near SAYC and pretrain Belief Net remains valid.
    belief_update_epochs: int  = 3            # P95: only used if freeze_belief=False
    belief_update_lr:     float = 1e-5        # P95: only used if freeze_belief=False
    freeze_belief:        bool  = True        # P96: frozen by default

    # ── EWC for Belief Net (P97) ─────────────────────────────────────
    use_ewc:              bool  = False       # P97: EWC-protected on-policy update
    ewc_lambda:           float = 100.0       # P97: EWC penalty strength (Fisher normalized, mean penalty)
    ewc_fisher_samples:   int   = 5000        # Samples for Fisher computation

    # ── Critic Warmup ───────────────────────────────────────────────────────
    critic_prewarm_deals:  int   = 2048
    critic_prewarm_epochs: int   = 10
    critic_prewarm_conv_tol: float = 0.05

    # ── BC Warmup（rule-based）──────────────────────────────────────────────
    bc_warmup_samples: int  = 5000
    bc_warmup_epochs:  int  = 20
    bc_warmup_lr:      float = 1e-4

    hidden_dim:       int   = 1024        # 4 × 1024 MLP

    active_players:      Optional[List[int]] = None
    eval_interval:       int   = 200
    log_interval:        int   = 50
    device:              str   = 'cpu'
    early_stop_patience:   int   = 8
    early_stop_vl_delta:   float = 0.15
    early_stop_enabled:    bool  = False   # P123: disabled — fixed schedule for reproducibility


# ==============================================================================
# Belief Replay Buffer (P84)
# ==============================================================================

class FlatRolloutBuffer:
    """See the formal README for the current behavior contract."""

    def __init__(self, device: str):
        self.device = device
        self.reset()

    def reset(self):
        self.flat_obs:      List[torch.Tensor] = []
        self.legal_actions: List[torch.Tensor] = []
        self.actions:       List[torch.Tensor] = []
        self.log_probs:     List[torch.Tensor] = []
        self.rewards:       List[float]         = []
        self.values:        List[torch.Tensor] = []
        self.dones:         List[bool]          = []
        self.all_hands:     List[torch.Tensor] = []
        self.advantages:    Optional[torch.Tensor] = None
        self.returns:       Optional[torch.Tensor] = None

    def add(self, flat_obs, legal_actions, action, log_prob, reward, value, done,
            all_hands=None):
        self.flat_obs.append(torch.tensor(flat_obs,      dtype=torch.float32))
        self.legal_actions.append(torch.tensor(legal_actions, dtype=torch.float32))
        self.actions.append(torch.tensor(action,         dtype=torch.int64))
        self.log_probs.append(log_prob.detach().cpu() if hasattr(log_prob, 'detach')
                              else torch.tensor(log_prob))
        self.rewards.append(float(reward))
        self.values.append(value.detach().cpu()   if hasattr(value,    'detach')
                           else torch.tensor(value))
        self.dones.append(bool(done))
        if all_hands is not None:
            self.all_hands.append(torch.tensor(all_hands, dtype=torch.float32))

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        values_np  = np.array([v.item() for v in self.values], dtype=np.float32)

        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else values_np[t + 1]
            # dones[t] belongs to the transition stored at t.  Looking at
            # dones[t + 1] leaks the first value of the next episode across a
            # terminal boundary and cuts off the terminal reward one step early.
            non_terminal = 1.0 - float(self.dones[t])
            delta     = (self.rewards[t] + gamma * next_val * non_terminal
                         - values_np[t])
            last_gae  = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns    = torch.tensor(returns,    dtype=torch.float32)

    def __len__(self):
        return len(self.actions)

    def get_batches(self, batch_size: int):
        n       = len(self.actions)
        indices = np.random.permutation(n)
        device  = self.device

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:          # skip incomplete last batch (std() needs ≥2 samples)
                continue
            batch = {
                'flat_obs':      torch.stack([self.flat_obs[i]      for i in idx]).to(device),
                'legal_actions': torch.stack([self.legal_actions[i] for i in idx]).to(device),
                'actions':       torch.stack([self.actions[i]       for i in idx]).to(device),
                'old_log_probs': torch.stack([self.log_probs[i]     for i in idx]).to(device),
                'old_values':    torch.stack([self.values[i]        for i in idx]).to(device),
                'advantages':    self.advantages[idx].to(device),
                'returns':       self.returns[idx].to(device),
            }
            if self.all_hands:
                batch['all_hands'] = torch.stack(
                    [self.all_hands[i] for i in idx]).to(device)
            yield batch


# ==============================================================================
# SubgameTrainer
# ==============================================================================

class SubgameTrainer:
    """See the formal README for the current behavior contract."""

    def __init__(self, env, config: SubgameConfig,
                 reward_stats: Optional[RunningStats] = None):
        self.env     = env
        self.config  = config
        self.device  = config.device
        self.active_players = config.active_players or list(range(NUM_PLAYERS))

        # BeliefNet is a training-only communication critic.  The deployed
        # policy always consumes the standard OpenSpiel observation.
        _obs_dim = OBS_DIM
        mappo_cfg = MAPPOConfig(
            lr              = config.lr,
            gamma           = config.gamma,
            gae_lambda      = config.gae_lambda,
            clip_ratio      = config.clip_ratio,
            num_epochs      = config.num_epochs,
            batch_size      = config.batch_size,
            entropy_coef    = config.entropy_coef,
            value_coef      = config.value_coef,
            max_grad_norm   = config.max_grad_norm,
            device          = config.device,
            critic_lr_ratio = config.critic_lr_ratio,
            hidden_dim      = config.hidden_dim,
            obs_dim         = _obs_dim,
            actor_belief_conditioned = config.belief_conditioned,
            actor_belief_hidden_dim  = config.hidden_dim,
        )
        self.agent = MAPPOAgent(mappo_cfg)

        # ── Per-player flat rollout buffers ────────────────────────────────
        self.buffers: Dict[int, FlatRolloutBuffer] = {
            p: FlatRolloutBuffer(self.device) for p in range(NUM_PLAYERS)
        }

        # Optional training-time semantic model.
        self.belief_net:     Optional[BeliefNetwork]     = None
        self.dual_info:      Optional[DualInfoComputer]  = None
        self.belief_optimizer = None

        _need_belief = config.use_info_bonus or config.belief_conditioned
        if _need_belief:
            self.belief_net  = BeliefNetwork(hidden_dim=config.hidden_dim).to(self.device)
            if config.use_info_bonus:
                self.dual_info = DualInfoComputer(self.belief_net, beta=config.beta)
            if not config.freeze_belief:
                self.belief_optimizer = torch.optim.Adam(
                    self.belief_net.parameters(), lr=config.belief_update_lr)
            self.belief_net.eval()
            for parameter in self.belief_net.parameters():
                parameter.requires_grad_(False)

        # A single calibration constant is computed before training and then
        # shared by B/C.  It never follows the policy during training.
        self.info_scale_factor: Optional[float] = None
        self.info_scale_metadata: Optional[dict] = None

        self.reward_stats: RunningStats = reward_stats or RunningStats()

        # ── FSP pool ───────────────────────────────────────────────────────
        # P126: fsp_pool_size=0 → unlimited (use 999999 as practical max)
        _pool_max = config.fsp_pool_size if config.fsp_pool_size > 0 else 999999
        self.fsp_pool = FSPPool(max_size=_pool_max)

        self.bc_actors: Optional[Dict[int, MLPPolicyNetwork]] = None

        self._fsp_actor_cache: dict = {}  # role -> MLPPolicyNetwork
        self._fsp_cache_source: Optional[dict] = None

        # Keyed on (hands_rm.tobytes(), dealer, history_tuple) → flat obs array
        # Cleared at the start of each collect_episodes_batch call.
        self._obs_cache:       dict = {}
        self._obs_state_cache: dict = {}  # same key → pyspiel.State (for incremental extend)

        self.log: List[dict] = []
        self._global_step = 0
        self._vl_history: List[float] = []
        self._fsp_seeded: bool = False

    def initialize_actor_beliefs_from_judge(self) -> None:
        """Warm-start every deployed decoder from the separate frozen Judge."""
        if self.belief_net is None or not self.config.belief_conditioned:
            return
        for player in range(NUM_PLAYERS):
            self.agent.get_actor(player).initialize_belief_from_judge(
                self.belief_net
            )

    def _prepare_fsp_cache(self, fsp_sd: Optional[dict]) -> None:
        """Make cached role actors belong to one and only one snapshot."""
        if self._fsp_cache_source is not fsp_sd:
            self._fsp_actor_cache.clear()
            # Keep the source alive as well as comparing by identity, so Python
            # cannot recycle an id and make a new snapshot look like the old one.
            self._fsp_cache_source = fsp_sd

    # ======================================================================
    # ======================================================================

    def run_bc_warmup(self, num_samples: int = None, num_epochs: int = None,
                      lr: float = None):
        """See the formal README for the current behavior contract."""
        from subgames.competitive_env import generate_rule_based_bc_data

        num_samples = num_samples or self.config.bc_warmup_samples
        num_epochs  = num_epochs  or self.config.bc_warmup_epochs
        lr          = lr          or self.config.bc_warmup_lr

        print(f"\n[BC Warmup] Generating {num_samples} rule-based samples...")
        data = generate_rule_based_bc_data(self.env, num_samples)

        if not data:
            print("[BC Warmup] No data generated, skipping.")
            return

        flat_obs_np = np.stack([d['flat_obs'] for d in data])  # (N, 480)
        actions_np  = np.array([d['action']   for d in data], dtype=np.int64)
        legal_np    = np.ones((len(data), NUM_BIDS), dtype=np.float32)

        flat_t   = torch.tensor(flat_obs_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.int64)
        legal_t  = torch.tensor(legal_np,    dtype=torch.float32)

        print(f"[BC Warmup] Training {num_epochs} epochs on {len(data)} samples...")

        for player in [NORTH, SOUTH, EAST, WEST]:
            actor = self.agent.get_actor(player)
            opt   = torch.optim.Adam(actor.parameters(), lr=lr)

            for epoch in range(num_epochs):
                idx  = np.random.permutation(len(data))
                loss_sum = 0.0
                n_batches = 0

                for start in range(0, len(data), self.config.batch_size):
                    b_idx = idx[start:start + self.config.batch_size]
                    b_flat   = flat_t[b_idx].to(self.device)
                    b_legal  = legal_t[b_idx].to(self.device)
                    b_act    = actions_t[b_idx].to(self.device)

                    logits = actor(b_flat, b_legal)
                    loss   = F.cross_entropy(logits, b_act)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                    opt.step()

                    loss_sum += loss.item()
                    n_batches += 1

                avg_loss = loss_sum / max(1, n_batches)

            name = {NORTH:'N', SOUTH:'S', EAST:'E', WEST:'W'}[player]
            print(f"  Player {name}: epoch {num_epochs} loss={avg_loss:.4f}")

        print("[BC Warmup] Done.")

    # ======================================================================
    # KL Anchor
    # ======================================================================

    def set_bc_anchor(self, agent_or_state_dict):
        """See the formal README for the current behavior contract."""
        def _frozen_copy(state_dict):
            net = MLPPolicyNetwork(obs_dim=OBS_DIM,
                                   hidden_dim=self.config.hidden_dim,
                                   belief_conditioned=self.config.belief_conditioned,
                                   belief_hidden_dim=self.config.hidden_dim).to(self.device)
            net.load_state_dict(state_dict)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            return net

        if isinstance(agent_or_state_dict, MAPPOAgent):
            src = agent_or_state_dict.model
        else:
            src = self.agent.model

        self.bc_actors = {
            NORTH: _frozen_copy(src.actor_n.state_dict()),
            SOUTH: _frozen_copy(src.actor_s.state_dict()),
            EAST:  _frozen_copy(src.actor_e.state_dict()),
            WEST:  _frozen_copy(src.actor_w.state_dict()),
        }
        print("[KL Anchor] BC anchor set for all 4 players.")

    def _get_kl_lambda(self, round_idx: int) -> float:
        cfg = self.config
        if cfg.kl_anneal_frac <= 0:
            # No annealing — fixed at start value
            return cfg.kl_lambda_start
        # Anneal over first kl_anneal_frac of rounds, then hold at end value
        anneal_rounds = max(1, int(cfg.num_rounds * cfg.kl_anneal_frac))
        progress = min(1.0, round_idx / anneal_rounds)
        return cfg.kl_lambda_start + (cfg.kl_lambda_end - cfg.kl_lambda_start) * progress

    # ======================================================================
    # FSP
    # ======================================================================

    def _maybe_add_to_fsp(self, round_idx: int):
        """
        P126: Quality-gated FSP pool insertion.

        Every fsp_add_interval rounds, evaluate the current snapshot vs SL
        on a small number of deals. Only add to pool if:
          - median auction length <= fsp_gate_max_auction_len
          - doubled contract rate  <= fsp_gate_max_double_rate

        This prevents chaos bidding snapshots from polluting the pool.
        Round 0 is always admitted (initial SL weights, no training yet).
        """
        if self.config.self_play:
            return
        if round_idx % self.config.fsp_add_interval != 0:
            return

        cfg = self.config

        # ── Quality gate evaluation (skip round 0 — not yet trained) ─────
        if cfg.fsp_quality_gate and self._fsp_seeded and round_idx > 0:
            # Use the SL permanent member (first entry) as evaluation opponent
            sl_sd = self.fsp_pool._permanent[0] if self.fsp_pool._permanent else None
            if sl_sd is not None:
                gate_ok, gate_info = self._fsp_quality_eval(sl_sd, cfg.fsp_gate_eval_deals)
                med_len = gate_info['median_auction_len']
                dbl_rate = gate_info['double_rate']
                if not gate_ok:
                    print(f"  [FSP] Quality gate REJECTED: "
                          f"median_len={med_len:.1f} (max={cfg.fsp_gate_max_auction_len}), "
                          f"dbl_rate={dbl_rate:.2f} (max={cfg.fsp_gate_max_double_rate:.2f}). "
                          f"Snapshot NOT added to pool.")
                    return
                else:
                    print(f"  [FSP] Quality gate PASSED: "
                          f"median_len={med_len:.1f}, dbl_rate={dbl_rate:.2f}")

        self.fsp_pool.add(self.agent)
        print(f"  [FSP] Pool size: {len(self.fsp_pool)}")

    def _fsp_quality_eval(self, sl_sd: dict, num_deals: int) -> tuple:
        """
        P126: Quick evaluation of current agent vs SL to check for chaos bidding.

        Plays num_deals deals with current agent (NS) vs SL (sl_sd as EW opponent).
        Measures auction length and doubled contract frequency.
        Returns (passed: bool, info: dict).
        """
        auction_lengths = []
        doubled_count = 0

        # Collect episodes using current agent vs SL opponent
        eps, _ = self._collect_episodes_batch(
            num_deals, train_side='NS', fsp_sd=sl_sd,
            batch_size=min(32, num_deals),
            use_belief_prior=False)

        for ep in eps:
            # Each step in ep is one bid AFTER the fixed prefix (1H-1S).
            # Auction length = prefix (2) + number of steps in episode.
            auction_lengths.append(len(ep) + 2)

            # Detect doubled contract: in bridge, auction ends with 3 consecutive
            # passes. The contract is doubled if the last non-Pass action before
            # the terminal passes was BID_DOUBLE (=1), and not followed by
            # BID_REDOUBLE (=2).
            actions = [step['action'] for step in ep]
            # Strip trailing passes to find the last substantive bid
            last_substantive = None
            for a in reversed(actions):
                if a != BID_PASS:
                    last_substantive = a
                    break
            if last_substantive == 1 or last_substantive == 2:  # BID_DOUBLE or BID_REDOUBLE
                doubled_count += 1

        if not auction_lengths:
            return True, {'median_auction_len': 0, 'double_rate': 0, 'num_deals': 0}

        med_len = statistics.median(auction_lengths)
        dbl_rate = doubled_count / len(auction_lengths)

        cfg = self.config
        passed = (med_len <= cfg.fsp_gate_max_auction_len
                  and dbl_rate <= cfg.fsp_gate_max_double_rate)

        return passed, {
            'median_auction_len': med_len,
            'double_rate': dbl_rate,
            'num_deals': len(auction_lengths),
        }

    def _apply_fsp_opponent(self):
        """
        P126: Weighted FSP sampling with SL minimum probability.

        With probability fsp_sl_sample_prob, always sample the SL permanent
        member. Otherwise, uniform random from the full pool (permanent + FIFO).
        This ensures the agent always trains against SL at least 30% of the time,
        preventing overfitting to a static set of degenerate opponents.
        """
        if self.fsp_pool.is_empty():
            return None

        cfg = self.config
        # P126: weighted sampling — SL permanent member gets guaranteed floor
        if (cfg.fsp_sl_sample_prob > 0
                and self.fsp_pool._permanent
                and np.random.rand() < cfg.fsp_sl_sample_prob):
            # Sample from permanent members (typically just SL)
            return random.choice(self.fsp_pool._permanent)

        # Otherwise uniform from full pool
        return self.fsp_pool.sample()

    # ======================================================================
    # Critic Warmup
    # ======================================================================

    def critic_warmup(self, num_deals: int = None, num_epochs: int = None):
        """See the formal README for the current behavior contract."""
        num_deals  = num_deals  or self.config.critic_prewarm_deals
        num_epochs = num_epochs or self.config.critic_prewarm_epochs
        conv_tol   = self.config.critic_prewarm_conv_tol

        half = num_deals // 2
        print(f"[Critic Warmup] Collecting {num_deals} deals (NS:{half} + EW:{half}, batch={self.config.deals_per_step})...")
        ns_eps, _ = self._collect_episodes_batch(half, train_side='NS', fsp_sd=None,
                                               batch_size=self.config.deals_per_step,
                                               skip_dual_table=False)
        ew_eps, _ = self._collect_episodes_batch(half, train_side='EW', fsp_sd=None,
                                               batch_size=self.config.deals_per_step,
                                               skip_dual_table=False)
        self._store_episodes(ns_eps)
        self._store_episodes(ew_eps)

        for player in self.active_players:
            buf = self.buffers[player]
            if len(buf) < 2:
                buf.reset()
                continue

            critic     = self.agent.get_critic(player)
            critic_opt = self.agent.get_critic_optimizer(player)

            # Bootstrap last value
            with torch.no_grad():
                fo = buf.flat_obs[-1].unsqueeze(0).to(self.device)
                ah = (buf.all_hands[-1].unsqueeze(0).to(self.device)
                      if buf.all_hands else None)
                last_val = critic(fo, ah).item()

            buf.compute_returns_and_advantages(
                last_val, self.agent.config.gamma, self.agent.config.gae_lambda)

            prev_loss = None
            for epoch in range(num_epochs):
                epoch_loss, n_batches = 0.0, 0
                for batch in buf.get_batches(self.agent.config.batch_size):
                    vals = critic(batch['flat_obs'], batch.get('all_hands'))
                    loss = F.mse_loss(vals, batch['returns'])
                    critic_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
                    critic_opt.step()
                    epoch_loss += loss.item()
                    n_batches  += 1
                epoch_loss /= max(1, n_batches)
                if prev_loss is not None and prev_loss > 1e-8:
                    if abs(epoch_loss - prev_loss) / prev_loss < conv_tol:
                        print(f"  [Critic Warmup] Player {player} converged @ epoch {epoch+1}")
                        break
                prev_loss = epoch_loss
            buf.reset()

    # ======================================================================
    # Dual-table IMP reward (P55)
    # ======================================================================

    @staticmethod
    def _extract_receiver_belief_data(episodes: List[List[dict]]):
        """Flatten partner and opponent receiver views into supervised samples.

        Every sample predicts the bidder's hand.  The observation belongs to a
        receiver and therefore includes that receiver's private hand.
        """
        observations = []
        target_positions = []
        targets = []
        for episode in episodes:
            for step in episode:
                if not step.get('_rinfo'):
                    continue
                for key in ('partner_obs_after', 'opponent_obs_after'):
                    observations.append(step[key])
                    target_positions.append(step['target_pos'])
                    targets.append(step['belief_target'])
        if not observations:
            return None
        return (
            torch.tensor(np.stack(observations), dtype=torch.float32),
            torch.tensor(target_positions, dtype=torch.long),
            torch.tensor(np.stack(targets), dtype=torch.float32),
        )

    # ======================================================================
    # BeliefNet pretraining
    # ======================================================================

    def pretrain_belief(self, num_rounds: int = 5, deals_per_round: int = 2000,
                        epochs_per_round: int = 5, max_epochs: int = 300):
        """See the formal README for the current behavior contract."""
        if self.belief_net is None:
            return

        for parameter in self.belief_net.parameters():
            parameter.requires_grad_(True)

        total_deals = num_rounds * deals_per_round
        print(f"\n[Belief Pretrain] Collecting {total_deals} deals (1 pass)...")

        all_episodes, _ = self._collect_episodes_batch(
            total_deals, train_side='NS',
            fsp_sd=None, batch_size=self.config.deals_per_step,
            skip_dual_table=True, use_belief_prior=True)

        tensors = self._extract_receiver_belief_data(all_episodes)
        if tensors is None:
            print("[Belief Pretrain] No belief data collected, skipping.")
            self.belief_net.eval()
            for parameter in self.belief_net.parameters():
                parameter.requires_grad_(False)
            return
        obs_all, tp_all, tgt_all = tensors
        N = len(obs_all)
        print(f"[Belief Pretrain] Dataset: {N} samples. Training to convergence...")

        # 90/10 train/val split
        split    = int(N * 0.9)
        perm_all = np.random.permutation(N)
        tr_idx   = perm_all[:split]
        va_idx   = perm_all[split:]

        criterion = None
        optimizer = torch.optim.Adam(
            self.belief_net.parameters(), lr=self.config.belief_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        bs         = min(512, split)
        best_val   = float('inf')
        patience   = 15
        no_improve = 0

        self.belief_net.train()
        for epoch in range(1, max_epochs + 1):
            # Train
            perm     = np.random.permutation(split)
            tr_loss  = 0.0; nb = 0
            for s in range(0, split, bs):
                idx    = tr_idx[perm[s:s+bs]]
                loss   = self.belief_net.compute_loss(
                    obs_all[idx].to(self.device),
                    tp_all[idx].to(self.device),
                    tgt_all[idx].to(self.device))
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    self.belief_net.parameters(), self.config.max_grad_norm)
                optimizer.step()
                tr_loss += loss.item(); nb += 1
            tr_loss /= max(1, nb)

            # Validate
            self.belief_net.eval()
            with torch.no_grad():
                val_loss = self.belief_net.compute_loss(
                    obs_all[va_idx].to(self.device),
                    tp_all[va_idx].to(self.device),
                    tgt_all[va_idx].to(self.device)).item()
                probs    = self.belief_net.get_probs(
                    obs_all[va_idx].to(self.device),
                    tp_all[va_idx].to(self.device))
                acc      = belief_accuracy(probs, tgt_all[va_idx].to(self.device))
            self.belief_net.train()

            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']

            print(f"  epoch {epoch:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"honor={acc['honor_acc']:.3f}  length={acc['length_acc']:.3f}  "
                  f"overall={acc['overall_acc']:.3f}  lr={lr_now:.2e}")

            # Early stopping
            if val_loss < best_val - 1e-4:
                best_val   = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [Belief Pretrain] Early stop @ epoch {epoch} "
                          f"(no improve for {patience} epochs)")
                    break

        self.belief_net.eval()
        for parameter in self.belief_net.parameters():
            parameter.requires_grad_(False)
        print(f"[Belief Pretrain] Done. best_val_loss={best_val:.4f}")

        # ── P97: Compute Fisher Information Matrix for EWC ────────────
        if self.config.use_ewc:
            print(f"[Belief Pretrain] Computing Fisher for EWC "
                  f"(λ_ewc={self.config.ewc_lambda}, samples={self.config.ewc_fisher_samples})...")
            self.belief_net.compute_fisher(
                obs_all, tp_all, tgt_all,
                num_samples=self.config.ewc_fisher_samples)

        # ── P97b: Save pretrain data for replay mixing ─────────────
        # Store a subsample of pretrain data to mix into on-policy updates.
        # This directly prevents catastrophic forgetting by keeping pretrain
        # loss in the training objective (data-level protection, not weight-level).
        replay_n = min(10000, N)
        replay_idx = np.random.permutation(N)[:replay_n]
        self._pretrain_replay = {
            'obs': obs_all[replay_idx].clone(),
            'tp':  tp_all[replay_idx].clone(),
            'tgt': tgt_all[replay_idx].clone(),
        }
        print(f"[Belief Pretrain] Saved {replay_n} pretrain samples for replay mixing.")

    # ======================================================================
    # On-Policy Belief Update (P93)
    # ======================================================================

    def update_belief_on_policy(self, episodes: List[List[dict]]) -> Optional[float]:
        """See the formal README for the current behavior contract."""
        if self.belief_net is None or self.belief_optimizer is None:
            return None

        tensors = self._extract_receiver_belief_data(episodes)
        if tensors is None:
            return None
        obs, tp, tgt = tensors
        if len(obs) < 100:
            return None

        N = len(obs)
        split = int(N * 0.9)
        perm = np.random.permutation(N)
        tr_idx = perm[:split]
        va_idx = perm[split:]

        # ── Check if pretrain replay data is available ──
        has_replay = hasattr(self, '_pretrain_replay') and self._pretrain_replay is not None
        if has_replay:
            rp = self._pretrain_replay
            rp_n = rp['obs'].size(0)

        self.belief_net.train()
        bs = min(512, split)
        half_bs = bs // 2  # half for on-policy, half for replay
        best_val = float('inf')
        no_improve = 0
        final_loss = 0.0

        for epoch in range(self.config.belief_update_epochs):
            ep_perm = np.random.permutation(split)
            rp_perm = np.random.permutation(rp_n) if has_replay else None
            tl = 0.0; tl_rp = 0.0; tl_ewc = 0.0; nb = 0
            rp_ptr = 0

            for s in range(0, split, bs):
                idx = tr_idx[ep_perm[s:s+bs]]

                # ── On-policy loss ──
                loss_op = self.belief_net.compute_loss(
                    obs[idx].to(self.device),
                    tp[idx].to(self.device),
                    tgt[idx].to(self.device))

                # ── Pretrain replay loss (P97b) ──
                loss_rp = torch.tensor(0.0, device=self.device)
                if has_replay:
                    # Sample a batch of pretrain data (same size as on-policy batch)
                    rp_size = min(len(idx), rp_n)
                    if rp_ptr + rp_size > rp_n:
                        rp_perm = np.random.permutation(rp_n)
                        rp_ptr = 0
                    rp_idx = rp_perm[rp_ptr:rp_ptr + rp_size]
                    rp_ptr += rp_size

                    loss_rp = self.belief_net.compute_loss(
                        rp['obs'][rp_idx].to(self.device),
                        rp['tp'][rp_idx].to(self.device),
                        rp['tgt'][rp_idx].to(self.device))

                # ── Combined loss: 80% on-policy + 20% replay (P97c) ──
                # On-policy dominant so belief tracks current policy,
                # replay minority prevents catastrophic forgetting of pretrain.
                loss = 0.8 * loss_op + 0.2 * loss_rp if has_replay else loss_op

                # ── Optional EWC penalty ──
                ewc_loss_val = 0.0
                if self.config.use_ewc and self.belief_net.has_ewc:
                    ewc_loss = self.belief_net.ewc_penalty()
                    loss = loss + (self.config.ewc_lambda / 2.0) * ewc_loss
                    ewc_loss_val = ewc_loss.item()

                self.belief_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.belief_net.parameters(), self.config.max_grad_norm)
                self.belief_optimizer.step()
                tl += loss_op.item(); tl_rp += loss_rp.item(); tl_ewc += ewc_loss_val; nb += 1
            train_loss = tl / max(1, nb)

            # Validation (on-policy data only — this is what matters for r_info)
            self.belief_net.eval()
            with torch.no_grad():
                val_loss = self.belief_net.compute_loss(
                    obs[va_idx].to(self.device),
                    tp[va_idx].to(self.device),
                    tgt[va_idx].to(self.device)).item()
            self.belief_net.train()

            final_loss = val_loss
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 2:  # early stop after 2 epochs no improvement
                    break

        rp_str = f" replay_loss={tl_rp/max(1,nb):.4f}" if has_replay else ""
        ewc_str = f" ewc_pen={tl_ewc/max(1,nb):.6f}" if (self.config.use_ewc and self.belief_net.has_ewc) else ""
        print(f"  [Belief Update] {N} samples, "
              f"{epoch+1} epochs, val_loss={final_loss:.4f}{rp_str}{ewc_str}")
        return final_loss

    # ======================================================================
    # Episode Collection
    # ======================================================================

    def _collect_episodes_batch(
        self,
        num_deals:        int,
        train_side:       str,
        fsp_sd:           Optional[dict] = None,
        batch_size:       int = 32,
        skip_dual_table:  bool = False,   # P55: skip swapped-table rollout (critic warmup)
        use_belief_prior: bool = False,   # P98b: use prior features instead of belief net
    ) -> List[List[dict]]:
        """See the formal README for the current behavior contract."""
        from subgames.competitive_env import CompetitiveSubgameEnv

        # P109: clear obs cache at start of each batch to bound memory usage
        self._obs_cache.clear()
        self._obs_state_cache.clear()
        self._prepare_fsp_cache(fsp_sd)

        envs = [CompetitiveSubgameEnv.__new__(CompetitiveSubgameEnv)
                for _ in range(batch_size)]
        for e in envs:
            e.loader          = self.env.loader
            e.env             = __import__('env', fromlist=['BridgeBiddingEnv']).BridgeBiddingEnv(60)
            e.max_history_len = 60
            e.dealer          = NORTH
            e._sampled_dealer  = NORTH
            e._is_constrained_data = self.env._is_constrained_data
            e._filtered_deals      = self.env._filtered_deals
            e._current_hands = None; e._current_dd = None
            e._vulnerability = (False, False); e.history_int = []

        slot_obs    = [None] * batch_size
        slot_hist   = [None] * batch_size
        slot_dealer = [NORTH] * batch_size   # per-slot dealer (set on reset)
        slot_ep     = [[]   for _ in range(batch_size)]
        slot_done   = [True] * batch_size
        all_episodes: List[List[dict]] = []
        _pending_rewards: List[Tuple]   = []
        collected = 0

        def _reset(i):
            # P122: randomize vulnerability for each new deal
            _vul_choices = [(False, False), (True, False),
                            (False, True),  (True, True)]
            _vul = _vul_choices[np.random.randint(4)]
            hands, dd_table = envs[i].generate_deal()
            dealer = envs[i]._sampled_dealer
            obs = envs[i].reset(
                hands,
                dd_table,
                vulnerability=_vul,
                dealer=dealer,
            )
            slot_hist[i]   = list(envs[i].history_int)
            slot_dealer[i] = envs[i].dealer
            slot_ep[i]     = []
            slot_done[i]   = False
            return obs

        slot_obs = [_reset(i) for i in range(batch_size)]

        while collected < num_deals:
            active = [i for i in range(batch_size) if not slot_done[i]]
            if not active:
                break

            from collections import defaultdict
            groups = defaultdict(list)  # role_key -> [slot_idx]
            for i in active:
                player = envs[i].current_player
                is_train = ((train_side == 'NS' and player in (NORTH, SOUTH)) or
                            (train_side == 'EW' and player in (EAST,  WEST)))
                role = {NORTH:'actor_n', EAST:'actor_e',
                        SOUTH:'actor_s', WEST:'actor_w'}[player]
                key = f"train_{role}" if is_train else f"fsp_{role}"
                groups[key].append(i)

            actions_map = {}  # slot -> (action, log_prob, value, flat, legal, is_train)

            for key, slots in groups.items():
                is_train = key.startswith('train_')
                role = key.split('_', 1)[1]  # actor_n/s/e/w

                if is_train:
                    actor  = getattr(self.agent.model, role)
                    critic = getattr(self.agent.model,
                                     role.replace('actor','critic'))
                else:
                    if fsp_sd and role in fsp_sd:
                        if role not in self._fsp_actor_cache:
                            net = MLPPolicyNetwork(
                                obs_dim=OBS_DIM,
                                hidden_dim=self.config.hidden_dim,
                                belief_conditioned=self.config.belief_conditioned,
                                belief_hidden_dim=self.config.hidden_dim).to(self.device)
                            net.load_state_dict(
                                {k: v.to(self.device) for k, v in fsp_sd[role].items()})
                            net.eval()
                            self._fsp_actor_cache[role] = net
                        actor = self._fsp_actor_cache[role]
                    else:
                        actor = getattr(self.agent.model, role)
                    critic = getattr(self.agent.model,
                                     role.replace('actor','critic'))

                obs571_batch = np.stack([
                    self._encode_for_actor(
                        slot_obs[i], slot_dealer[i], slot_hist[i],
                        envs[i].current_player,
                        all_hands=envs[i]._current_hands,
                        use_prior=True,
                        vulnerability=envs[i]._vulnerability)[:OBS_DIM]
                    for i in slots])
                legal_batch = np.stack([slot_obs[i]['legal_actions'] for i in slots])
                ah_batch    = np.stack([envs[i]._current_hands for i in slots])

                flat_batch = obs571_batch

                flat_t  = torch.tensor(flat_batch,  dtype=torch.float32).to(self.device)
                legal_t = torch.tensor(legal_batch, dtype=torch.float32).to(self.device)
                ah_t    = torch.tensor(ah_batch,    dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    actions, log_probs, _ = actor.get_action(flat_t, legal_t)
                    values = critic(flat_t, ah_t)

                for j, i in enumerate(slots):
                    actions_map[i] = (actions[j].item(), log_probs[j].cpu(),
                                      values[j].cpu(), flat_batch[j],
                                      legal_batch[j], is_train)

            for i in active:
                action, log_prob, value, flat_obs, legal_actions, is_train = actions_map[i]
                player    = envs[i].current_player
                all_hands = envs[i]._current_hands

                _dealer_i      = slot_dealer[i]
                opener_seats_i = {_dealer_i, (_dealer_i + 2) % 4}
                obs_571 = flat_obs[:OBS_DIM]
                step = {
                    'flat_obs': flat_obs, 'legal_actions': legal_actions,
                    'action': action, 'log_prob': log_prob, 'value': value,
                    'reward': 0.0, 'done': False,
                    'all_hands': all_hands.copy(), 'player': player,
                    'is_training_side': is_train,
                    'is_opener': player in opener_seats_i,
                    'obs_571': obs_571,
                    'dealer': _dealer_i,
                    'vulnerability': envs[i]._vulnerability,
                }

                if self.config.use_info_bonus and is_train:
                    partner = (player + 2) % 4
                    opponent = (player + 1) % 4
                    step.update({
                        '_rinfo': True,
                        'target_pos': physical_to_openspiel_player(
                            player, _dealer_i
                        ),
                        'belief_target': hand_to_belief_target(all_hands[player]),
                        'partner_pos': partner,
                        'opponent_pos': opponent,
                        'partner_obs_before': self._encode_for_actor(
                            slot_obs[i], _dealer_i, slot_hist[i], partner,
                            all_hands=all_hands,
                            vulnerability=envs[i]._vulnerability,
                        )[:OBS_DIM],
                        'opponent_obs_before': self._encode_for_actor(
                            slot_obs[i], _dealer_i, slot_hist[i], opponent,
                            all_hands=all_hands,
                            vulnerability=envs[i]._vulnerability,
                        )[:OBS_DIM],
                    })

                slot_hist[i].append(action)

                # Encode the public state immediately after this action.  This
                # isolates information revealed by this bid; using the player's
                # next turn would also include three intervening actions.
                obs_next, reward, done, info = envs[i].step(action)
                if step.get('_rinfo'):
                    step['partner_obs_after'] = self._encode_for_actor(
                        obs_next,
                        _dealer_i,
                        slot_hist[i],
                        step['partner_pos'],
                        all_hands=all_hands,
                        vulnerability=envs[i]._vulnerability,
                    )[:OBS_DIM]
                    step['opponent_obs_after'] = self._encode_for_actor(
                        obs_next,
                        _dealer_i,
                        slot_hist[i],
                        step['opponent_pos'],
                        all_hands=all_hands,
                        vulnerability=envs[i]._vulnerability,
                    )[:OBS_DIM]
                step['reward'] = reward
                step['done']   = done
                slot_ep[i].append(step)
                slot_obs[i] = obs_next

                if done:
                    all_episodes.append(slot_ep[i])
                    if not skip_dual_table:
                        _pending_rewards.append((
                            len(all_episodes) - 1,
                            envs[i]._current_dd.copy(),
                            slot_dealer[i],
                            envs[i]._vulnerability,
                            envs[i].env.state.final_contract,
                        ))
                    collected += 1
                    slot_done[i] = True
                    if collected < num_deals:
                        slot_obs[i] = _reset(i)



        if not skip_dual_table and _pending_rewards:
            for ep_idx, dd, dealer, vul, contract in _pending_rewards:
                score_ns  = self.env._compute_score_ns(contract, dd, vul)
                score_opt = self.env._compute_dds_optimal_score_ns(dd, vul)
                imp_ns    = float(score_to_imp(score_ns - score_opt))
                imp_ew    = -imp_ns

                last_step_idx: Dict[int, int] = {}
                for s_idx, s in enumerate(all_episodes[ep_idx]):
                    last_step_idx[s['player']] = s_idx

                for player, s_idx in last_step_idx.items():
                    s = all_episodes[ep_idx][s_idx]
                    s['done']   = True
                    s['reward'] = imp_ns if player in (NORTH, SOUTH) else imp_ew

        # Build contiguous arrays for batched information-reward inference.
        result_eps = all_episodes[:num_deals]
        if self.config.use_info_bonus:
            # Count _rinfo steps to pre-allocate (avoids np.stack on list-of-arrays)
            n_rinfo = sum(1 for ep in result_eps
                          for step in ep if step.get('_rinfo'))
            if n_rinfo > 0:
                partner_before = np.empty((n_rinfo, OBS_DIM), dtype=np.float32)
                partner_after = np.empty((n_rinfo, OBS_DIM), dtype=np.float32)
                opponent_before = np.empty((n_rinfo, OBS_DIM), dtype=np.float32)
                opponent_after = np.empty((n_rinfo, OBS_DIM), dtype=np.float32)
                targets = np.empty((n_rinfo, BELIEF_DIM), dtype=np.float32)
                target_positions = np.empty(n_rinfo, dtype=np.int64)
                ep_step_list: List[Tuple] = []
                ptr = 0

                for ep_idx, ep in enumerate(result_eps):
                    for s_idx, step in enumerate(ep):
                        if not step.get('_rinfo'):
                            continue
                        partner_before[ptr] = step['partner_obs_before']
                        partner_after[ptr] = step['partner_obs_after']
                        opponent_before[ptr] = step['opponent_obs_before']
                        opponent_after[ptr] = step['opponent_obs_after']
                        targets[ptr] = step['belief_target']
                        target_positions[ptr] = step['target_pos']
                        ep_step_list.append((ep_idx, s_idx))
                        ptr += 1

                _rinfo_data = {
                    'partner_before': partner_before[:ptr],
                    'partner_after': partner_after[:ptr],
                    'opponent_before': opponent_before[:ptr],
                    'opponent_after': opponent_after[:ptr],
                    'target': targets[:ptr],
                    'target_pos': target_positions[:ptr],
                    'ep_step': ep_step_list,
                }
            else:
                _rinfo_data = None
        else:
            _rinfo_data = None

        return result_eps, _rinfo_data

    def _get_fsp_actor(self, player: int, fsp_sd: dict) -> MLPPolicyNetwork:
        """See the formal README for the current behavior contract."""
        role = {NORTH:'actor_n', EAST:'actor_e',
                SOUTH:'actor_s', WEST:'actor_w'}[player]
        if role not in fsp_sd:
            return self.agent.get_actor(player)
        self._prepare_fsp_cache(fsp_sd)
        if role not in self._fsp_actor_cache:
            net = MLPPolicyNetwork(obs_dim=OBS_DIM,
                                   hidden_dim=self.config.hidden_dim,
                                   belief_conditioned=self.config.belief_conditioned,
                                   belief_hidden_dim=self.config.hidden_dim).to(self.device)
            net.load_state_dict({k: v.to(self.device) for k, v in fsp_sd[role].items()})
            net.eval()
            self._fsp_actor_cache[role] = net
        return self._fsp_actor_cache[role]

    def _store_episodes(self, episodes: List[List[dict]]):
        """See the formal README for the current behavior contract."""
        for ep in episodes:
            for step in ep:
                p = step['player']
                if p not in self.active_players:
                    continue
                buf = self.buffers[p]
                buf.add(
                    flat_obs      = step['flat_obs'],
                    legal_actions = step['legal_actions'],
                    action        = step['action'],
                    log_prob      = step['log_prob'],
                    reward        = step['reward'],
                    value         = step['value'],
                    done          = step['done'],
                    all_hands     = step.get('all_hands'),
                )

    def _encode_for_actor(
        self,
        obs: dict,
        dealer: int,
        history_int: list,
        player: int,
        all_hands: Optional[np.ndarray] = None,
        belief_net: Optional[BeliefNetwork] = None,
        opponent_bn: Optional[BeliefNetwork] = None,
        use_prior: bool = False,
        vulnerability: tuple = None,
    ) -> np.ndarray:
        """See the formal README for the current behavior contract."""
        if vulnerability is None:
            vulnerability = getattr(self.env, '_vulnerability', (False, False))
        hands_sm = all_hands
        if hands_sm is not None:
            hands_rm = convert_hands_suit_to_rank(hands_sm)

            # Build cache key from (hands bytes, dealer, history, vul)
            hist_tuple = tuple(history_int)
            cache_key  = (hands_rm.tobytes(), dealer, hist_tuple, vulnerability, player)
            cached = self._obs_cache.get(cache_key)

            if cached is not None:
                flat = cached
            else:
                # Check if we can extend the previous state by 1 action
                # rather than rebuilding from scratch.
                prev_flat = None
                if hist_tuple:
                    prev_key = (
                        hands_rm.tobytes(), dealer, hist_tuple[:-1], vulnerability, player
                    )
                    prev_state = self._obs_state_cache.get(prev_key)
                    if prev_state is not None:
                        # OpenSpiel states are mutable.  Extending the cached
                        # object in place corrupts the observation stored for the
                        # preceding history and invalidates before/after rewards.
                        os_state = prev_state.clone()
                        last_a = hist_tuple[-1]
                        if not os_state.is_terminal():
                            legal = os_state.legal_actions()
                            if len(legal) > 0 and legal[0] >= 52:
                                os_action = ours_to_openspiel_raw(last_a)
                                if os_action >= 0 and os_action in legal:
                                    os_state.apply_action(os_action)
                        observer = physical_to_openspiel_player(player, dealer)
                        flat = get_openspiel_obs(os_state, observer)
                        # Store new state for future extension
                        if len(self._obs_state_cache) >= _OBS_CACHE_MAX:
                            self._obs_state_cache.pop(next(iter(self._obs_state_cache)))
                        self._obs_state_cache[cache_key] = os_state
                        self._obs_cache[cache_key] = flat
                        prev_flat = flat

                if prev_flat is None:
                    # Full rebuild (first step or cache miss)
                    os_state = hands_to_openspiel_state(hands_rm, dealer,
                                                        vulnerability=vulnerability)
                    for a in history_int:
                        if os_state.is_terminal():
                            break
                        legal = os_state.legal_actions()
                        if len(legal) > 0 and legal[0] < 52:
                            break
                        os_action = ours_to_openspiel_raw(a)
                        if os_action >= 0 and os_action in legal:
                            os_state.apply_action(os_action)
                    observer = physical_to_openspiel_player(player, dealer)
                    flat = get_openspiel_obs(os_state, observer)
                    if len(self._obs_cache) >= _OBS_CACHE_MAX:
                        self._obs_cache.pop(next(iter(self._obs_cache)))
                    if len(self._obs_state_cache) >= _OBS_CACHE_MAX:
                        self._obs_state_cache.pop(next(iter(self._obs_state_cache)))
                    self._obs_cache[cache_key]       = flat
                    self._obs_state_cache[cache_key] = os_state
        else:
            flat = np.zeros(OBS_DIM, dtype=np.float32)

        return flat

    # ======================================================================
    # ======================================================================

    def _compute_raw_info_bonus(
        self,
        episodes: List[List[dict]],
        rinfo_data: Optional[dict] = None,
        beta_override: Optional[float] = None,
    ) -> List[List[float]]:
        """Compute unscaled signed Judge deltas for every actor transition."""
        if self.dual_info is None:
            return [[0.0] * len(ep) for ep in episodes]

        raw_ep_bonuses: List[List[float]] = [[0.0] * len(ep) for ep in episodes]

        if rinfo_data is not None:
            partner_before = rinfo_data['partner_before']
            partner_after = rinfo_data['partner_after']
            opponent_before = rinfo_data['opponent_before']
            opponent_after = rinfo_data['opponent_after']
            target_arr = rinfo_data['target']
            target_pos_arr = rinfo_data['target_pos']
            ep_step_idx = rinfo_data['ep_step']
            n = len(ep_step_idx)
        else:
            valid_steps = []
            ep_step_idx_list = []
            for ep_idx, ep in enumerate(episodes):
                for s_idx, step in enumerate(ep):
                    if step.get('_rinfo'):
                        valid_steps.append(step)
                        ep_step_idx_list.append((ep_idx, s_idx))
            if not valid_steps:
                return raw_ep_bonuses
            partner_before = np.stack([s['partner_obs_before'] for s in valid_steps])
            partner_after = np.stack([s['partner_obs_after'] for s in valid_steps])
            opponent_before = np.stack([s['opponent_obs_before'] for s in valid_steps])
            opponent_after = np.stack([s['opponent_obs_after'] for s in valid_steps])
            target_arr = np.stack([s['belief_target'] for s in valid_steps])
            target_pos_arr = np.array([s['target_pos'] for s in valid_steps])
            ep_step_idx = ep_step_idx_list
            n = len(valid_steps)

        if n == 0:
            return raw_ep_bonuses

        partner_before_t = torch.tensor(partner_before, dtype=torch.float32).to(self.device)
        partner_after_t = torch.tensor(partner_after, dtype=torch.float32).to(self.device)
        opponent_before_t = torch.tensor(opponent_before, dtype=torch.float32).to(self.device)
        opponent_after_t = torch.tensor(opponent_after, dtype=torch.float32).to(self.device)
        target_t = torch.tensor(target_arr, dtype=torch.float32).to(self.device)
        target_pos_t = torch.tensor(target_pos_arr, dtype=torch.long).to(self.device)

        # P_OPT2: merge partner+opponent into one 2x-batch forward (4 calls → 2 calls).
        # Layout: rows [0:B] = partner queries, rows [B:2B] = opponent queries.
        CHUNK = 2048
        bonuses_flat = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, CHUNK):
                sl = slice(start, start + CHUNK)
                B = partner_before_t[sl].shape[0]

                obs_b_2x = torch.cat(
                    [partner_before_t[sl], opponent_before_t[sl]], dim=0
                )
                obs_a_2x = torch.cat(
                    [partner_after_t[sl], opponent_after_t[sl]], dim=0
                )
                tp_2x = torch.cat([target_pos_t[sl], target_pos_t[sl]], dim=0)

                probs_b_2x = self.belief_net.get_probs(obs_b_2x, tp_2x)
                probs_a_2x = self.belief_net.get_probs(obs_a_2x, tp_2x)

                b_before_p, b_before_o = probs_b_2x[:B], probs_b_2x[B:]
                b_after_p,  b_after_o  = probs_a_2x[:B], probs_a_2x[B:]

                partner_gain = self.dual_info.compute_info_gain(
                    b_before_p, b_after_p, target_t[sl]
                )
                opp_leak = self.dual_info.compute_info_gain(
                    b_before_o, b_after_o, target_t[sl]
                )
                beta = self.config.beta if beta_override is None else beta_override
                bonus = partner_gain - beta * opp_leak
                bonuses_flat[start:start + CHUNK] = bonus.cpu().numpy()

        for i, (ep_idx, s_idx) in enumerate(ep_step_idx):
            raw_ep_bonuses[ep_idx][s_idx] = float(bonuses_flat[i])

        return raw_ep_bonuses

    def _compute_info_bonus(self, episodes: List[List[dict]],
                             rinfo_data: Optional[dict] = None) -> List[List[float]]:
        """Apply the single frozen pre-training scale to signed Judge deltas."""
        raw_ep_bonuses = self._compute_raw_info_bonus(episodes, rinfo_data)
        if self.dual_info is None:
            return raw_ep_bonuses
        if self.info_scale_factor is None:
            raise RuntimeError(
                "Information scale is unset. Calibrate it once before training."
            )
        return [
            [bonus * self.info_scale_factor for bonus in episode]
            for episode in raw_ep_bonuses
        ]

    @staticmethod
    def _episode_task_rewards(episodes: List[List[dict]]) -> List[float]:
        rewards = []
        for episode in episodes:
            for step in episode:
                if step.get('done') and step.get('is_training_side'):
                    rewards.append(float(step['reward']))
                    break
        return rewards

    def calibrate_info_scale(self, num_deals: Optional[int] = None) -> dict:
        """Pre-sample one fixed B/C scale from partner-only episode totals.

        Calibration deliberately forces beta=0.  Agent B and Agent C therefore
        share exactly the same units and differ only by C's leakage penalty.
        """
        if self.dual_info is None:
            raise RuntimeError("Information calibration requires an info trainer")
        num_deals = num_deals or self.config.info_scale_calibration_deals
        half = max(1, num_deals // 2)
        ns_eps, ns_data = self._collect_episodes_batch(
            half, train_side='NS', fsp_sd=None,
            batch_size=min(self.config.deals_per_step, half),
            skip_dual_table=False,
        )
        ew_eps, ew_data = self._collect_episodes_batch(
            num_deals - half, train_side='EW', fsp_sd=None,
            batch_size=min(self.config.deals_per_step, max(1, num_deals - half)),
            skip_dual_table=False,
        )
        raw = (
            self._compute_raw_info_bonus(ns_eps, ns_data, beta_override=0.0)
            + self._compute_raw_info_bonus(ew_eps, ew_data, beta_override=0.0)
        )
        info_totals = np.asarray([sum(episode) for episode in raw], dtype=np.float64)
        task_rewards = np.asarray(
            self._episode_task_rewards(ns_eps) + self._episode_task_rewards(ew_eps),
            dtype=np.float64,
        )
        if len(info_totals) < 2 or len(task_rewards) < 2:
            raise RuntimeError("Too few calibration episodes to estimate a fixed scale")
        info_std = max(float(info_totals.std(ddof=1)), 1e-6)
        imp_std = max(float(task_rewards.std(ddof=1)), 1.0)
        self.info_scale_factor = min(imp_std / info_std, 1000.0) * self.config.info_reward_weight
        self.info_scale_metadata = {
            'num_deals': int(len(info_totals)),
            'info_weight': float(self.config.info_reward_weight),
            'partner_only_info_std': info_std,
            'task_imp_std': imp_std,
            'scale_factor': float(self.info_scale_factor),
        }
        print(
            "[Info Scale] frozen partner-only calibration: "
            f"deals={len(info_totals)} info_std={info_std:.6f} "
            f"imp_std={imp_std:.6f} weight={self.config.info_reward_weight:.3f} "
            f"scale={self.info_scale_factor:.6f}"
        )
        return dict(self.info_scale_metadata)

    def copy_info_scale_from(self, other: "SubgameTrainer") -> None:
        if other.info_scale_factor is None:
            raise RuntimeError("Source trainer has no calibrated information scale")
        self.info_scale_factor = float(other.info_scale_factor)
        self.info_scale_metadata = dict(other.info_scale_metadata or {})

    # ======================================================================
    # PPO Update
    # ======================================================================

    def _safe_update(self, player: int, round_idx: int) -> dict:
        """See the formal README for the current behavior contract."""
        buf   = self.buffers[player]
        if len(buf) < self.config.batch_size:
            buf.reset()
            return {}

        actor      = self.agent.get_actor(player)
        critic     = self.agent.get_critic(player)
        actor_opt  = self.agent.get_actor_optimizer(player)
        critic_opt = self.agent.get_critic_optimizer(player)
        kl_lambda  = self._get_kl_lambda(round_idx)

        # Bootstrap last value
        with torch.no_grad():
            fo  = buf.flat_obs[-1].unsqueeze(0).to(self.device)
            ah  = (buf.all_hands[-1].unsqueeze(0).to(self.device)
                   if buf.all_hands else None)
            last_val = critic(fo, ah).item()

        buf.compute_returns_and_advantages(
            last_val, self.agent.config.gamma, self.agent.config.gae_lambda)

        total_policy = total_value = total_entropy = total_kl = 0.0
        total_actor_belief = 0.0
        n_updates = 0

        for epoch in range(self.config.num_epochs):
            epoch_kl = 0.0
            n_batch_kl = 0

            for batch in buf.get_batches(self.agent.config.batch_size):
                b_flat   = batch['flat_obs']
                b_legal  = batch['legal_actions']
                b_act    = batch['actions']
                b_old_lp = batch['old_log_probs']
                b_old_v  = batch['old_values']
                b_ret    = batch['returns']
                b_adv    = batch['advantages']
                b_ah     = batch.get('all_hands')

                # Normalize advantage — guard against degenerate batches
                adv_std = b_adv.std()
                if torch.isnan(adv_std) or adv_std < 1e-8:
                    continue   # skip this batch entirely
                adv = (b_adv - b_adv.mean()) / (adv_std + 1e-8)

                # Actor loss
                log_probs, entropy = actor.evaluate_actions(b_flat, b_legal, b_act)
                ratio = torch.exp(log_probs - b_old_lp)
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.config.clip_ratio,
                                       1 + self.config.clip_ratio) * adv
                ).mean()

                # KL anchor
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.bc_actors is not None and player in self.bc_actors and kl_lambda > 0:
                    bc_actor = self.bc_actors[player]
                    with torch.no_grad():
                        bc_logits = bc_actor(b_flat, b_legal)
                        bc_probs  = F.softmax(bc_logits, dim=-1).clamp(1e-8, 1.0)
                    cur_logits = actor(b_flat, b_legal)
                    cur_probs  = F.softmax(cur_logits, dim=-1).clamp(1e-8, 1.0)
                    kl_loss    = F.kl_div(
                        cur_probs.log(), bc_probs, reduction='batchmean')

                actor_belief_loss = torch.tensor(0.0, device=self.device)
                if (self.config.belief_conditioned
                        and self.config.actor_belief_coef > 0
                        and b_ah is not None):
                    hands_np = b_ah.detach().cpu().numpy()
                    partner_targets = batch_hand_to_belief_target(
                        hands_np[:, (player + 2) % 4]
                    )
                    rho_targets = batch_hand_to_belief_target(
                        hands_np[:, (player - 1) % 4]
                    )
                    belief_targets = torch.as_tensor(
                        np.stack([partner_targets, rho_targets], axis=1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    actor_belief_loss = actor.compute_belief_loss(
                        b_flat, belief_targets
                    )

                actor_loss = (policy_loss
                              - self.config.entropy_coef * entropy.mean()
                              + kl_lambda * kl_loss
                              + self.config.actor_belief_coef * actor_belief_loss)

                if torch.isnan(actor_loss):
                    continue   # skip batch — NaN loss would corrupt weights

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                actor_opt.step()

                # Critic loss
                vals    = critic(b_flat, b_ah)
                v_clip  = b_old_v + (vals - b_old_v).clamp(
                    -self.config.clip_ratio, self.config.clip_ratio)
                value_loss = torch.max(
                    F.mse_loss(vals, b_ret, reduction='none'),
                    F.mse_loss(v_clip, b_ret, reduction='none'),
                ).mean()
                critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
                critic_opt.step()

                # Approx KL for early stopping
                with torch.no_grad():
                    approx_kl = ((b_old_lp - log_probs).mean()).item()
                epoch_kl   += approx_kl
                n_batch_kl += 1

                total_policy  += policy_loss.item()
                total_value   += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl      += kl_loss.item()
                total_actor_belief += actor_belief_loss.item()
                n_updates     += 1

            # Epoch-level KL early stopping
            epoch_kl /= max(1, n_batch_kl)
            if (self.config.kl_early_stop_threshold > 0
                    and epoch_kl > self.config.kl_early_stop_threshold):
                break

        buf.reset()

        if n_updates == 0:
            return {}

        return {
            'policy_loss': total_policy  / n_updates,
            'value_loss':  total_value   / n_updates,
            'entropy':     total_entropy / n_updates,
            'kl_loss':     total_kl      / n_updates,
            'kl_lambda':   kl_lambda,
            'actor_belief_loss': total_actor_belief / n_updates,
        }

    # ======================================================================
    # ======================================================================

    def run(self, num_rounds: int = None, sl_trainer: "SubgameTrainer" = None,
            h2h_deals: int = 500) -> List[dict]:
        """See the formal README for the current behavior contract."""
        num_rounds = num_rounds or self.config.num_rounds
        n_deals    = self.config.steps_per_phase * self.config.deals_per_step
        batch_sz   = self.config.deals_per_step
        cfg = self.config
        print(f"[Config] kl_lambda_start={cfg.kl_lambda_start}  kl_lambda_end={cfg.kl_lambda_end}  kl_anneal_frac={cfg.kl_anneal_frac}")
        print(f"[Config] num_rounds={num_rounds}  deals_per_step={cfg.deals_per_step}  n_deals_per_phase={n_deals}  use_info_bonus={cfg.use_info_bonus}  beta={cfg.beta}  info_weight={cfg.info_reward_weight}")

        print("\n[Trainer] Critic warmup...")
        self.critic_warmup()

        if not self.config.self_play and not self._fsp_seeded and self.config.fsp_pool_size >= 0:
            self.fsp_pool.add_permanent(self.agent)
            self._fsp_seeded = True
            print(f"  [FSP] Seeded pool with BC checkpoint as permanent (pool size: {len(self.fsp_pool)})")
        if self.config.self_play:
            print("  [Self-Play] Pure self-play mode: opponent = current agent (FSP disabled)")

        for rnd in range(num_rounds):
            print(f"\n══════ Round {rnd+1}/{num_rounds} ══════")

            _bw = False

            if self.config.self_play:
                fsp_sd = None
            else:
                self._maybe_add_to_fsp(rnd)
                fsp_sd = self._apply_fsp_opponent()
            _ir_vals: list = []; _bl_vals: list = []
            ns_metrics: dict = {}; ew_metrics: dict = {}

            # ── (P93: JIT burn-in removed — belief update moved to after PPO) ──

            import time as _time

            print(f"  [Table1/NS] Collecting {n_deals} deals (batch={batch_sz})...")
            _t = _time.time()
            ns_eps, ns_rinfo = self._collect_episodes_batch(
                n_deals, train_side='NS', fsp_sd=fsp_sd, batch_size=batch_sz,
                use_belief_prior=_bw)

            raw_ns_vals = []
            for ep in ns_eps:
                for step in ep:
                    if step.get('done') and step['player'] in (NORTH, SOUTH):
                        v = float(step['reward'])
                        self.reward_stats.update(v)
                        raw_ns_vals.append(v)
                        break

            _t = _time.time()
            if self.config.use_info_bonus:
                bonuses = self._compute_info_bonus(ns_eps, rinfo_data=ns_rinfo)
                for ep, bs in zip(ns_eps, bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b
                        if b != 0.0: _ir_vals.append(b)

            _t = _time.time()
            for ep in ns_eps:
                for step in ep:
                    if step['player'] in (NORTH, SOUTH):
                        self.buffers[step['player']].add(
                            flat_obs=step['flat_obs'], legal_actions=step['legal_actions'],
                            action=step['action'], log_prob=step['log_prob'],
                            reward=step['reward'], value=step['value'],
                            done=step['done'], all_hands=step.get('all_hands'))
            for p in (NORTH, SOUTH):
                m = self._safe_update(p, rnd)
                if m: ns_metrics[p] = m

            print(f"  [Table2/EW] Collecting {n_deals} deals (batch={batch_sz})...")
            _t = _time.time()
            ew_eps, ew_rinfo = self._collect_episodes_batch(
                n_deals, train_side='EW', fsp_sd=fsp_sd, batch_size=batch_sz,
                use_belief_prior=_bw)

            raw_ew_vals = []
            for ep in ew_eps:
                for step in ep:
                    if step.get('done') and step['player'] in (EAST, WEST):
                        v = float(step['reward'])
                        self.reward_stats.update(v)
                        raw_ew_vals.append(v)
                        break

            _t = _time.time()
            if self.config.use_info_bonus:
                ew_bonuses = self._compute_info_bonus(ew_eps, rinfo_data=ew_rinfo)
                for ep, bs in zip(ew_eps, ew_bonuses):
                    for step, b in zip(ep, bs):
                        step['reward'] += b
                        if b != 0.0: _ir_vals.append(b)

            _t = _time.time()
            for ep in ew_eps:
                for step in ep:
                    if step['player'] in (EAST, WEST):
                        self.buffers[step['player']].add(
                            flat_obs=step['flat_obs'], legal_actions=step['legal_actions'],
                            action=step['action'], log_prob=step['log_prob'],
                            reward=step['reward'], value=step['value'],
                            done=step['done'], all_hands=step.get('all_hands'))
            for p in (EAST, WEST):
                m = self._safe_update(p, rnd)
                if m: ew_metrics[p] = m

            # ── Belief Update: frozen (P96) or on-policy (P93/P95) ─────────
            _t = _time.time()
            if self.belief_net is not None and not self.config.freeze_belief:
                bl = self.update_belief_on_policy(ns_eps + ew_eps)
                if bl is not None: _bl_vals.append(bl)

            all_vals = raw_ns_vals + raw_ew_vals
            mean_r = float(np.mean(all_vals)) if all_vals else 0.0
            std_r  = float(np.std(all_vals))  if all_vals else 0.0
            mean_ns = float(np.mean(raw_ns_vals)) if raw_ns_vals else 0.0
            mean_ew = float(np.mean(raw_ew_vals)) if raw_ew_vals else 0.0
            _actor_belief_vals = [
                metrics['actor_belief_loss']
                for metrics in list(ns_metrics.values()) + list(ew_metrics.values())
                if 'actor_belief_loss' in metrics
            ]

            log_entry = {
                'round': rnd+1, 'mean_reward': mean_r, 'std_reward': std_r,
                'mean_task_imp': mean_r,
                'mean_ns_task_imp': mean_ns, 'mean_ew_task_imp': mean_ew,
                'ns_metrics': ns_metrics, 'ew_metrics': ew_metrics,
                'fsp_pool_size': len(self.fsp_pool),
                'mean_ir':    float(np.mean(_ir_vals)) if _ir_vals else None,
                'belief_loss': float(np.mean(_bl_vals)) if _bl_vals else None,
                'actor_belief_loss': (
                    float(np.mean(_actor_belief_vals))
                    if _actor_belief_vals else None
                ),
                'imp_std_running': float(self.reward_stats.std),
            }
            self.log.append(log_entry)
            self._print_log(log_entry)

            # P110: Per-round H2H removed — too noisy (500-deal IMP std≈9).
            # All statistical evaluation is deferred to Stage 3 (5000 deals).

            if self.config.early_stop_enabled:
                all_vl = []
                for m in list(ns_metrics.values()) + list(ew_metrics.values()):
                    if 'value_loss' in m:
                        all_vl.append(m['value_loss'])
                if all_vl:
                    self._vl_history.append(float(np.mean(all_vl)))
                pat = self.config.early_stop_patience
                if len(self._vl_history) >= pat:
                    window = self._vl_history[-pat:]
                    vl_range = max(window) - min(window)
                    if vl_range < self.config.early_stop_vl_delta:
                        print(f"\n  [Early Stop] vl plateau (range={vl_range:.3f} < {self.config.early_stop_vl_delta} over {pat} rounds). Stopping at round {rnd+1}.")
                        break

        return self.log

    def _print_log(self, entry: dict):
        rnd = entry['round']
        fsp = entry['fsp_pool_size']
        mr  = entry['mean_reward']
        sr  = entry['std_reward']
        ns_r = entry.get('mean_ns_task_imp', 0.0)
        ew_r = entry.get('mean_ew_task_imp', 0.0)

        print(f"  [Round {rnd}] rollout_task_IMP={mr:+.3f}±{sr:.3f}  (NS={ns_r:+.3f} EW={ew_r:+.3f})  fsp={fsp}")

        ns = entry.get('ns_metrics', {})
        ns_n = ns.get(NORTH, {}); ns_s = ns.get(SOUTH, {})
        if ns_n or ns_s:
            print(f"    NS │ N: pl={ns_n.get('policy_loss',0):+.4f} "
                  f"vl={ns_n.get('value_loss',0):.3f} "
                  f"ent={ns_n.get('entropy',0):.3f} │ "
                  f"S: pl={ns_s.get('policy_loss',0):+.4f} "
                  f"vl={ns_s.get('value_loss',0):.3f} "
                  f"ent={ns_s.get('entropy',0):.3f} "
                  f"kl={ns_s.get('kl_loss',0):.5f}(λ={ns_s.get('kl_lambda',0):.3f})")

        ew = entry.get('ew_metrics', {})
        ew_e = ew.get(EAST, {}); ew_w = ew.get(WEST, {})
        if ew_e or ew_w:
            print(f"    EW │ E: pl={ew_e.get('policy_loss',0):+.4f} "
                  f"vl={ew_e.get('value_loss',0):.3f} "
                  f"ent={ew_e.get('entropy',0):.3f} │ "
                  f"W: pl={ew_w.get('policy_loss',0):+.4f} "
                  f"vl={ew_w.get('value_loss',0):.3f} "
                  f"ent={ew_w.get('entropy',0):.3f} "
                  f"kl={ew_e.get('kl_loss',0):.5f}(λ={ew_e.get('kl_lambda',0):.3f})")

        ir = entry.get('mean_ir')
        bl = entry.get('belief_loss')
        actor_bl = entry.get('actor_belief_loss')
        if actor_bl is not None:
            print(f"    actor_belief_loss={actor_bl:.4f}")
        if ir is not None:
            bl_str = f"{bl:.4f}" if bl is not None else "N/A"
            print(f"    r_info │ step_ir={ir:.4f}  belief_loss={bl_str}")
