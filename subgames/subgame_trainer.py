"""MAPPO/FSP trainer for the competitive bridge-bidding subgame.

Actors expose the standard 571-dimensional OpenSpiel observation API and build
partner/RHO belief features internally.  A separate frozen BeliefNet is the
training-only Judge.  For a bid by player ``i``, both Judge receiver views
predict ``i``'s hand: the partner view conditions on the partner's private hand
and the opponent view conditions on the next opponent's private hand.
"""

from __future__ import annotations

import numpy as np
import math
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from env import (
    NUM_PLAYERS, NUM_BIDS, BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
    NORTH, EAST, SOUTH, WEST,
)
from networks.belief_net import BeliefNetwork, DualInfoComputer
from networks.policy_net import (
    MLPPolicyNetwork, MLPValueNetwork, OBS_DIM,
    convert_hands_suit_to_rank, hands_to_openspiel_state,
    get_openspiel_obs, ours_to_openspiel_raw,
    physical_to_openspiel_player, encode_openspiel_auction_observation,
)
from algorithms.mappo import MAPPOAgent, MAPPOConfig
from utils.running_stats import RunningStats
from utils.hand_features import (
    hand_to_belief_target, batch_hand_to_belief_target,
    belief_accuracy, BELIEF_DIM,
)
from utils.fsp_pool import FSPPool
from utils.imp import score_to_imp
from dri.evidence_removal import remove_target_evidence
from networks.task_q import (
    ALL_HANDS_ENCODER_SHARED_BRIDGE,
    OBSERVATION_ENCODER_STRUCTURED_AUCTION,
    StructuredTaskQNetwork,
    TaskQConfig,
    normalize_dd_table_ctde,
    normalize_reference_score_ctde,
)
from experiments.dri_task_q_dataset import (
    CTDE_SEAT_ORDER_ACTING_RELATIVE,
    build_structured_action_features,
    normalize_ctde_hands,
)

# P109: max entries for _encode_for_actor obs/state caches per trainer instance.
# A rollout batch of 512 deals x ~8 steps x 4 players approximately 16k unique (hands, hist) keys.
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
    # CPU threads used for OpenSpiel observation construction and environment
    # stepping during rollout collection.  Policy/critic inference remains a
    # single batched GPU operation, so PPO stays on-policy.
    collector_workers: int = 1
    fast_observation_encoding: bool = True
    accumulate_steps: int   = 4
    # Bound transient episode/r_info memory without changing the amount of
    # on-policy data consumed by each PPO update.
    rollout_chunk_deals: int = 8192

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

    # -- KL Early Stopping ---------------------------------------------------
    kl_early_stop_threshold: float = 0.015

    # Dual-information reward
    use_info_bonus:      bool  = False
    beta:                float = 0.05
    info_reward_weight:  float = 0.05
    info_scale_calibration_deals: int = 2048
    # Policy-invariant alternative to the legacy immediate Judge delta.  For
    # each seat's own decision process, use F=gamma*Phi(next own turn)-Phi(now)
    # with terminal Phi=0.  Phi=-CE_partner+beta*CE_opponent.
    info_potential_shaping: bool = False

    # Receiver-mediated help reward.  This is the inexpensive Monte-Carlo
    # estimator H=(1-pi_deaf(a)/pi_heard(a))*Q_task.  It is routed back to the
    # sender transition; no shaped value is used inside H.
    use_help_bonus:       bool  = False
    help_beta:            float = 1.0
    help_reward_weight:   float = 0.05
    help_weight_clip:     float = 10.0
    help_return_equivalent: bool = False
    help_receiver_value_baseline: bool = False
    # Expensive COMA-style variant: estimate the receiver path directly as
    # sum_a (pi_heard(a)-pi_deaf(a))*Q_task(s,a).  Its Task-Q is independent
    # from PPO and is trained only on task-only terminal IMP.
    help_all_action_q: bool = False
    help_task_q_lr: float = 3e-4
    help_task_q_batch_size: int = 1024
    help_task_q_epochs: int = 1
    help_task_q_min_samples: int = 4096
    help_task_q_hidden_dim: int = 256

    # Deployed actor: 571 external observation -> internal 96-dim belief head.
    belief_conditioned: bool = True
    actor_belief_coef:  float = 0.1

    # Fictitious self-play
    fsp_pool_size:    int   = 10
    fsp_add_interval: int   = 2
    self_play:        bool  = False

    # -- P126: FSP Quality Gate + Weighted SL Sampling ----------------------
    fsp_quality_gate:          bool  = True   # Enable quality gate before pool insertion
    fsp_gate_eval_deals:       int   = 200    # Deals to play vs SL for quality evaluation
    fsp_gate_max_auction_len:  int   = 7      # Reject if median auction length > this (SL median=6)
    fsp_gate_max_double_rate:  float = 0.60   # Reject if doubled contract rate > this (SL approximately 53%)
    fsp_sl_sample_prob:        float = 0.30   # Minimum probability of sampling SL permanent member

    # -- Belief Net Update ------------------------------------------------
    # P93: on-policy update (epochs=8, lr=5e-5) - caused catastrophic
    #   forgetting of pretrain foundation (val_loss 1.76->2.19 in Round 1).
    # P96: freeze_belief=True by default. With strong KL (lambda=0.5),
    #   policy stays near SAYC and pretrain Belief Net remains valid.
    belief_update_epochs: int  = 3            # P95: only used if freeze_belief=False
    belief_update_lr:     float = 1e-5        # P95: only used if freeze_belief=False
    freeze_belief:        bool  = True        # P96: frozen by default

    # -- EWC for Belief Net (P97) -------------------------------------
    use_ewc:              bool  = False       # P97: EWC-protected on-policy update
    ewc_lambda:           float = 100.0       # P97: EWC penalty strength (Fisher normalized, mean penalty)
    ewc_fisher_samples:   int   = 5000        # Samples for Fisher computation

    # -- Critic Warmup -------------------------------------------------------
    critic_prewarm_deals:  int   = 2048
    critic_prewarm_epochs: int   = 10
    critic_prewarm_conv_tol: float = 0.05

    # -- BC Warmup(rule-based)----------------------------------------------
    bc_warmup_samples: int  = 5000
    bc_warmup_epochs:  int  = 20
    bc_warmup_lr:      float = 1e-4

    hidden_dim:       int   = 1024        # 4 x 1024 MLP

    active_players:      Optional[List[int]] = None
    eval_interval:       int   = 200
    log_interval:        int   = 50
    device:              str   = 'cpu'
    early_stop_patience:   int   = 8
    early_stop_vl_delta:   float = 0.15
    early_stop_enabled:    bool  = False   # P123: disabled - fixed schedule for reproducibility


# ==============================================================================
# Belief Replay Buffer (P84)
# ==============================================================================

class FlatRolloutBuffer:
    """CPU rollout storage compacted into tensors after every collection chunk."""

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
        self._packed: dict = {}
        self._packed_count = 0
        self._materialized: Optional[dict] = None

    def add(self, flat_obs, legal_actions, action, log_prob, reward, value, done,
            all_hands=None):
        if self._materialized is not None:
            raise RuntimeError("Cannot append to a materialized rollout buffer")
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

    def pack_pending(self) -> None:
        """Replace thousands of tiny Python/Tensor objects with dense chunks."""
        if not self.actions:
            return
        fields = {
            'flat_obs': torch.stack(self.flat_obs),
            'legal_actions': torch.stack(self.legal_actions),
            'actions': torch.stack(self.actions),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.as_tensor(self.rewards, dtype=torch.float32),
            'values': torch.stack(self.values).reshape(-1),
            'dones': torch.as_tensor(self.dones, dtype=torch.bool),
        }
        if self.all_hands:
            if len(self.all_hands) != len(self.actions):
                raise RuntimeError("all_hands missing from part of a rollout chunk")
            fields['all_hands'] = torch.stack(self.all_hands)
        for key, tensor in fields.items():
            self._packed.setdefault(key, []).append(tensor)
        self._packed_count += len(self.actions)
        self.flat_obs.clear(); self.legal_actions.clear(); self.actions.clear()
        self.log_probs.clear(); self.rewards.clear(); self.values.clear()
        self.dones.clear(); self.all_hands.clear()

    def add_steps(self, steps: List[dict]) -> None:
        """Append one player's chunk directly as dense CPU tensors."""
        if not steps:
            return
        if self._materialized is not None:
            raise RuntimeError("Cannot append to a materialized rollout buffer")
        self.pack_pending()
        fields = {
            'flat_obs': torch.from_numpy(np.stack(
                [step['flat_obs'] for step in steps]
            ).astype(np.float32, copy=False)),
            'legal_actions': torch.from_numpy(np.stack(
                [step['legal_actions'] for step in steps]
            ).astype(np.float32, copy=False)),
            'actions': torch.as_tensor(
                [step['action'] for step in steps], dtype=torch.int64
            ),
            'log_probs': torch.stack([
                step['log_prob'].detach().cpu()
                if hasattr(step['log_prob'], 'detach')
                else torch.as_tensor(step['log_prob'])
                for step in steps
            ]),
            'rewards': torch.as_tensor(
                [step['reward'] for step in steps], dtype=torch.float32
            ),
            'values': torch.stack([
                step['value'].detach().cpu()
                if hasattr(step['value'], 'detach')
                else torch.as_tensor(step['value'])
                for step in steps
            ]).reshape(-1),
            'dones': torch.as_tensor(
                [step['done'] for step in steps], dtype=torch.bool
            ),
        }
        if steps[0].get('all_hands') is not None:
            fields['all_hands'] = torch.from_numpy(np.stack(
                [step['all_hands'] for step in steps]
            ).astype(np.float32, copy=False))
        for key, tensor in fields.items():
            self._packed.setdefault(key, []).append(tensor)
        self._packed_count += len(steps)

    def _materialize(self) -> dict:
        if self._materialized is None:
            self.pack_pending()
            self._materialized = {
                key: (chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0))
                for key, chunks in self._packed.items()
            }
            self._packed.clear()
        return self._materialized

    def last_inputs(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        data = self._materialize()
        return data['flat_obs'][-1], (
            data['all_hands'][-1] if 'all_hands' in data else None
        )

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ):
        data = self._materialize()
        rewards = data['rewards'].numpy()
        values_np = data['values'].numpy()
        dones = data['dones'].numpy()
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else values_np[t + 1]
            # dones[t] belongs to the transition stored at t.  Looking at
            # dones[t + 1] leaks the first value of the next episode across a
            # terminal boundary and cuts off the terminal reward one step early.
            non_terminal = 1.0 - float(dones[t])
            delta     = (rewards[t] + gamma * next_val * non_terminal
                         - values_np[t])
            last_gae  = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns    = torch.tensor(returns,    dtype=torch.float32)

    def __len__(self):
        if self._materialized is not None:
            return len(self._materialized['actions'])
        return self._packed_count + len(self.actions)

    def get_batches(self, batch_size: int):
        data = self._materialize()
        n       = len(data['actions'])
        indices = np.random.permutation(n)
        device  = self.device

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:          # skip incomplete last batch (std() needs >=2 samples)
                continue
            batch = {
                'flat_obs':      data['flat_obs'][idx].to(device),
                'legal_actions': data['legal_actions'][idx].to(device),
                'actions':       data['actions'][idx].to(device),
                'old_log_probs': data['log_probs'][idx].to(device),
                'old_values':    data['values'][idx].to(device),
                'advantages':    self.advantages[idx].to(device),
                'returns':       self.returns[idx].to(device),
            }
            if 'all_hands' in data:
                batch['all_hands'] = data['all_hands'][idx].to(device)
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

        # -- Per-player flat rollout buffers --------------------------------
        self.buffers: Dict[int, FlatRolloutBuffer] = {
            p: FlatRolloutBuffer(self.device) for p in range(NUM_PLAYERS)
        }
        self.capture_task_actor_gradients = False
        self._last_task_actor_gradients: Dict[int, torch.Tensor] = {}
        self._last_task_actor_gradient_metadata: Dict[int, dict] = {}

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
        self.help_reward_metadata: Optional[dict] = None
        self.help_scale_factor: Optional[float] = None
        self.help_task_q: Optional[StructuredTaskQNetwork] = None
        self.help_task_q_optimizer = None
        self.help_task_q_samples_seen = 0
        self.help_task_q_updates = 0
        if config.use_help_bonus and config.help_all_action_q:
            q_config = TaskQConfig(
                hidden_dim=config.help_task_q_hidden_dim,
                num_hidden_layers=2,
                all_hands_embedding_dim=128,
                all_hands_encoder_kind=ALL_HANDS_ENCODER_SHARED_BRIDGE,
                observation_encoder_kind=OBSERVATION_ENCODER_STRUCTURED_AUCTION,
                observation_embedding_dim=128,
                analytic_stop_baseline_required=False,
            )
            # Keep initialization independent from the policy RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(91_337)
                self.help_task_q = StructuredTaskQNetwork(q_config).to(self.device)
            self.help_task_q_optimizer = torch.optim.Adam(
                self.help_task_q.parameters(), lr=config.help_task_q_lr
            )

        self.reward_stats: RunningStats = reward_stats or RunningStats()

        # -- FSP pool -------------------------------------------------------
        # P126: fsp_pool_size=0 -> unlimited (use 999999 as practical max)
        _pool_max = config.fsp_pool_size if config.fsp_pool_size > 0 else 999999
        self.fsp_pool = FSPPool(max_size=_pool_max)

        self.bc_actors: Optional[Dict[int, MLPPolicyNetwork]] = None

        self._fsp_actor_cache: dict = {}  # role -> MLPPolicyNetwork
        self._fsp_cache_source: Optional[dict] = None

        # Keyed on (hands_rm.tobytes(), dealer, history_tuple) -> flat obs array
        # Cleared at the start of each collect_episodes_batch call.
        self._obs_cache:       dict = {}
        self._obs_state_cache: dict = {}  # same key -> pyspiel.State (for incremental extend)

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
            # No annealing - fixed at start value
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

        # -- Quality gate evaluation (skip round 0 - not yet trained) -----
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
            auction_lengths.append(
                len(ep) + int(getattr(self.env, 'initial_history_length', 0))
            )

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
        # P126: weighted sampling - SL permanent member gets guaranteed floor
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
                last_obs, last_hands = buf.last_inputs()
                fo = last_obs.unsqueeze(0).to(self.device)
                ah = (last_hands.unsqueeze(0).to(self.device)
                      if last_hands is not None else None)
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

        # -- P97: Compute Fisher Information Matrix for EWC ------------
        if self.config.use_ewc:
            print(f"[Belief Pretrain] Computing Fisher for EWC "
                  f"(lambda_ewc={self.config.ewc_lambda}, samples={self.config.ewc_fisher_samples})...")
            self.belief_net.compute_fisher(
                obs_all, tp_all, tgt_all,
                num_samples=self.config.ewc_fisher_samples)

        # -- P97b: Save pretrain data for replay mixing -------------
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

        # -- Check if pretrain replay data is available --
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

                # -- On-policy loss --
                loss_op = self.belief_net.compute_loss(
                    obs[idx].to(self.device),
                    tp[idx].to(self.device),
                    tgt[idx].to(self.device))

                # -- Pretrain replay loss (P97b) --
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

                # -- Combined loss: 80% on-policy + 20% replay (P97c) --
                # On-policy dominant so belief tracks current policy,
                # replay minority prevents catastrophic forgetting of pretrain.
                loss = 0.8 * loss_op + 0.2 * loss_rp if has_replay else loss_op

                # -- Optional EWC penalty --
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

            # Validation (on-policy data only - this is what matters for r_info)
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
        from concurrent.futures import ThreadPoolExecutor

        # P109: clear obs cache at start of each batch to bound memory usage
        self._obs_cache.clear()
        self._obs_state_cache.clear()
        self._prepare_fsp_cache(fsp_sd)

        envs = [self.env.clone_for_worker() for _ in range(batch_size)]

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

        worker_count = max(1, int(self.config.collector_workers))
        executor = (ThreadPoolExecutor(max_workers=worker_count)
                    if worker_count > 1 else None)

        def _parallel_map(fn, items):
            if executor is None:
                return [fn(item) for item in items]
            return list(executor.map(fn, items))

        try:
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

                def _encode_slot(i):
                    return self._encode_for_actor(
                        slot_obs[i], slot_dealer[i], slot_hist[i],
                        envs[i].current_player,
                        all_hands=envs[i]._current_hands,
                        use_prior=True,
                        vulnerability=envs[i]._vulnerability)[:OBS_DIM]

                obs571_batch = np.stack(_parallel_map(_encode_slot, slots))
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

            def _advance_slot(i):
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
                    'dd_table': envs[i]._current_dd.copy(),
                    'is_training_side': is_train,
                    'is_opener': player in opener_seats_i,
                    'obs_571': obs_571,
                    'dealer': _dealer_i,
                    'vulnerability': envs[i]._vulnerability,
                    'public_history_before': tuple(slot_hist[i]),
                }

                if (self.config.use_info_bonus or self.config.use_help_bonus) and is_train:
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
                        obs_next, _dealer_i, slot_hist[i], step['partner_pos'],
                        all_hands=all_hands,
                        vulnerability=envs[i]._vulnerability,
                    )[:OBS_DIM]
                    step['opponent_obs_after'] = self._encode_for_actor(
                        obs_next, _dealer_i, slot_hist[i], step['opponent_pos'],
                        all_hands=all_hands,
                        vulnerability=envs[i]._vulnerability,
                    )[:OBS_DIM]
                step['reward'] = reward
                step['done']   = done
                return i, step, obs_next, done

            advanced = _parallel_map(_advance_slot, active)
            for i, step, obs_next, done in advanced:
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

        finally:
            if executor is not None:
                executor.shutdown(wait=True)



        if not skip_dual_table and _pending_rewards:
            for ep_idx, dd, dealer, vul, contract in _pending_rewards:
                score_ns  = self.env._compute_score_ns(contract, dd, vul)
                score_opt = self.env._compute_dds_optimal_score_ns(dd, vul, dealer)
                imp_ns    = float(score_to_imp(score_ns - score_opt))
                imp_ew    = -imp_ns

                last_step_idx: Dict[int, int] = {}
                for s_idx, s in enumerate(all_episodes[ep_idx]):
                    # The duplicate payoff context is immutable across this
                    # episode and is required by the CTDE Task-Q receiver path.
                    s['reference_score_ns'] = int(score_opt)
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
            if self.config.fast_observation_encoding:
                return encode_openspiel_auction_observation(
                    hands_sm, dealer, history_int, player, vulnerability
                )
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

        # P_OPT2: merge partner+opponent into one 2x-batch forward (4 calls -> 2 calls).
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

    def _compute_raw_info_potential_bonus(
        self,
        episodes: List[List[dict]],
        rinfo_data: Optional[dict] = None,
        beta_override: Optional[float] = None,
    ) -> List[List[float]]:
        """Compute strict per-seat potential shaping on the PPO time axis.

        A seat's next state is its next stored decision state, not the next
        global auction action.  This matches ``FlatRolloutBuffer`` discounting.
        Terminal potential is exactly zero, so the discounted shaping return
        telescopes to ``-Phi(first_state)`` for every seat and deal.
        """
        if self.dual_info is None:
            return [[0.0] * len(ep) for ep in episodes]

        potentials = [[None] * len(ep) for ep in episodes]
        if rinfo_data is not None:
            partner_before = rinfo_data['partner_before']
            opponent_before = rinfo_data['opponent_before']
            target_arr = rinfo_data['target']
            target_pos_arr = rinfo_data['target_pos']
            ep_step_idx = rinfo_data['ep_step']
        else:
            valid_steps = []
            ep_step_idx = []
            for ep_idx, episode in enumerate(episodes):
                for step_idx, step in enumerate(episode):
                    if step.get('_rinfo'):
                        valid_steps.append(step)
                        ep_step_idx.append((ep_idx, step_idx))
            if not valid_steps:
                return [[0.0] * len(ep) for ep in episodes]
            partner_before = np.stack([
                step['partner_obs_before'] for step in valid_steps
            ])
            opponent_before = np.stack([
                step['opponent_obs_before'] for step in valid_steps
            ])
            target_arr = np.stack([
                step['belief_target'] for step in valid_steps
            ])
            target_pos_arr = np.asarray([
                step['target_pos'] for step in valid_steps
            ], dtype=np.int64)

        if len(ep_step_idx) == 0:
            return [[0.0] * len(ep) for ep in episodes]
        partner_t = torch.as_tensor(
            partner_before, dtype=torch.float32, device=self.device
        )
        opponent_t = torch.as_tensor(
            opponent_before, dtype=torch.float32, device=self.device
        )
        target_t = torch.as_tensor(
            target_arr, dtype=torch.float32, device=self.device
        )
        target_pos_t = torch.as_tensor(
            target_pos_arr, dtype=torch.long, device=self.device
        )
        beta = self.config.beta if beta_override is None else beta_override
        values = np.empty(len(ep_step_idx), dtype=np.float32)
        chunk = 2048
        with torch.no_grad():
            for start in range(0, len(ep_step_idx), chunk):
                sl = slice(start, start + chunk)
                partner_belief = self.belief_net.get_probs(
                    partner_t[sl], target_pos_t[sl]
                )
                opponent_belief = self.belief_net.get_probs(
                    opponent_t[sl], target_pos_t[sl]
                )
                partner_ce = self.dual_info.compute_cross_entropy(
                    partner_belief, target_t[sl]
                )
                opponent_ce = self.dual_info.compute_cross_entropy(
                    opponent_belief, target_t[sl]
                )
                values[start:start + chunk] = (
                    -partner_ce + float(beta) * opponent_ce
                ).cpu().numpy()
        for value, (ep_idx, step_idx) in zip(values, ep_step_idx):
            potentials[ep_idx][step_idx] = float(value)

        gamma = float(self.config.gamma)
        bonuses = [[0.0] * len(ep) for ep in episodes]
        for ep_idx, episode in enumerate(episodes):
            by_player: Dict[int, List[int]] = {}
            for step_idx, step in enumerate(episode):
                if potentials[ep_idx][step_idx] is not None:
                    by_player.setdefault(int(step['player']), []).append(step_idx)
            for indices in by_player.values():
                for position, step_idx in enumerate(indices):
                    current = float(potentials[ep_idx][step_idx])
                    next_phi = (
                        float(potentials[ep_idx][indices[position + 1]])
                        if position + 1 < len(indices) else 0.0
                    )
                    bonuses[ep_idx][step_idx] = gamma * next_phi - current
        return bonuses

    def _compute_help_bonus(
        self,
        episodes: List[List[dict]],
        *,
        fsp_sd: Optional[dict],
    ) -> Tuple[List[List[float]], dict]:
        """Estimate receiver-mediated task help and route it to each sender.

        The receiver's realized call is sampled by the rollout itself.  Q is
        the terminal task IMP from that receiver partnership's perspective.
        An optional detached receiver-state value is an action-independent
        variance-reduction baseline.  It does not change the unclipped
        importance-weighted estimand.
        """
        partner_bonus = [[0.0 for _ in episode] for episode in episodes]
        opponent_bonus = [[0.0 for _ in episode] for episode in episodes]
        records: Dict[Tuple[int, bool, str], List[dict]] = {}

        def next_receiver(ep: List[dict], start: int, player: int):
            for index in range(start + 1, len(ep)):
                if int(ep[index]['player']) == player:
                    return index, ep[index]
            return None

        def task_imp(ep: List[dict], partnership: int) -> float:
            for step in reversed(ep):
                if (int(step['player']) % 2 == partnership
                        and bool(step.get('done'))):
                    return float(step['reward'])
            raise RuntimeError("terminal task IMP missing for receiver partnership")

        for ep_index, episode in enumerate(episodes):
            for sender_index, sender in enumerate(episode):
                if not sender.get('_rinfo') or not sender.get('is_training_side'):
                    continue
                sender_player = int(sender['player'])
                for role, receiver_player, slot, before_key, after_key in (
                    ('partner', (sender_player + 2) % 4, 'partner',
                     'partner_obs_before', 'partner_obs_after'),
                    ('opponent', (sender_player + 1) % 4, 'rho',
                     'opponent_obs_before', 'opponent_obs_after'),
                ):
                    found = next_receiver(episode, sender_index, receiver_player)
                    if found is None:
                        continue
                    receiver_index, receiver = found
                    key = (receiver_player,
                           bool(receiver.get('is_training_side')), slot)
                    records.setdefault(key, []).append({
                        'episode_index': ep_index,
                        'sender_index': sender_index,
                        'receiver_index': receiver_index,
                        'role': role,
                        'receiver_obs': receiver['obs_571'],
                        'legal_actions': receiver['legal_actions'],
                        'action': int(receiver['action']),
                        'before': sender[before_key],
                        'after': sender[after_key],
                        'q_task': task_imp(episode, receiver_player % 2),
                        'receiver_value': float(receiver['value']),
                        'all_hands': receiver['all_hands'],
                        'dd_table': receiver['dd_table'],
                        'reference_score_ns': receiver['reference_score_ns'],
                        'dealer': receiver['dealer'],
                        'vulnerability': receiver['vulnerability'],
                        'public_history_before': receiver['public_history_before'],
                        'receiver_player': receiver_player,
                    })

        raw_weights = []
        clipped_weights = []
        partner_values = []
        opponent_values = []
        clip_count = 0
        use_all_action_q = bool(self.config.help_all_action_q)
        q_ready = bool(
            use_all_action_q
            and self.help_task_q is not None
            and self.help_task_q_samples_seen >= self.config.help_task_q_min_samples
        )
        direct_values = []
        heard_deaf_l1_values = []
        legal_q_span_values = []
        for (receiver_player, is_training, slot), rows in records.items():
            actor = (
                self.agent.get_actor(receiver_player)
                if is_training else self._get_fsp_actor(receiver_player, fsp_sd)
            )
            actor.eval()
            chunk = 4096
            for start in range(0, len(rows), chunk):
                selected = rows[start:start + chunk]
                receiver_obs = np.stack([row['receiver_obs'] for row in selected])
                before = np.stack([row['before'] for row in selected])
                after = np.stack([row['after'] for row in selected])
                legal = torch.as_tensor(
                    np.stack([row['legal_actions'] for row in selected]),
                    dtype=torch.float32, device=self.device,
                )
                actions = torch.as_tensor(
                    [row['action'] for row in selected],
                    dtype=torch.long, device=self.device,
                )
                removed = remove_target_evidence(
                    actor, receiver_obs, before, after, target_slot=slot
                )
                obs_t = torch.as_tensor(
                    receiver_obs, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    heard_logits = actor.forward_with_belief_features(
                        obs_t, legal, removed.heard_features
                    )
                    deaf_logits = actor.forward_with_belief_features(
                        obs_t, legal, removed.deaf_features
                    )
                    heard_log = F.log_softmax(heard_logits.double(), dim=-1)
                    deaf_log = F.log_softmax(deaf_logits.double(), dim=-1)
                    if use_all_action_q:
                        if q_ready:
                            q_batch = self._help_task_q_batch(selected)
                            q_values = self.help_task_q(
                                q_batch['observations'], q_batch['all_hands_ctde'],
                                q_batch['legal_action_mask'],
                                dd_table_ctde=q_batch['dd_table_ctde'],
                                reference_score_ctde=q_batch['reference_score_ctde'],
                                action_features_ctde=q_batch['action_features_ctde'],
                            ).double()
                            direct = self._all_action_receiver_help(
                                heard_log, deaf_log, q_values, legal
                            )
                            heard_probability = heard_log.exp()
                            deaf_probability = deaf_log.exp()
                            heard_deaf_l1_values.extend(
                                (heard_probability - deaf_probability).abs()
                                .sum(dim=-1).cpu().tolist()
                            )
                            legal_bool = legal.to(torch.bool)
                            for row_q, row_legal in zip(q_values, legal_bool):
                                selected_q = row_q[row_legal]
                                legal_q_span_values.append(float(
                                    (selected_q.max() - selected_q.min()).cpu()
                                ))
                            if not bool(torch.all(torch.isfinite(direct))):
                                raise FloatingPointError(
                                    "non-finite all-action help value"
                                )
                            weight = direct
                            clipped = direct
                        else:
                            # The first data block only fits Task-Q.  Randomly
                            # initialized all-action values must never become reward.
                            weight = torch.zeros(
                                len(selected), dtype=torch.double,
                                device=self.device,
                            )
                            clipped = weight
                    else:
                        weight, clipped = self._sampled_action_receiver_weight(
                            heard_log, deaf_log, actions,
                            float(self.config.help_weight_clip),
                        )
                raw_np = weight.cpu().numpy()
                clipped_np = clipped.cpu().numpy()
                raw_weights.extend(raw_np.tolist())
                clipped_weights.extend(clipped_np.tolist())
                if not use_all_action_q:
                    clip_count += int(np.sum(
                        raw_np < -float(self.config.help_weight_clip)
                    ))
                for row, value in zip(selected, clipped_np):
                    if use_all_action_q:
                        help_value = float(value)
                        direct_values.append(help_value)
                    else:
                        baseline = (
                            float(row['receiver_value'])
                            if self.config.help_receiver_value_baseline else 0.0
                        )
                        help_value = float(value) * (
                            float(row['q_task']) - baseline
                        )
                    if row['role'] == 'partner':
                        partner_bonus[row['episode_index']][
                            row['sender_index']
                        ] += help_value
                        partner_values.append(help_value)
                    else:
                        opponent_bonus[row['episode_index']][
                            row['sender_index']
                        ] += help_value
                        opponent_values.append(help_value)

        partner_totals = np.asarray(
            [sum(episode) for episode in partner_bonus], dtype=np.float64
        )
        partner_total_std = float(partner_totals.std(ddof=1)) if len(
            partner_totals
        ) > 1 else 0.0
        # The sampled-action estimator needs a frozen cap because its
        # importance ratio has a negative tail.  The all-action estimator has
        # no such tail; normalize it per chunk so a newly opening belief
        # channel is neither numerically silent nor allowed to grow unchecked.
        should_calibrate_scale = (
            use_all_action_q or self.help_scale_factor is None
        )
        if should_calibrate_scale and partner_total_std > 1e-12:
            task_rewards = np.asarray(
                self._episode_task_rewards(episodes), dtype=np.float64
            )
            task_std = max(float(task_rewards.std(ddof=1)), 1.0)
            ratio = task_std / max(partner_total_std, 1e-12)
            if not use_all_action_q:
                ratio = min(ratio, 1000.0)
            self.help_scale_factor = ratio * float(self.config.help_reward_weight)
        scale = float(self.help_scale_factor or 0.0)
        bonuses = [
            [
                scale * (partner - float(self.config.help_beta) * opponent)
                for partner, opponent in zip(partner_ep, opponent_ep)
            ]
            for partner_ep, opponent_ep in zip(partner_bonus, opponent_bonus)
        ]

        raw = np.asarray(raw_weights, dtype=np.float64)
        clipped = np.asarray(clipped_weights, dtype=np.float64)
        shaped = np.asarray(
            [value for episode in bonuses for value in episode if value != 0.0],
            dtype=np.float64,
        )
        metadata = {
            'receiver_event_count': int(raw.size),
            'rewarded_sender_count': int(shaped.size),
            'raw_weight_mean': float(raw.mean()) if raw.size else 0.0,
            'raw_weight_std': float(raw.std()) if raw.size else 0.0,
            'raw_weight_abs_max': float(np.max(np.abs(raw))) if raw.size else 0.0,
            'clip_count': int(clip_count),
            'clip_fraction': float(clip_count / raw.size) if raw.size else 0.0,
            'shaped_reward_mean': float(shaped.mean()) if shaped.size else 0.0,
            'shaped_reward_std': float(shaped.std()) if shaped.size else 0.0,
            'shaped_reward_abs_max': (
                float(np.max(np.abs(shaped))) if shaped.size else 0.0
            ),
            'partner_help_mean_imp': (
                float(np.mean(partner_values)) if partner_values else 0.0
            ),
            'opponent_help_mean_imp': (
                float(np.mean(opponent_values)) if opponent_values else 0.0
            ),
            'help_beta': float(self.config.help_beta),
            'help_reward_weight': float(self.config.help_reward_weight),
            'frozen_scale_factor': scale,
            'scale_mode': (
                'per_chunk_partner_total_std'
                if use_all_action_q else 'frozen_capped_mc'
            ),
            'help_weight_clip': float(self.config.help_weight_clip),
            'baseline': (
                'detached_receiver_state_value'
                if self.config.help_receiver_value_baseline
                else 'zero_action_independent'
            ),
            'q_source': (
                'online_structured_all_action_task_q'
                if use_all_action_q else 'terminal_task_imp_mc'
            ),
            'all_action_q': use_all_action_q,
            'task_q_ready': q_ready,
            'task_q_samples_seen': int(self.help_task_q_samples_seen),
            'task_q_updates': int(self.help_task_q_updates),
            'direct_help_abs_mean_imp': (
                float(np.mean(np.abs(direct_values))) if direct_values else 0.0
            ),
            'heard_deaf_policy_l1_mean': (
                float(np.mean(heard_deaf_l1_values))
                if heard_deaf_l1_values else 0.0
            ),
            'legal_q_span_mean_imp': (
                float(np.mean(legal_q_span_values))
                if legal_q_span_values else 0.0
            ),
            'return_equivalent': bool(self.config.help_return_equivalent),
        }
        self.help_reward_metadata = metadata
        return bonuses, metadata

    @staticmethod
    def _sampled_action_receiver_weight(
        heard_log_probs: torch.Tensor,
        deaf_log_probs: torch.Tensor,
        actions: torch.Tensor,
        negative_clip: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw/clipped ``1-pi_deaf(a)/pi_heard(a)`` weights."""
        if heard_log_probs.shape != deaf_log_probs.shape:
            raise ValueError("heard/deaf log-probability shapes must match")
        if actions.shape != (heard_log_probs.shape[0],):
            raise ValueError("one sampled action is required per policy row")
        if negative_clip <= 0:
            raise ValueError("negative importance-weight clip must be positive")
        chosen_heard = heard_log_probs.gather(1, actions[:, None]).squeeze(1)
        chosen_deaf = deaf_log_probs.gather(1, actions[:, None]).squeeze(1)
        log_ratio = chosen_deaf - chosen_heard
        if not bool(torch.all(torch.isfinite(log_ratio))):
            raise FloatingPointError("non-finite help importance weight")
        # w<=1 but has an unbounded negative tail. Evaluate a finite diagnostic
        # copy and impose the declared lower clip in log space.
        raw = -torch.expm1(torch.clamp(log_ratio, max=50.0))
        clipped = -torch.expm1(torch.minimum(
            log_ratio,
            torch.full_like(log_ratio, math.log1p(negative_clip)),
        ))
        return raw, clipped

    @staticmethod
    def _all_action_receiver_help(
        heard_log_probs: torch.Tensor,
        deaf_log_probs: torch.Tensor,
        q_values: torch.Tensor,
        legal_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exact COMA-style receiver path value for one all-action Task-Q."""
        if not (
            heard_log_probs.shape == deaf_log_probs.shape == q_values.shape
            == legal_action_mask.shape
        ):
            raise ValueError("all-action help tensors must have identical shapes")
        legal = legal_action_mask.to(torch.bool)
        safe_q = q_values.masked_fill(~legal, 0.0)
        return (
            (heard_log_probs.exp() - deaf_log_probs.exp()) * safe_q
        ).sum(dim=-1)

    def _help_task_q_batch(self, rows: List[dict]) -> Dict[str, torch.Tensor]:
        """Build one actor-relative CTDE batch for the online help Task-Q."""
        observations = np.stack([row['receiver_obs'] for row in rows])
        legal = np.stack([row['legal_actions'] for row in rows]).astype(np.float32)
        hands = np.stack([
            normalize_ctde_hands(
                row['all_hands'], int(row['receiver_player']),
                CTDE_SEAT_ORDER_ACTING_RELATIVE,
            ) for row in rows
        ])
        action_features = np.stack([
            build_structured_action_features(
                row['public_history_before'], int(row['dealer']),
                int(row['receiver_player']), row['vulnerability'],
                row['dd_table'], row['legal_actions'],
            ) for row in rows
        ])
        dd_raw = torch.as_tensor(
            np.stack([row['dd_table'] for row in rows]),
            dtype=torch.float32, device=self.device,
        )
        ref_raw = torch.as_tensor(
            [[row['reference_score_ns']] for row in rows],
            dtype=torch.float32, device=self.device,
        )
        return {
            'observations': torch.as_tensor(
                observations, dtype=torch.float32, device=self.device),
            'all_hands_ctde': torch.as_tensor(
                hands, dtype=torch.float32, device=self.device),
            'legal_action_mask': torch.as_tensor(
                legal, dtype=torch.float32, device=self.device),
            'dd_table_ctde': normalize_dd_table_ctde(dd_raw),
            'reference_score_ctde': normalize_reference_score_ctde(ref_raw),
            'action_features_ctde': torch.as_tensor(
                action_features, dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(
                [row['action'] for row in rows], dtype=torch.long,
                device=self.device),
            'terminal_duplicate_dds_imp': torch.as_tensor(
                [row['q_task'] for row in rows], dtype=torch.float32,
                device=self.device),
        }

    def _train_help_task_q(self, episodes: List[List[dict]]) -> dict:
        """Fit chosen receiver actions on task-only terminal IMP after reward use."""
        if self.help_task_q is None or self.help_task_q_optimizer is None:
            return {}

        terminal_by_episode = []
        for episode in episodes:
            terminal = {}
            for step in reversed(episode):
                partnership = int(step['player']) % 2
                if partnership not in terminal and bool(step.get('done')):
                    terminal[partnership] = float(step['reward'])
            if set(terminal) != {0, 1}:
                raise RuntimeError("Task-Q training episode lacks both task IMP labels")
            terminal_by_episode.append(terminal)

        rows = []
        for ep_index, episode in enumerate(episodes):
            for step in episode:
                player = int(step['player'])
                rows.append({
                    'receiver_obs': step['obs_571'],
                    'legal_actions': step['legal_actions'],
                    'action': int(step['action']),
                    'q_task': terminal_by_episode[ep_index][player % 2],
                    'all_hands': step['all_hands'],
                    'dd_table': step['dd_table'],
                    'reference_score_ns': step['reference_score_ns'],
                    'dealer': step['dealer'],
                    'vulnerability': step['vulnerability'],
                    'public_history_before': step['public_history_before'],
                    'receiver_player': player,
                })

        self.help_task_q.train()
        losses = []
        absolute_errors = []
        squared_errors = []
        batch_size = max(1, int(self.config.help_task_q_batch_size))
        for _ in range(max(1, int(self.config.help_task_q_epochs))):
            for start in range(0, len(rows), batch_size):
                batch = self._help_task_q_batch(rows[start:start + batch_size])
                self.help_task_q_optimizer.zero_grad(set_to_none=True)
                q_values = self.help_task_q(
                    batch['observations'], batch['all_hands_ctde'],
                    batch['legal_action_mask'],
                    dd_table_ctde=batch['dd_table_ctde'],
                    reference_score_ctde=batch['reference_score_ctde'],
                    action_features_ctde=batch['action_features_ctde'],
                )
                prediction = q_values.gather(
                    1, batch['actions'][:, None]
                ).squeeze(1)
                target = batch['terminal_duplicate_dds_imp']
                error = prediction.detach() - target
                absolute_errors.extend(error.abs().cpu().tolist())
                squared_errors.extend(error.square().cpu().tolist())
                loss = F.smooth_l1_loss(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.help_task_q.parameters(), 5.0)
                self.help_task_q_optimizer.step()
                losses.append(float(loss.detach().cpu()))
                self.help_task_q_updates += 1
        self.help_task_q_samples_seen += len(rows)
        self.help_task_q.eval()
        return {
            'task_q_train_samples': len(rows),
            'task_q_train_loss': float(np.mean(losses)) if losses else 0.0,
            'task_q_chosen_mae_preupdate': (
                float(np.mean(absolute_errors)) if absolute_errors else 0.0
            ),
            'task_q_chosen_rmse_preupdate': (
                float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0
            ),
            'task_q_samples_seen_after': int(self.help_task_q_samples_seen),
            'task_q_updates_after': int(self.help_task_q_updates),
        }

    @staticmethod
    def _apply_return_equivalent_help(
        episodes: List[List[dict]],
        bonuses: List[List[float]],
        *,
        tolerance: float = 1e-6,
    ) -> dict:
        """Move task return to sender steps without changing any seat's return.

        Each training player's help credits are added at that player's sender
        transitions and subtracted from the same player's final transition.
        Consequently every PPO buffer sees exactly the original undiscounted
        episode return, while GAE receives earlier task-credit evidence.
        """
        if len(episodes) != len(bonuses):
            raise ValueError("episode/bonus batch length mismatch")

        max_error = 0.0
        transferred_abs = 0.0
        corrected_players = 0
        for episode, episode_bonuses in zip(episodes, bonuses):
            if len(episode) != len(episode_bonuses):
                raise ValueError("episode/bonus step length mismatch")

            training_players = {
                int(step['player'])
                for step in episode
                if bool(step.get('is_training_side'))
            }
            before = {
                player: math.fsum(
                    float(step['reward']) for step in episode
                    if int(step['player']) == player
                )
                for player in training_players
            }
            last_index = {
                player: max(
                    index for index, step in enumerate(episode)
                    if int(step['player']) == player
                )
                for player in training_players
            }
            transfers = {player: 0.0 for player in training_players}

            for step, bonus in zip(episode, episode_bonuses):
                value = float(bonus)
                if value == 0.0:
                    continue
                player = int(step['player'])
                if player not in training_players:
                    raise RuntimeError(
                        "return-equivalent help was routed to a non-training player"
                    )
                step['reward'] = float(step['reward']) + value
                transfers[player] += value
                transferred_abs += abs(value)

            for player, transfer in transfers.items():
                if transfer == 0.0:
                    continue
                terminal = episode[last_index[player]]
                if not bool(terminal.get('done')):
                    raise RuntimeError(
                        f"player {player} has no terminal transition for help correction"
                    )
                terminal['reward'] = float(terminal['reward']) - transfer
                corrected_players += 1

            for player in training_players:
                after = math.fsum(
                    float(step['reward']) for step in episode
                    if int(step['player']) == player
                )
                error = abs(after - before[player])
                max_error = max(max_error, error)
                if error > tolerance:
                    raise RuntimeError(
                        "return-equivalent help conservation failed: "
                        f"player={player} before={before[player]} after={after}"
                    )

        return {
            'reward_conservation_max_abs_error': float(max_error),
            'transferred_reward_abs_sum': float(transferred_abs),
            'corrected_player_count': int(corrected_players),
        }

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
            (
                self._compute_raw_info_potential_bonus(
                    ns_eps, ns_data, beta_override=0.0
                )
                + self._compute_raw_info_potential_bonus(
                    ew_eps, ew_data, beta_override=0.0
                )
                if self.config.info_potential_shaping
                else self._compute_raw_info_bonus(
                    ns_eps, ns_data, beta_override=0.0
                ) + self._compute_raw_info_bonus(
                    ew_eps, ew_data, beta_override=0.0
                )
            )
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
            'reward_mode': (
                'strict_per_seat_potential_v1'
                if self.config.info_potential_shaping
                else 'legacy_immediate_delta_v1'
            ),
            'gamma': float(self.config.gamma),
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

    def _auction_health(self, episodes: List[List[dict]]) -> dict:
        """Small structured health summary for pilot and long-run monitoring."""
        if not episodes:
            return {
                'num_auctions': 0,
                'mean_length': 0.0,
                'median_length': 0.0,
                'p95_length': 0.0,
                'competitive_rate': 0.0,
                'all_pass_rate': 0.0,
                'double_call_rate': 0.0,
                'redouble_call_rate': 0.0,
            }

        prefix_actions = list(getattr(self.env, 'initial_history_actions', []))
        prefix_len = len(prefix_actions)
        lengths = np.asarray([len(ep) + prefix_len for ep in episodes])
        competitive = 0
        all_pass = 0
        doubles = 0
        redoubles = 0
        calls = 0
        for ep in episodes:
            dealer = int(ep[0].get('dealer', NORTH)) if ep else NORTH
            sides = {
                (dealer + index) % 2
                for index, action in enumerate(prefix_actions)
                if action >= BID_1C
            }
            sides.update({
                step['player'] % 2
                for step in ep
                if step['action'] >= BID_1C
            })
            actions = prefix_actions + [step['action'] for step in ep]
            competitive += int(len(sides) == 2)
            all_pass += int(actions and all(action == BID_PASS for action in actions))
            doubles += sum(step['action'] == BID_DOUBLE for step in ep)
            redoubles += sum(step['action'] == BID_REDOUBLE for step in ep)
            calls += len(actions)

        n = len(episodes)
        return {
            'num_auctions': n,
            'mean_length': float(lengths.mean()),
            'median_length': float(np.median(lengths)),
            'p95_length': float(np.percentile(lengths, 95)),
            'competitive_rate': competitive / n,
            'all_pass_rate': all_pass / n,
            'double_call_rate': doubles / max(calls, 1),
            'redouble_call_rate': redoubles / max(calls, 1),
        }

    def _new_auction_health_accumulator(self) -> dict:
        return {
            'lengths': [], 'competitive': 0, 'all_pass': 0,
            'doubles': 0, 'redoubles': 0, 'calls': 0,
        }

    def _accumulate_auction_health(self, accumulator: dict,
                                   episodes: List[List[dict]]) -> None:
        """Accumulate exact health statistics while rollout chunks are freed."""
        prefix_actions = list(getattr(self.env, 'initial_history_actions', []))
        prefix_len = len(prefix_actions)
        for ep in episodes:
            accumulator['lengths'].append(len(ep) + prefix_len)
            dealer = int(ep[0].get('dealer', NORTH)) if ep else NORTH
            sides = {
                (dealer + index) % 2
                for index, action in enumerate(prefix_actions)
                if action >= BID_1C
            }
            sides.update({
                step['player'] % 2 for step in ep
                if step['action'] >= BID_1C
            })
            actions = prefix_actions + [step['action'] for step in ep]
            accumulator['competitive'] += int(len(sides) == 2)
            accumulator['all_pass'] += int(
                bool(actions) and all(action == BID_PASS for action in actions)
            )
            accumulator['doubles'] += sum(
                step['action'] == BID_DOUBLE for step in ep
            )
            accumulator['redoubles'] += sum(
                step['action'] == BID_REDOUBLE for step in ep
            )
            accumulator['calls'] += len(actions)

    @staticmethod
    def _finalize_auction_health(accumulator: dict) -> dict:
        lengths = np.asarray(accumulator['lengths'])
        n = len(lengths)
        if not n:
            return {
                'num_auctions': 0, 'mean_length': 0.0,
                'median_length': 0.0, 'p95_length': 0.0,
                'competitive_rate': 0.0, 'all_pass_rate': 0.0,
                'double_call_rate': 0.0, 'redouble_call_rate': 0.0,
            }
        return {
            'num_auctions': n,
            'mean_length': float(lengths.mean()),
            'median_length': float(np.median(lengths)),
            'p95_length': float(np.percentile(lengths, 95)),
            'competitive_rate': accumulator['competitive'] / n,
            'all_pass_rate': accumulator['all_pass'] / n,
            'double_call_rate': accumulator['doubles'] / max(accumulator['calls'], 1),
            'redouble_call_rate': accumulator['redoubles'] / max(accumulator['calls'], 1),
        }

    def copy_info_scale_from(self, other: "SubgameTrainer") -> None:
        if other.info_scale_factor is None:
            raise RuntimeError("Source trainer has no calibrated information scale")
        self.info_scale_factor = float(other.info_scale_factor)
        self.info_scale_metadata = dict(other.info_scale_metadata or {})

    # ======================================================================
    # PPO Update
    # ======================================================================

    def enable_task_actor_gradient_capture(self, enabled: bool = True) -> None:
        """Capture one pure, pre-clipping task policy gradient per seat."""
        self.capture_task_actor_gradients = bool(enabled)
        if not enabled:
            self._last_task_actor_gradients.clear()
            self._last_task_actor_gradient_metadata.clear()

    def captured_task_actor_gradient(self, player: int) -> Optional[torch.Tensor]:
        gradient = self._last_task_actor_gradients.get(int(player))
        return None if gradient is None else gradient.clone()

    def captured_task_actor_gradient_metadata(self, player: int) -> dict:
        return dict(self._last_task_actor_gradient_metadata.get(int(player), {}))

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
            last_obs, last_hands = buf.last_inputs()
            fo = last_obs.unsqueeze(0).to(self.device)
            ah = (last_hands.unsqueeze(0).to(self.device)
                  if last_hands is not None else None)
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

                # Normalize advantage - guard against degenerate batches
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
                    continue   # skip batch - NaN loss would corrupt weights

                if (
                    self.capture_task_actor_gradients
                    and player not in self._last_task_actor_gradients
                ):
                    parameters = [
                        parameter for parameter in actor.parameters()
                        if parameter.requires_grad
                    ]
                    pure_policy_gradients = torch.autograd.grad(
                        policy_loss, parameters, retain_graph=True,
                        allow_unused=True,
                    )
                    pieces = [
                        torch.zeros_like(parameter).reshape(-1).cpu()
                        if gradient is None
                        else gradient.detach().reshape(-1).cpu()
                        for parameter, gradient in zip(
                            parameters, pure_policy_gradients, strict=True
                        )
                    ]
                    if pieces:
                        self._last_task_actor_gradients[player] = torch.cat(pieces)
                        self._last_task_actor_gradient_metadata[player] = {
                            "definition": "pure_task_policy_loss_pre_clipping_v2",
                            "raw_advantage_mean": float(b_adv.mean().item()),
                            "raw_advantage_std": float(adv_std.item()),
                            "normalized_advantage_mean": float(adv.mean().item()),
                            "normalized_advantage_std": float(adv.std().item()),
                            "event_count": int(b_adv.numel()),
                        }
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
            h2h_deals: int = 500, start_round: int = 0,
            skip_warmup: bool = False, round_callback=None) -> List[dict]:
        """See the formal README for the current behavior contract."""
        num_rounds = self.config.num_rounds if num_rounds is None else num_rounds
        if not 0 <= start_round <= num_rounds:
            raise ValueError(
                f"start_round must be in [0, {num_rounds}], got {start_round}"
            )
        n_deals    = self.config.steps_per_phase * self.config.deals_per_step
        batch_sz   = self.config.deals_per_step
        cfg = self.config
        chunk_deals = min(n_deals, max(batch_sz, int(cfg.rollout_chunk_deals)))
        if cfg.rollout_chunk_deals <= 0:
            raise ValueError("rollout_chunk_deals must be positive")
        print(f"[Config] kl_lambda_start={cfg.kl_lambda_start}  kl_lambda_end={cfg.kl_lambda_end}  kl_anneal_frac={cfg.kl_anneal_frac}")
        print(f"[Config] num_rounds={num_rounds}  deals_per_step={cfg.deals_per_step}  n_deals_per_phase={n_deals}  rollout_chunk_deals={chunk_deals}  use_info_bonus={cfg.use_info_bonus}  info_potential_shaping={cfg.info_potential_shaping}  beta={cfg.beta}  info_weight={cfg.info_reward_weight}  use_help_bonus={cfg.use_help_bonus}  help_beta={cfg.help_beta}  help_weight={cfg.help_reward_weight}  help_return_equivalent={cfg.help_return_equivalent}  help_receiver_value_baseline={cfg.help_receiver_value_baseline}")

        if not skip_warmup:
            print("\n[Trainer] Critic warmup...")
            import time as _time
            warmup_started = _time.time()
            self.critic_warmup()
            print(f"  [Timing] critic_warmup={_time.time() - warmup_started:.1f}s")
        else:
            print(f"\n[Trainer] Resuming at round {start_round + 1}; critic warmup skipped.")

        if not self.config.self_play and not self._fsp_seeded and self.config.fsp_pool_size >= 0:
            self.fsp_pool.add_permanent(self.agent)
            self._fsp_seeded = True
            print(f"  [FSP] Seeded pool with BC checkpoint as permanent (pool size: {len(self.fsp_pool)})")
        if self.config.self_play:
            print("  [Self-Play] Pure self-play mode: opponent = current agent (FSP disabled)")

        for rnd in range(start_round, num_rounds):
            import time as _time
            round_started = _time.time()
            if self.capture_task_actor_gradients:
                self._last_task_actor_gradients.clear()
                self._last_task_actor_gradient_metadata.clear()
            print(f"\n====== Round {rnd+1}/{num_rounds} ======")

            _bw = False

            if self.config.self_play:
                fsp_sd = None
            else:
                self._maybe_add_to_fsp(rnd)
                fsp_sd = self._apply_fsp_opponent()
            _ir_vals: list = []; _bl_vals: list = []
            ns_metrics: dict = {}; ew_metrics: dict = {}

            # -- (P93: JIT burn-in removed - belief update moved to after PPO) --

            health_acc = self._new_auction_health_accumulator()
            belief_eps = []

            def _collect_phase(train_side: str, players: Tuple[int, int]):
                raw_vals = []
                remaining = n_deals
                timing = {'environment': 0.0, 'health_and_task': 0.0,
                          'information': 0.0, 'buffer_pack': 0.0}
                while remaining:
                    current = min(chunk_deals, remaining)
                    started = _time.time()
                    eps, rinfo = self._collect_episodes_batch(
                        current, train_side=train_side, fsp_sd=fsp_sd,
                        batch_size=batch_sz, use_belief_prior=_bw)
                    timing['environment'] += _time.time() - started
                    started = _time.time()
                    self._accumulate_auction_health(health_acc, eps)
                    for ep in eps:
                        for step in ep:
                            if step.get('done') and step['player'] in players:
                                value = float(step['reward'])
                                self.reward_stats.update(value)
                                raw_vals.append(value)
                                break
                    timing['health_and_task'] += _time.time() - started
                    if self.config.use_info_bonus:
                        started = _time.time()
                        if self.config.info_potential_shaping:
                            raw_bonuses = self._compute_raw_info_potential_bonus(
                                eps, rinfo_data=rinfo
                            )
                            bonuses = [
                                [value * self.info_scale_factor for value in episode]
                                for episode in raw_bonuses
                            ]
                        else:
                            bonuses = self._compute_info_bonus(eps, rinfo_data=rinfo)
                        for ep, episode_bonuses in zip(eps, bonuses):
                            for step, bonus in zip(ep, episode_bonuses):
                                step['reward'] += bonus
                                if bonus != 0.0:
                                    _ir_vals.append(bonus)
                        timing['information'] += _time.time() - started
                    elif self.config.use_help_bonus:
                        started = _time.time()
                        bonuses, help_meta = self._compute_help_bonus(
                            eps, fsp_sd=fsp_sd
                        )
                        if self.config.help_all_action_q:
                            # Reward inference uses the frozen pre-chunk Q;
                            # current task labels update Q only afterwards.
                            help_meta.update(self._train_help_task_q(eps))
                            self.help_reward_metadata = help_meta
                        if self.config.help_return_equivalent:
                            help_meta.update(
                                self._apply_return_equivalent_help(eps, bonuses)
                            )
                        else:
                            for ep, episode_bonuses in zip(eps, bonuses):
                                for step, bonus in zip(ep, episode_bonuses):
                                    step['reward'] += bonus
                        for episode_bonuses in bonuses:
                            _ir_vals.extend(
                                bonus for bonus in episode_bonuses if bonus != 0.0
                            )
                        timing['information'] += _time.time() - started
                        print(
                            "  [Help Reward] "
                            f"receivers={help_meta['receiver_event_count']} "
                            f"senders={help_meta['rewarded_sender_count']} "
                            f"std={help_meta['shaped_reward_std']:.4f} "
                            f"scale={help_meta['frozen_scale_factor']:.4f} "
                            f"clip={help_meta['clip_fraction']:.4%} "
                            f"return_eq={help_meta['return_equivalent']} "
                            f"q_ready={help_meta.get('task_q_ready', False)} "
                            f"q_seen={help_meta.get('task_q_samples_seen_after', 0)} "
                            f"conservation={help_meta.get('reward_conservation_max_abs_error', 0.0):.2e}"
                        )
                    started = _time.time()
                    player_steps = {player: [] for player in players}
                    for ep in eps:
                        for step in ep:
                            if step['player'] in player_steps:
                                player_steps[step['player']].append(step)
                    for player in players:
                        self.buffers[player].add_steps(player_steps[player])
                    timing['buffer_pack'] += _time.time() - started
                    if self.belief_net is not None and not self.config.freeze_belief:
                        belief_eps.extend(eps)
                    remaining -= current
                    del eps, rinfo
                return raw_vals, timing

            print(f"  [Table1/NS] Collecting {n_deals} deals (batch={batch_sz}, chunk={chunk_deals})...")
            _t = _time.time()
            raw_ns_vals, ns_timing = _collect_phase('NS', (NORTH, SOUTH))
            started = _time.time()
            for p in (NORTH, SOUTH):
                m = self._safe_update(p, rnd)
                if m: ns_metrics[p] = m
            ns_timing['ppo'] = _time.time() - started

            print(f"  [Table2/EW] Collecting {n_deals} deals (batch={batch_sz}, chunk={chunk_deals})...")
            _t = _time.time()
            raw_ew_vals, ew_timing = _collect_phase('EW', (EAST, WEST))
            started = _time.time()
            for p in (EAST, WEST):
                m = self._safe_update(p, rnd)
                if m: ew_metrics[p] = m
            ew_timing['ppo'] = _time.time() - started

            # -- Belief Update: frozen (P96) or on-policy (P93/P95) ---------
            _t = _time.time()
            belief_started = _time.time()
            if self.belief_net is not None and not self.config.freeze_belief:
                bl = self.update_belief_on_policy(belief_eps)
                if bl is not None: _bl_vals.append(bl)
            belief_seconds = _time.time() - belief_started

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
            auction_health = self._finalize_auction_health(health_acc)
            timing_seconds = {
                'ns': ns_timing, 'ew': ew_timing,
                'belief_update': belief_seconds,
                'round_compute_total': _time.time() - round_started,
            }

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
                'auction_health': auction_health,
                'timing_seconds': timing_seconds,
            }
            self.log.append(log_entry)
            self._print_log(log_entry)

            # P110: Per-round H2H removed - too noisy (500-deal IMP stdapproximately9).
            # All statistical evaluation is deferred to Stage 3 (5000 deals).

            should_stop = False
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
                        should_stop = True

            if round_callback is not None:
                round_callback(self, rnd + 1, log_entry)
            if should_stop:
                break

        return self.log

    def _print_log(self, entry: dict):
        rnd = entry['round']
        fsp = entry['fsp_pool_size']
        mr  = entry['mean_reward']
        sr  = entry['std_reward']
        ns_r = entry.get('mean_ns_task_imp', 0.0)
        ew_r = entry.get('mean_ew_task_imp', 0.0)

        print(f"  [Round {rnd}] rollout_task_IMP={mr:+.3f}+/-{sr:.3f}  (NS={ns_r:+.3f} EW={ew_r:+.3f})  fsp={fsp}")

        health = entry.get('auction_health') or {}
        timing = entry.get('timing_seconds') or {}
        if timing:
            ns_t = timing.get('ns', {})
            ew_t = timing.get('ew', {})
            print(
                "  [Timing] "
                f"NS env={ns_t.get('environment', 0):.1f}s "
                f"info={ns_t.get('information', 0):.1f}s "
                f"pack={ns_t.get('buffer_pack', 0):.1f}s "
                f"ppo={ns_t.get('ppo', 0):.1f}s | "
                f"EW env={ew_t.get('environment', 0):.1f}s "
                f"info={ew_t.get('information', 0):.1f}s "
                f"pack={ew_t.get('buffer_pack', 0):.1f}s "
                f"ppo={ew_t.get('ppo', 0):.1f}s | "
                f"round={timing.get('round_compute_total', 0):.1f}s"
            )
        if health:
            print(
                "    auction_health: "
                f"len={health['mean_length']:.2f} "
                f"p95={health['p95_length']:.1f} "
                f"competitive={health['competitive_rate']:.1%} "
                f"all_pass={health['all_pass_rate']:.1%} "
                f"dbl={health['double_call_rate']:.2%} "
                f"rdbl={health['redouble_call_rate']:.2%}"
            )

        ns = entry.get('ns_metrics', {})
        ns_n = ns.get(NORTH, {}); ns_s = ns.get(SOUTH, {})
        if ns_n or ns_s:
            print(f"    NS | N: pl={ns_n.get('policy_loss',0):+.4f} "
                  f"vl={ns_n.get('value_loss',0):.3f} "
                  f"ent={ns_n.get('entropy',0):.3f} | "
                  f"S: pl={ns_s.get('policy_loss',0):+.4f} "
                  f"vl={ns_s.get('value_loss',0):.3f} "
                  f"ent={ns_s.get('entropy',0):.3f} "
                  f"kl={ns_s.get('kl_loss',0):.5f}(lambda={ns_s.get('kl_lambda',0):.3f})")

        ew = entry.get('ew_metrics', {})
        ew_e = ew.get(EAST, {}); ew_w = ew.get(WEST, {})
        if ew_e or ew_w:
            print(f"    EW | E: pl={ew_e.get('policy_loss',0):+.4f} "
                  f"vl={ew_e.get('value_loss',0):.3f} "
                  f"ent={ew_e.get('entropy',0):.3f} | "
                  f"W: pl={ew_w.get('policy_loss',0):+.4f} "
                  f"vl={ew_w.get('value_loss',0):.3f} "
                  f"ent={ew_w.get('entropy',0):.3f} "
                  f"kl={ew_e.get('kl_loss',0):.5f}(lambda={ew_e.get('kl_lambda',0):.3f})")

        ir = entry.get('mean_ir')
        bl = entry.get('belief_loss')
        actor_bl = entry.get('actor_belief_loss')
        if actor_bl is not None:
            print(f"    actor_belief_loss={actor_bl:.4f}")
        if ir is not None:
            bl_str = f"{bl:.4f}" if bl is not None else "N/A"
            print(f"    r_info | step_ir={ir:.4f}  belief_loss={bl_str}")
