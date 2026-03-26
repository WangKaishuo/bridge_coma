"""
Subgame Validation — Competitive Path (P100/P101)
==================================================

P100 key change: BCA (Belief-Conditioned Actor) is now the STANDARD BASELINE
for ALL agents. BCA is not an experimental treatment — it is the minimum
capability that any bridge bidding agent should possess. An agent that cannot
interpret bidding history is not "playing bridge" in any meaningful sense.

P101 key change: Actor input expanded to 397-dim:
  301 (base obs) + 48 (partner belief) + 48 (RHO belief) = 397
  This enables agents to understand BOTH partner and opponent bids.

Experiment design:
  - Agent A: MAPPO + BCA (control)
  - Agent B: MAPPO + BCA + r_info (treatment)
  - SL baseline: BCA (no RL training)
  - Only difference between A and B: r_info reward shaping

Training mode (default):
    python experiments/subgame_validation.py \\
        --data path/to/competitive_500k.npz \\
        --sl_checkpoint results/sl_base_bca.pt \\
        --seed 42 --rounds 10

Eval-only mode (paired evaluation on saved checkpoints):
    python experiments/subgame_validation.py \\
        --eval-only \\
        --data data/competitive_500k.npz \\
        --sl_checkpoint results/sl_base_bca.pt \\
        --agent_a results/competitive/agent_a_seed42.pt \\
        --agent_b results/competitive/agent_b_seed42.pt \\
        --num_deals 2000 --seed 42

Legacy 301-dim mode (backward compatible):
    python experiments/subgame_validation.py \\
        --no_belief_conditioned \\
        --sl_checkpoint results/sl_base.pt \\
        --data data/competitive_500k.npz \\
        --seed 42

Training diagnostics (README §Phase 1):
    ✅  ir > 0（info gain 有效）
    ✅  entropy 不坍塌（> 0.5）
    ✅  KL anchor 可控（< 0.3）
    ✅  value_loss 收敛（无爆炸）
"""

import argparse
import os
import random
import sys
from pathlib import Path

# 将 bridge_coma 根目录加入 sys.path，确保包结构可被找到
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from scipy.stats import wilcoxon

from subgames.competitive_env import CompetitiveSubgameEnv, make_agent_policy
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
from utils.running_stats import RunningStats
from utils.imp import score_to_imp
from networks.policy_net import OBS_DIM, BELIEF_OBS_DIM, load_sl_into_mappo_agent
from algorithms.mappo import MAPPOAgent, MAPPOConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_competitive(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 公共参数（先算，后打印）─────────────────────────────────────────
    # P100: BCA is now the standard baseline for ALL agents.
    # --belief_conditioned is always True; --no_belief_conditioned opts out.
    _bc = not getattr(args, 'no_belief_conditioned', False)
    if args.kl_lambda is not None:
        _kl = args.kl_lambda
    else:
        _kl = 0.0 if _bc else 0.3

    # P98b: BCA mode uses anneal schedule (0.3→0.0) instead of fixed 0.0
    # This prevents policy collapse while belief net adapts to RL distribution
    if _bc and args.kl_lambda is None:
        _kl_start = 0.3
        _kl_end   = 0.0
        _kl_anneal = 0.5   # anneal over first 50% of rounds
    else:
        _kl_start = _kl
        _kl_end   = _kl
        _kl_anneal = 0.0

    _ent_coef = getattr(args, 'entropy_coef', 0.01)

    print(f"\n[Competitive Subgame] seed={args.seed}  device={device}")
    print(f"  beta={args.beta}  info_weight={args.info_weight}  rounds={args.rounds}  quick={args.quick}")
    print(f"  belief_conditioned={_bc}  kl_lambda={_kl_start}→{_kl_end}  entropy_coef={_ent_coef}")
    if _bc:
        print(f"  [P101] BCA standard: ALL agents use 397-dim belief-conditioned actors (partner + RHO)")

    # ── 环境 ────────────────────────────────────────────────────────────────
    print("\n[1] Initializing environment...")
    env = CompetitiveSubgameEnv(data_path=args.data)

    # ── 公共参数 ───────────────────────────────────────────────────────

    # ── Agent A（控制组：MAPPO + BCA，no r_info）──────────────────────────
    print("\n[2] Building Agent A (MAPPO + BCA, no r_info)...")
    cfg_a = SubgameConfig(
        num_rounds       = args.rounds,
        steps_per_phase  = 10  if args.quick else 64,
        deals_per_step   = 32  if args.quick else 512,
        lr               = 3e-6,
        batch_size       = 256,
        use_info_bonus   = False,
        beta             = 0.0,
        info_reward_weight = 0.0,
        fsp_pool_size    = 10,
        fsp_add_interval = 1,
        kl_lambda_start  = _kl_start,
        kl_lambda_end    = _kl_end,
        kl_anneal_frac   = _kl_anneal,
        entropy_coef     = _ent_coef,
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5    if args.quick else 20,
        device           = device,
        # P100: BCA is standard for ALL agents — the only difference is r_info
        belief_conditioned = _bc,
        # P99/P100: Both A and B use identical belief settings for fair comparison.
        freeze_belief    = False if _bc else True,
        belief_update_epochs = 1,
        belief_update_lr  = 1e-5,
        belief_warmup_rounds = 0,
    )
    reward_stats_a = RunningStats()
    trainer_a = SubgameTrainer(env, cfg_a, reward_stats=reward_stats_a)

    # ── Agent B（实验组：MAPPO + BCA + r_info）────────────────────
    _a_only = getattr(args, 'agent_a_only', False)
    trainer_b = None
    cfg_b = None
    reward_stats_b = None
    if not _a_only:
        print(f"\n[3] Building Agent B (MAPPO + BCA + r_info, β={args.beta}, w={args.info_weight})...")
        cfg_b = SubgameConfig(
            # P100: identical to A except use_info_bonus, beta, info_reward_weight
            num_rounds       = args.rounds,
            steps_per_phase  = 10  if args.quick else 64,
            deals_per_step   = 32  if args.quick else 512,
            lr               = 3e-6,
            batch_size       = 256,
            use_info_bonus   = True,
            beta             = args.beta,
            info_reward_weight = args.info_weight,
            fsp_pool_size    = 10,
            fsp_add_interval = 1,
            kl_lambda_start  = _kl_start,
            kl_lambda_end    = _kl_end,
            kl_anneal_frac   = _kl_anneal,
            entropy_coef     = _ent_coef,
            bc_warmup_samples= 1000 if args.quick else 5000,
            bc_warmup_epochs = 5    if args.quick else 20,
            device           = device,
            # P100: BCA is standard for ALL agents
            belief_conditioned = _bc,
            freeze_belief    = False,
            use_ewc          = False,
            belief_update_epochs = 1,
            belief_update_lr  = 1e-5,
            belief_warmup_rounds = 0,
        )
        reward_stats_b = RunningStats()
        trainer_b = SubgameTrainer(env, cfg_b, reward_stats=reward_stats_b)
    else:
        print(f"\n[3] Skipping Agent B (--agent_a_only mode)")

    # ── Agent C（实验组：MAPPO + BCA + r_info + β opponent penalty）──────
    _beta_c = getattr(args, 'beta_c', 0.05)
    _enable_c = not getattr(args, 'no_agent_c', False) and not _a_only
    trainer_c = None
    cfg_c = None
    if _enable_c:
        print(f"\n[4] Building Agent C (MAPPO + BCA + r_info, β={_beta_c}, w={args.info_weight})...")
        cfg_c = SubgameConfig(
            # P100: identical to B except beta > 0 (opponent penalty active)
            num_rounds       = args.rounds,
            steps_per_phase  = 10  if args.quick else 64,
            deals_per_step   = 32  if args.quick else 512,
            lr               = 3e-6,
            batch_size       = 256,
            use_info_bonus   = True,
            beta             = _beta_c,
            info_reward_weight = args.info_weight,
            fsp_pool_size    = 10,
            fsp_add_interval = 1,
            kl_lambda_start  = _kl_start,
            kl_lambda_end    = _kl_end,
            kl_anneal_frac   = _kl_anneal,
            entropy_coef     = _ent_coef,
            bc_warmup_samples= 1000 if args.quick else 5000,
            bc_warmup_epochs = 5    if args.quick else 20,
            device           = device,
            belief_conditioned = _bc,
            freeze_belief    = False,
            use_ewc          = False,
            belief_update_epochs = 1,
            belief_update_lr  = 1e-5,
            belief_warmup_rounds = 0,
        )
        reward_stats_c = RunningStats()
        trainer_c = SubgameTrainer(env, cfg_c, reward_stats=reward_stats_c)

    # ── Stage 1: 初始化（SL checkpoint 优先，否则 rule-based BC）──────────────
    from env import NORTH as _N, EAST as _E, SOUTH as _S, WEST as _W, NUM_PLAYERS
    sl_path = getattr(args, 'sl_checkpoint', None)

    # 确定哪些trainer需要SL初始化
    _all_trainers = [trainer_a]
    if trainer_b is not None:
        _all_trainers.append(trainer_b)
    if trainer_c is not None:
        _all_trainers.append(trainer_c)
    trainers_to_init = [t for t in _all_trainers if t is not None]
    if args.load_agent_a:
        trainers_to_init = [t for t in trainers_to_init if t is not trainer_a]
    is_bca_ckpt = False  # default, updated below if BCA checkpoint detected

    if sl_path and os.path.exists(sl_path):
        print(f"\n[Stage 1] Loading SL checkpoint: {sl_path}")
        ckpt = torch.load(sl_path, map_location=device, weights_only=False)
        sl_obs_dim = ckpt.get('obs_dim', OBS_DIM)
        sl_encoding = ckpt.get('encoding', 'legacy')
        is_p105 = ('model_state' in ckpt)  # P105 format: single shared model_state
        is_bca_ckpt = (sl_obs_dim == BELIEF_OBS_DIM)

        print(f"  SL format: {'P105' if is_p105 else 'legacy'}  "
              f"obs_dim={sl_obs_dim}  encoding={sl_encoding}")

        if is_p105:
            # ── P105 format: model_state → all 4 actors ─────────────────
            sl_sd = {k: v.to(device) for k, v in ckpt['model_state'].items()}
            for trainer in trainers_to_init:
                for player in range(NUM_PLAYERS):
                    actor = trainer.agent.get_actor(player)
                    if _bc and sl_obs_dim < BELIEF_OBS_DIM:
                        # 571→667: zero-init belief feature columns
                        target_sd = actor.state_dict()
                        for param_name, sl_val in sl_sd.items():
                            if param_name in target_sd:
                                tgt_shape = target_sd[param_name].shape
                                if sl_val.shape == tgt_shape:
                                    target_sd[param_name] = sl_val
                                elif param_name == 'net.0.weight' and len(sl_val.shape) == 2:
                                    target_sd[param_name][:, :sl_val.shape[1]] = sl_val
                                    target_sd[param_name][:, sl_val.shape[1]:] = 0.0
                                else:
                                    target_sd[param_name] = sl_val
                        actor.load_state_dict(target_sd)
                    else:
                        actor.load_state_dict(sl_sd)
        else:
            # ── Legacy format: actor_n, actor_s, actor_e, actor_w ────────
            player_key_map = [(_N, 'actor_n'), (_E, 'actor_e'),
                              (_S, 'actor_s'), (_W, 'actor_w')]

            if _bc and is_bca_ckpt:
                print(f"  [BCA] BCA checkpoint detected (obs_dim={sl_obs_dim})")

            for trainer in trainers_to_init:
                for player, key in player_key_map:
                    if key not in ckpt:
                        continue
                    sl_sd = {k: v.to(device) for k, v in ckpt[key].items()}
                    actor = trainer.agent.get_actor(player)

                    if not _bc:
                        if is_bca_ckpt:
                            print(f"  ⚠️  Skipping {key}: BCA checkpoint incompatible with non-BCA agent")
                            continue
                        actor.load_state_dict(sl_sd)
                    elif is_bca_ckpt:
                        actor.load_state_dict(sl_sd)
                    else:
                        target_sd = actor.state_dict()
                        for param_name, sl_val in sl_sd.items():
                            if param_name in target_sd:
                                tgt_shape = target_sd[param_name].shape
                                if sl_val.shape == tgt_shape:
                                    target_sd[param_name] = sl_val
                                elif param_name == 'net.0.weight' and len(sl_val.shape) == 2:
                                    target_sd[param_name][:, :sl_val.shape[1]] = sl_val
                                    target_sd[param_name][:, sl_val.shape[1]:] = 0.0
                                else:
                                    target_sd[param_name] = sl_val
                        actor.load_state_dict(target_sd)

                # Load belief_net from legacy BCA checkpoint if available
                if _bc and 'belief_net' in ckpt and trainer.belief_net is not None:
                    ckpt_belief_sd = {k: v.to(device) for k, v in ckpt['belief_net'].items()}
                    ckpt_hidden = ckpt.get(
                        'belief_hidden_dim',
                        ckpt_belief_sd['trunk.0.weight'].shape[0]
                    )
                    if ckpt_hidden != trainer.belief_net.trunk[0].out_features:
                        from networks.belief_net import BeliefNetwork
                        trainer.belief_net = BeliefNetwork(hidden_dim=ckpt_hidden).to(device)
                        print(f"  Belief Net rebuilt with hidden_dim={ckpt_hidden}")
                    trainer.belief_net.load_state_dict(ckpt_belief_sd)
                    _trainer_name = 'A' if trainer is trainer_a else ('B' if trainer is trainer_b else 'C')
                    print(f"  Belief Net loaded for {_trainer_name}")

        _n_agents = len(trainers_to_init)
        init_names = f"{_n_agents} agents" if _n_agents > 1 else "1 agent"
        dim_note = (f" [P105 {sl_obs_dim}-dim]" if is_p105 else
                    f" [BCA {sl_obs_dim}-dim]" if is_bca_ckpt else "")
        print(f"  [SL Init] Weights loaded for all actors ({init_names}).{dim_note}")

        # ── P99: Zero-init belief feature columns in first layer ────────────
        # BCA SL pretrain trains all 397 input columns jointly, so the first
        # layer weight[:, 301:397] encodes belief-feature dependencies learned
        # on GLOBAL SAYC data. On the competitive subgame, belief features have
        # a different distribution → extreme logit shifts → entropy collapse.
        # Fix: zero out the 48 belief columns so the actor initially behaves
        # identically to 301-dim SL. Belief influence enters gradually via RL.
        # This preserves the BCA architecture while avoiding distribution shift.
        if _bc and is_bca_ckpt:
            _base_dim = OBS_DIM  # 301
            for trainer in trainers_to_init:
                for player, key in player_key_map:
                    actor = trainer.agent.get_actor(player)
                    with torch.no_grad():
                        w = actor.net[0].weight  # (hidden_dim, 397)
                        w[:, _base_dim:] = 0.0
                        b = actor.net[0].bias    # not affected, but confirm
                    # Also zero the corresponding columns in bc_anchor later
            print(f"  [P99] Zeroed belief-feature columns in first layer "
                  f"(all actors, both agents)")
    else:
        print("\n[Stage 1] BC Warmup (rule-based, SL checkpoint not found)...")
        for trainer in trainers_to_init:
            trainer.run_bc_warmup()

    # Stage 1 结束：将当前 actor 设为 KL anchor
    for trainer in trainers_to_init:
        trainer.set_bc_anchor(trainer.agent)
    print(f"  [KL Anchor] BC anchor set for {len(trainers_to_init)} agents.")

    # ── Stage 1.5: Belief Net 独立预训练 ──────────────────────────────
    belief_pretrain_rounds = getattr(args, 'belief_pretrain_rounds', 5)
    deals = 200 if args.quick else 2000

    # P98: when belief_conditioned, ALL agents need belief net pretrained
    # But skip if already loaded from BCA checkpoint
    trainers_for_belief = []
    if belief_pretrain_rounds > 0:
        _bn_from_ckpt = _bc and is_bca_ckpt and 'belief_net' in (ckpt if sl_path and os.path.exists(sl_path) else {})
        for name, trainer in [('A', trainer_a), ('B', trainer_b), ('C', trainer_c)]:
            if trainer is None:
                continue
            if name == 'A' and args.load_agent_a:
                continue
            if trainer.belief_net is not None and not _bn_from_ckpt:
                trainers_for_belief.append((name, trainer))

    for name, trainer in trainers_for_belief:
        print(f"\n[Stage 1.5] Belief Net Pretrain (Agent {name}, "
              f"total {belief_pretrain_rounds * deals} deals, early stopping)...")
        trainer.pretrain_belief(
            num_rounds=belief_pretrain_rounds,
            deals_per_round=deals,
            epochs_per_round=5,
            max_epochs=getattr(args, 'belief_pretrain_max_epochs', 300),
        )

    # ── P99: Seed belief replay buffer for BCA-loaded trainers ────────────
    # When belief net is loaded from BCA checkpoint (skipping pretrain_belief),
    # _pretrain_replay is never created. Without replay protection,
    # update_belief_on_policy destroys the pretrained belief net
    # (catastrophic forgetting: val_loss 1.9 → 7.26 in Round 1).
    # Fix: collect SL-policy rollouts to seed the replay buffer.
    # FAIRNESS: both agents get identical replay data (same deals, same policy)
    # since their actors are identical at this point (both loaded from BCA ckpt).
    if _bc and is_bca_ckpt and 'belief_net' in (ckpt if sl_path and os.path.exists(sl_path) else {}):
        _all_named = [('A', trainer_a)]
        if trainer_b is not None:
            _all_named.append(('B', trainer_b))
        if trainer_c is not None:
            _all_named.append(('C', trainer_c))
        _need_replay = [
            (name, trainer) for name, trainer in _all_named
            if (trainer.belief_net is not None
                and not trainer.config.freeze_belief
                and not hasattr(trainer, '_pretrain_replay'))
        ]
        if _need_replay:
            _replay_deals = 100 if args.quick else 2000
            # Collect once using trainer_a (identical policy to B at this point)
            _ref_trainer = _need_replay[0][1]
            print(f"\n  [P99] Seeding belief replay ({_replay_deals} SL-policy deals, "
                  f"shared by {', '.join(n for n, _ in _need_replay)})...")
            _replay_eps = _ref_trainer._collect_episodes_batch(
                _replay_deals, train_side='NS', fsp_sd=None,
                batch_size=min(512, _replay_deals), skip_dual_table=True)
            _bd = []
            for ep in _replay_eps:
                for step in ep:
                    if 'belief_target' in step and not step.get('ew_diagnostic'):
                        _bd.append(step)
            if _bd:
                import torch as _t
                _shared_replay = {
                    'oh':  _t.tensor(np.stack([s['observer_hand'] for s in _bd]),
                                     dtype=_t.float32),
                    'h':   _t.tensor(np.stack([s['history']       for s in _bd]),
                                     dtype=_t.float32),
                    'op':  _t.tensor(np.array([s['observer_pos']  for s in _bd]),
                                     dtype=_t.long),
                    'tp':  _t.tensor(np.array([s['target_pos']    for s in _bd]),
                                     dtype=_t.long),
                    'tgt': _t.tensor(np.stack([s['belief_target'] for s in _bd]),
                                     dtype=_t.float32),
                }
                for name, trainer in _need_replay:
                    trainer._pretrain_replay = _shared_replay
                print(f"  [P99] Saved {len(_bd)} belief replay samples "
                      f"(shared by {', '.join(n for n, _ in _need_replay)})")
            else:
                print(f"  [P99] WARNING: No belief data collected for replay seeding")

    # ── Build SL baseline trainer for mini eval during training ──────────────
    from algorithms.mappo import MAPPOAgent, MAPPOConfig
    from networks.policy_net import BELIEF_OBS_DIM as _BOD

    _sl_bc = _bc  # SL gets belief net whenever agents do
    _sl_obs_dim = _BOD if _sl_bc else OBS_DIM

    cfg_sl = SubgameConfig(
        device=device, belief_conditioned=_sl_bc, use_info_bonus=False,
        kl_lambda_start=0.0, kl_lambda_end=0.0,
        belief_warmup_rounds=0,
    )
    sl_agent_eval = MAPPOAgent(MAPPOConfig(device=device, obs_dim=_sl_obs_dim))
    if sl_path and os.path.exists(sl_path):
        ckpt_sl = torch.load(sl_path, map_location=device, weights_only=False)
        _is_p105_sl = ('model_state' in ckpt_sl)
        sl_obs_dim_ckpt = ckpt_sl.get('obs_dim', OBS_DIM)

        if _is_p105_sl:
            # P105 format: single model_state → all players
            sl_sd = {k: v.to(device) for k, v in ckpt_sl['model_state'].items()}
            for player in range(NUM_PLAYERS):
                actor = sl_agent_eval.get_actor(player)
                if _sl_bc and sl_obs_dim_ckpt < _BOD:
                    # 571→667: zero-init belief columns
                    target_sd = actor.state_dict()
                    for param_name, sl_val in sl_sd.items():
                        if param_name in target_sd:
                            tgt_shape = target_sd[param_name].shape
                            if sl_val.shape == tgt_shape:
                                target_sd[param_name] = sl_val
                            elif param_name == 'net.0.weight' and len(sl_val.shape) == 2:
                                target_sd[param_name][:, :sl_val.shape[1]] = sl_val
                                target_sd[param_name][:, sl_val.shape[1]:] = 0.0
                            else:
                                target_sd[param_name] = sl_val
                    actor.load_state_dict(target_sd)
                else:
                    actor.load_state_dict(sl_sd)
        else:
            # Legacy format
            for player, key in [(_N, 'actor_n'), (_E, 'actor_e'),
                                (_S, 'actor_s'), (_W, 'actor_w')]:
                if key not in ckpt_sl:
                    continue
                sl_sd = {k: v.to(device) for k, v in ckpt_sl[key].items()}
                actor = sl_agent_eval.get_actor(player)
                if _sl_bc and sl_obs_dim_ckpt != _BOD:
                    target_sd = actor.state_dict()
                    for param_name, sl_val in sl_sd.items():
                        if param_name in target_sd:
                            tgt_shape = target_sd[param_name].shape
                            if sl_val.shape == tgt_shape:
                                target_sd[param_name] = sl_val
                            elif param_name == 'net.0.weight' and len(sl_val.shape) == 2:
                                target_sd[param_name][:, :sl_val.shape[1]] = sl_val
                                target_sd[param_name][:, sl_val.shape[1]:] = 0.0
                            else:
                                target_sd[param_name] = sl_val
                    actor.load_state_dict(target_sd)
                else:
                    actor.load_state_dict(sl_sd)
    sl_trainer = SubgameTrainer(env, cfg_sl, reward_stats=RunningStats())
    sl_trainer.agent = sl_agent_eval

    # P99: Pretrain SL's belief net (same procedure as A/B, independent instance)
    if _sl_bc and sl_trainer.belief_net is not None:
        print(f"\n  [SL Eval] Pretraining SL baseline belief net "
              f"({belief_pretrain_rounds * deals} deals)...")
        sl_trainer.pretrain_belief(
            num_rounds=belief_pretrain_rounds,
            deals_per_round=deals,
            epochs_per_round=5,
            max_epochs=getattr(args, 'belief_pretrain_max_epochs', 300),
        )
        print(f"  [SL Eval] Belief-conditioned SL baseline ({_sl_obs_dim}-dim)")

    # ── Stage 2: RL 微调 ────────────────────────────────────────────────────
    print("\n[Stage 2] RL Fine-tuning...")

    if args.skip_training:
        # Load saved checkpoints instead of training
        a_path = os.path.join(args.save_dir, f'agent_a_seed{args.seed}.pt')
        print(f"  ── Skip training, loading checkpoints ──")
        print(f"  Agent A: {a_path}")
        trainer_a.agent.load(a_path)
        log_a, log_b = [], []
        if trainer_b is not None:
            b_path = os.path.join(args.save_dir, f'agent_b_seed{args.seed}.pt')
            print(f"  Agent B: {b_path}")
            trainer_b.agent.load(b_path)
        log_c = []
        if trainer_c is not None:
            c_path = os.path.join(args.save_dir, f'agent_c_seed{args.seed}.pt')
            print(f"  Agent C: {c_path}")
            trainer_c.agent.load(c_path)
    elif args.load_agent_a:
        print(f"  ── Agent A (loading from {args.load_agent_a}) ──")
        trainer_a.agent.load(args.load_agent_a)
        log_a = []
        print(f"  [Agent A] Loaded from checkpoint. Training skipped.")
    else:
        print("  ── Agent A ──")
        log_a = trainer_a.run(num_rounds=args.rounds, sl_trainer=sl_trainer)

    print("\n  ── Agent B ──")
    log_b = []
    if trainer_b is not None:
        log_b = trainer_b.run(num_rounds=args.rounds, sl_trainer=sl_trainer)
    else:
        print("  [Skipped — agent_a_only mode]")

    log_c = []
    if trainer_c is not None and not args.skip_training:
        print("\n  ── Agent C ──")
        log_c = trainer_c.run(num_rounds=args.rounds, sl_trainer=sl_trainer)

    # ── Stage 3: 评估 ───────────────────────────────────────────────────────
    print("\n[Stage 3] Evaluation...")
    h2h_deals = 200 if args.quick else args.eval_deals

    # sl_trainer already built before Stage 2

    # ── A vs SL ────────────────────────────────────────────────────────────
    print("\n  → Agent A vs SL baseline")
    h2h_a_sl = trainer_a.evaluate_head_to_head(
        sl_trainer, num_deals=h2h_deals, label_self="A", label_other="SL")

    # ── B vs SL ────────────────────────────────────────────────────────────
    h2h_b_sl = None
    h2h_ab = None
    if trainer_b is not None:
        print("\n  → Agent B vs SL baseline")
        h2h_b_sl = trainer_b.evaluate_head_to_head(
            sl_trainer, num_deals=h2h_deals, label_self="B", label_other="SL")

        # ── A vs B ─────────────────────────────────────────────────────────────
        print("\n  → Agent A vs Agent B")
        h2h_ab = trainer_a.evaluate_head_to_head(
            trainer_b, num_deals=h2h_deals, label_self="A", label_other="B")

    # ── Agent C evaluations ───────────────────────────────────────────────
    h2h_c_sl = h2h_ac = h2h_bc = None
    if trainer_c is not None:
        print("\n  → Agent C vs SL baseline")
        h2h_c_sl = trainer_c.evaluate_head_to_head(
            sl_trainer, num_deals=h2h_deals, label_self="C", label_other="SL")

        print("\n  → Agent A vs Agent C")
        h2h_ac = trainer_a.evaluate_head_to_head(
            trainer_c, num_deals=h2h_deals, label_self="A", label_other="C")

        if trainer_b is not None:
            print("\n  → Agent B vs Agent C")
            h2h_bc = trainer_b.evaluate_head_to_head(
                trainer_c, num_deals=h2h_deals, label_self="B", label_other="C")

    # Belief Net 评估
    for name, trainer, cfg in [('A', trainer_a, cfg_a), ('B', trainer_b, cfg_b), ('C', trainer_c, cfg_c)]:
        if trainer is not None and cfg is not None and cfg.use_info_bonus:
            print(f"\n  → Belief Network Evaluation (Agent {name}):")
            trainer.evaluate_belief(num_deals=50 if args.quick else 200)

    # ── P97c: Partner Info Gain diagnostic ──
    # Only run when B exists (need B's belief net as judge)
    if trainer_b is not None and cfg_b is not None and cfg_b.use_info_bonus and trainer_b.belief_net is not None:
        diag_deals = 100 if args.quick else 500
        print(f"\n  → Partner Info Gain Diagnostic ({diag_deals} deals)")

        # Judge: B's belief net (for A vs B comparison)
        print(f"\n  [Judge: B's belief net]")
        print("  Agent A (no r_info):")
        pig_a = trainer_a.evaluate_partner_info_gain(
            trainer_b.belief_net, num_deals=diag_deals)
        print("  Agent B (r_info, β=0):")
        pig_b = trainer_b.evaluate_partner_info_gain(
            trainer_b.belief_net, num_deals=diag_deals)
        diff_ba = pig_b['mean_partner_gain'] - pig_a['mean_partner_gain']
        print(f"\n  [Δ partner_gain] B - A = {diff_ba:+.4f}"
              f"  ({'B communicates more' if diff_ba > 0 else 'A communicates more'})")

        if trainer_c is not None and trainer_c.belief_net is not None:
            print(f"\n  [Judge: C's belief net]")
            print(f"  Agent C (r_info, β={_beta_c}):")
            pig_c = trainer_c.evaluate_partner_info_gain(
                trainer_c.belief_net, num_deals=diag_deals)
            diff_ca = pig_c['mean_partner_gain'] - pig_a['mean_partner_gain']
            diff_cb = pig_c['mean_partner_gain'] - pig_b['mean_partner_gain']
            print(f"\n  [Δ partner_gain] C - A = {diff_ca:+.4f}")
            print(f"  [Δ partner_gain] C - B = {diff_cb:+.4f}")

    # ── 排雷诊断 ────────────────────────────────────────────────────────────
    print("\n[Diagnostics]")
    if cfg_b is not None:
        _print_diagnostics(log_a, log_b, cfg_b, log_c=log_c, cfg_c=cfg_c)
    else:
        _print_diagnostics(log_a, [], cfg_a, log_c=log_c, cfg_c=cfg_c)

    # ── 最终摘要 ────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    _mode_label = "agent_a_only (drift sweep)" if _a_only else "P100: BCA standard for all agents"
    print(f"  FINAL SUMMARY ({_mode_label})")
    print("═" * 72)
    print(f"  A vs SL:  {h2h_a_sl['mean_imp']:+.3f} ± {h2h_a_sl['std_imp']:.3f} IMP  "
          f"p={h2h_a_sl['p_value']:.3f} {'✅' if h2h_a_sl['significant'] else '(ns)'}")
    if h2h_b_sl:
        print(f"  B vs SL:  {h2h_b_sl['mean_imp']:+.3f} ± {h2h_b_sl['std_imp']:.3f} IMP  "
              f"p={h2h_b_sl['p_value']:.3f} {'✅' if h2h_b_sl['significant'] else '(ns)'}")
    if h2h_c_sl:
        print(f"  C vs SL:  {h2h_c_sl['mean_imp']:+.3f} ± {h2h_c_sl['std_imp']:.3f} IMP  "
              f"p={h2h_c_sl['p_value']:.3f} {'✅' if h2h_c_sl['significant'] else '(ns)'}")
    if h2h_ab or h2h_ac or h2h_bc:
        print(f"  ──────────────────────────────────────────────")
    if h2h_ab:
        print(f"  A vs B:   {h2h_ab['mean_imp']:+.3f} ± {h2h_ab['std_imp']:.3f} IMP  "
              f"p={h2h_ab['p_value']:.3f} {'✅' if h2h_ab['significant'] else '(ns)'}")
    if h2h_ac:
        print(f"  A vs C:   {h2h_ac['mean_imp']:+.3f} ± {h2h_ac['std_imp']:.3f} IMP  "
              f"p={h2h_ac['p_value']:.3f} {'✅' if h2h_ac['significant'] else '(ns)'}")
    if h2h_bc:
        print(f"  B vs C:   {h2h_bc['mean_imp']:+.3f} ± {h2h_bc['std_imp']:.3f} IMP  "
              f"p={h2h_bc['p_value']:.3f} {'✅' if h2h_bc['significant'] else '(ns)'}")

    # Conclusion
    print(f"  ──────────────────────────────────────────────")
    _results = [('A', h2h_a_sl)]
    if h2h_b_sl:
        _results.append(('B', h2h_b_sl))
    if h2h_c_sl:
        _results.append(('C', h2h_c_sl))
    best_agent = max(_results, key=lambda x: x[1]['mean_imp'])
    print(f"  Best vs SL: Agent {best_agent[0]} ({best_agent[1]['mean_imp']:+.3f} IMP)")

    # Core hypothesis: B > A? C > B?
    if h2h_ab:
        if h2h_ab['mean_imp'] < 0 and h2h_ab['significant']:
            print(f"  ✅ B > A: r_info (partner-only) helps with BCA")
        elif h2h_ab['mean_imp'] > 0 and h2h_ab['significant']:
            print(f"  ⚠️  A > B: r_info hurts with BCA")
        else:
            print(f"  — A ≈ B: r_info has no significant effect with BCA")

    if h2h_bc:
        if h2h_bc['mean_imp'] > 0 and h2h_bc['significant']:
            print(f"  ✅ B > C: partner-only r_info better than dual (β hurts)")
        elif h2h_bc['mean_imp'] < 0 and h2h_bc['significant']:
            print(f"  ✅ C > B: β opponent penalty adds value")
        else:
            print(f"  — B ≈ C: β opponent penalty has no significant effect")
    print("═" * 72)

    # ── 保存 checkpoint ─────────────────────────────────────────────────────
    if args.save_dir:
        import json
        os.makedirs(args.save_dir, exist_ok=True)
        trainer_a.agent.save(os.path.join(args.save_dir, f'agent_a_seed{args.seed}.pt'))
        if trainer_b is not None:
            trainer_b.agent.save(os.path.join(args.save_dir, f'agent_b_seed{args.seed}.pt'))
        if trainer_c is not None:
            trainer_c.agent.save(os.path.join(args.save_dir, f'agent_c_seed{args.seed}.pt'))
        report = {
            'seed': args.seed, 'beta_b': 0.0, 'beta_c': _beta_c if trainer_c else None,
            'rounds': args.rounds,
            'a_vs_sl': h2h_a_sl,
            'a_vs_sl_imp': h2h_a_sl['mean_imp'] if h2h_a_sl else None,
            'b_vs_sl': h2h_b_sl,
            'c_vs_sl': h2h_c_sl,
            'a_vs_b': h2h_ab,
            'a_vs_c': h2h_ac,
            'b_vs_c': h2h_bc,
        }
        report_path = os.path.join(args.save_dir, f'report_seed{args.seed}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Checkpoints + report saved → {args.save_dir}")


def _print_diagnostics(log_a: list, log_b: list, cfg_b: SubgameConfig,
                       log_c: list = None, cfg_c: SubgameConfig = None):
    """排雷判断：检查训练健康性指标."""

    def _last_metric(log: list, player_key: int, metric: str) -> float:
        for entry in reversed(log):
            for key in ('ns_metrics', 'ew_metrics', 's_metrics', 'n_metrics'):
                m = entry.get(key, {}).get(player_key, {})
                if metric in m:
                    return m[metric]
        return float('nan')

    from env import SOUTH, EAST

    # Agent A（可能从checkpoint加载，无训练日志）
    if log_a:
        ent_a   = _last_metric(log_a, SOUTH, 'entropy')
        vl_a    = _last_metric(log_a, SOUTH, 'value_loss')
        ent_a_e = _last_metric(log_a, EAST, 'entropy')
        vl_a_e  = _last_metric(log_a, EAST, 'value_loss')
        print(f"  Agent A: NS entropy={ent_a:.3f} vl={vl_a:.4f} │ EW entropy={ent_a_e:.3f} vl={vl_a_e:.4f}")
    else:
        print(f"  Agent A: (loaded from checkpoint, no training log)")

    # Agent B
    ent_b   = _last_metric(log_b, SOUTH, 'entropy')
    vl_b    = _last_metric(log_b, SOUTH, 'value_loss')
    kl_b    = _last_metric(log_b, SOUTH, 'kl_loss')
    klam_b  = _last_metric(log_b, SOUTH, 'kl_lambda')
    ent_b_e = _last_metric(log_b, EAST, 'entropy')
    vl_b_e  = _last_metric(log_b, EAST, 'value_loss')
    print(f"  Agent B: NS entropy={ent_b:.3f} vl={vl_b:.4f} kl={kl_b:.5f}(λ={klam_b:.3f}) │ EW entropy={ent_b_e:.3f} vl={vl_b_e:.4f}")

    # Agent C
    if log_c:
        ent_c   = _last_metric(log_c, SOUTH, 'entropy')
        vl_c    = _last_metric(log_c, SOUTH, 'value_loss')
        kl_c    = _last_metric(log_c, SOUTH, 'kl_loss')
        klam_c  = _last_metric(log_c, SOUTH, 'kl_lambda')
        ent_c_e = _last_metric(log_c, EAST, 'entropy')
        vl_c_e  = _last_metric(log_c, EAST, 'value_loss')
        print(f"  Agent C: NS entropy={ent_c:.3f} vl={vl_c:.4f} kl={kl_c:.5f}(λ={klam_c:.3f}) │ EW entropy={ent_c_e:.3f} vl={vl_c_e:.4f}")

    ok = True
    for name, log, cfg in [('B', log_b, cfg_b), ('C', log_c, cfg_c)]:
        if not log or cfg is None:
            continue
        _ent = _last_metric(log, SOUTH, 'entropy')
        _vl  = _last_metric(log, SOUTH, 'value_loss')
        _kl  = _last_metric(log, SOUTH, 'kl_loss')
        if _ent < 0.5:
            print(f"  ⚠️  Agent {name}: Entropy collapse detected! Consider higher entropy_coef.")
            ok = False
        if _vl > 10:
            print(f"  ⚠️  Agent {name}: Value loss too high! Consider more critic warmup.")
            ok = False
        if _kl > 0.3:
            print(f"  ⚠️  Agent {name}: KL too high! Consider higher kl_lambda_start.")
            ok = False

        if cfg.use_info_bonus:
            ir_vals = [entry.get('info_ratio', None) for entry in log if 'info_ratio' in entry]
            if ir_vals:
                mean_ir = np.mean(ir_vals)
                print(f"  Agent {name}: mean ir={mean_ir:.4f}")
                if mean_ir <= 0:
                    print(f"  ⚠️  Agent {name}: ir ≤ 0! Check Belief Net and r_info wiring.")
                    ok = False

    if ok:
        print("  ✅ All diagnostics passed.")
    else:
        print("  ❌ Fix above issues before running Phase 3.")


# ==============================================================================
# Eval-Only Mode (absorbed from eval_paired.py, P93)
# ==============================================================================

def _load_agent(path: str, device: str) -> MAPPOAgent:
    """Load a saved MAPPOAgent checkpoint."""
    agent = MAPPOAgent(MAPPOConfig(device=device))
    agent.load(path)
    return agent


def _load_sl_as_agent(sl_path: str, device: str) -> MAPPOAgent:
    """Load SL checkpoint (P105 or legacy) as a MAPPOAgent."""
    agent = MAPPOAgent(MAPPOConfig(device=device, obs_dim=OBS_DIM))
    load_sl_into_mappo_agent(agent, sl_path)
    return agent


def _make_eval_policy(agent: MAPPOAgent, env: CompetitiveSubgameEnv, device: str):
    """
    Wrap a MAPPOAgent as a 3-arg policy(obs, player, history_int) → action_int.

    P105: Uses OpenSpiel state.observation_tensor() for 571-dim observations.
    Reconstructs OpenSpiel state from (hands, dealer, history) each step.

    env.dealer and env._eval_hands_rm must be set before each play_mixed call.
    """
    from networks.policy_net import (
        hands_to_openspiel_state, get_openspiel_obs, ours_to_openspiel_raw,
    )

    def policy(obs, player, history_int):
        os_state = hands_to_openspiel_state(env._eval_hands_rm, env.dealer)
        for a in history_int:
            os_a = ours_to_openspiel_raw(a)
            if os_a >= 0:
                os_state.apply_action(os_a)

        flat = get_openspiel_obs(os_state)
        flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
        legal  = torch.tensor(obs['legal_actions'], dtype=torch.float32
                              ).unsqueeze(0).to(device)
        actor  = agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
        return action.item()
    return policy


def _cross_eval_fixed_deals(env, deals, pol_a, pol_b):
    """
    Paired double-dummy evaluation on a fixed set of deals.

    Table 1: A = opener (NS), B = overcaller (EW)
    Table 2: B = opener (NS), A = overcaller (EW)
    IMP = score_to_imp(score_1 − score_2)  — positive means A wins.

    P105: env._eval_hands_rm must be set for OpenSpiel observation generation.
    env.dealer must be set before each play_mixed call.
    """
    from networks.policy_net import convert_hands_suit_to_rank

    imps = []
    for hands, dd_table, dealer, vul in deals:
        env.dealer = dealer
        env._eval_hands_rm = convert_hands_suit_to_rank(hands)
        _, score_1, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=pol_a, ew_policy=pol_b,
            vulnerability=vul, dealer=dealer)
        _, score_2, _ = env.play_mixed(
            hands, dd_table,
            ns_policy=pol_b, ew_policy=pol_a,
            vulnerability=vul, dealer=dealer)
        imps.append(float(score_to_imp(score_1 - score_2)))
    return np.array(imps)


def _print_matchup(label_a: str, label_b: str, imps: np.ndarray):
    """Print one matchup result line + return (mean, std, p)."""
    try:
        _, p = wilcoxon(imps)
    except Exception:
        p = 1.0
    m, s, wr = imps.mean(), imps.std(), (imps > 0).mean()
    sig = p < 0.05
    verdict = (
        f"✅ {label_a} wins" if m > 0 and sig
        else f"✅ {label_b} wins" if m < 0 and sig
        else "— ns"
    )
    print(f"  {label_a:8s} vs {label_b:8s}: "
          f"IMP={m:+.3f}±{s:.3f}  wr={wr:.1%}  p={p:.4f}  {verdict}")
    return m, s, p


def run_eval_only(args):
    """
    Paired evaluation mode: load saved checkpoints and compare A vs B vs SL
    on the same set of deals. Outputs Wilcoxon p-values and a paired diff test.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Eval-Only] device={device}  n={args.num_deals}  seed={args.seed}")

    env = CompetitiveSubgameEnv(data_path=args.data)

    # ── Load agents ──────────────────────────────────────────────────
    print("\n[Loading]")
    sl_agent = _load_sl_as_agent(args.sl_checkpoint, device)
    print(f"  SL:      {args.sl_checkpoint}")
    agent_a = _load_agent(args.agent_a, device)
    print(f"  Agent A: {args.agent_a}")
    agent_b = _load_agent(args.agent_b, device)
    print(f"  Agent B: {args.agent_b}")

    pol_sl = _make_eval_policy(sl_agent, env, device)
    pol_a  = _make_eval_policy(agent_a,  env, device)
    pol_b  = _make_eval_policy(agent_b,  env, device)

    # ── Sample fixed deal pool ────────────────────────────────────────
    print(f"\n[Sampling {args.num_deals} deals from {args.data}]")
    deals = []
    for _ in range(args.num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul = [(False, False), (True, False),
               (False, True),  (True, True)][np.random.randint(4)]
        deals.append((hands.copy(), dd_table.copy(), dealer, vul))
    print(f"  {len(deals)} deals sampled.")

    # ── Three matchups on identical deals ────────────────────────────
    print(f"\n[Paired H2H on {args.num_deals} identical deals]")
    print("=" * 72)
    imps_a_sl = _cross_eval_fixed_deals(env, deals, pol_a, pol_sl)
    m1, _, p1 = _print_matchup("Agent_A", "SL", imps_a_sl)

    imps_b_sl = _cross_eval_fixed_deals(env, deals, pol_b, pol_sl)
    m2, _, p2 = _print_matchup("Agent_B", "SL", imps_b_sl)

    imps_a_b = _cross_eval_fixed_deals(env, deals, pol_a, pol_b)
    m3, _, p3 = _print_matchup("Agent_A", "Agent_B", imps_a_b)
    print("=" * 72)

    # ── Paired diff: A_vs_SL − B_vs_SL ──────────────────────────────
    print(f"\n[Paired diff: A_vs_SL − B_vs_SL, per-deal]")
    diff = imps_a_sl - imps_b_sl
    se_d = diff.std() / np.sqrt(len(diff))
    try:
        _, p_paired = wilcoxon(diff)
    except Exception:
        p_paired = 1.0
    print(f"  mean diff = {diff.mean():+.3f}  std = {diff.std():.3f}  SE = {se_d:.3f}")
    print(f"  Wilcoxon p = {p_paired:.4f}  {'✅ sig' if p_paired < 0.05 else '(ns)'}")
    if diff.mean() > 0 and p_paired < 0.05:
        print("  → A significantly stronger than B vs SL")
    elif diff.mean() < 0 and p_paired < 0.05:
        print("  → B significantly stronger than A vs SL")
    else:
        print("  → No significant difference in vs-SL strength")

    corr = np.corrcoef(imps_a_sl, imps_b_sl)[0, 1]
    print(f"  Corr(A_vs_SL, B_vs_SL) = {corr:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  A vs SL:     {imps_a_sl.mean():+.3f} ± {imps_a_sl.std():.3f}  p={p1:.4f}")
    print(f"  B vs SL:     {imps_b_sl.mean():+.3f} ± {imps_b_sl.std():.3f}  p={p2:.4f}")
    print(f"  A vs B:      {imps_a_b.mean():+.3f} ± {imps_a_b.std():.3f}  p={p3:.4f}")
    print(f"  Paired diff: {diff.mean():+.3f}  p={p_paired:.4f}")
    print(f"  Correlation: {corr:.3f}")
    print(f"{'='*72}")


def parse_args():
    parser = argparse.ArgumentParser(description='Bridge-COMA Subgame Validation')
    parser.add_argument('--type', default='competitive',
                        choices=['competitive'],
                        help='Subgame type (competitive only for now)')
    parser.add_argument('--data', '--competitive_data', default='data/competitive_100k.npz',
                        help='Path to DDS data (npz)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=0.0,
                        help='P100: β for Agent B. Default 0.0 (partner-only r_info). '
                             'Agent B tests: does partner information shaping alone help?')
    parser.add_argument('--beta_c', type=float, default=0.05,
                        help='P100: β for Agent C (dual-information with opponent penalty). '
                             'Agent C tests: does the opponent leakage term add value?')
    parser.add_argument('--no_agent_c', action='store_true',
                        help='P100: Skip Agent C (only train A and B). '
                             'Useful for quick debugging or when β ablation is not needed.')
    parser.add_argument('--agent_a_only', action='store_true',
                        help='Drift sweep mode: only train Agent A, skip B and C entirely. '
                             'Saves ~50%% training time for convention drift experiments.')
    parser.add_argument('--info_weight', type=float, default=0.2,
                        help='r_info as fraction of IMP variance (P87b: 0.02→0.2, step_ir≈1.4)')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of rounds (P97d: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer deals for debugging')
    parser.add_argument('--belief_pretrain_rounds', type=int, default=50,
                        help='Rounds of data collection for Belief Net pretrain (P97b: 50, total=100k deals)')
    parser.add_argument('--belief_pretrain_max_epochs', type=int, default=50,
                        help='Max training epochs for Belief Net pretrain (P97b: 50, big data needs fewer epochs)')
    parser.add_argument('--sl_checkpoint', default='results/sl_base_bca.pt',
                        help='SL pretrained checkpoint (4-actor format). '
                             'P100 default: sl_base_bca.pt (397-dim BCA). '
                             'Use sl_base.pt for legacy 301-dim experiments.')
    parser.add_argument('--save_dir', default='results/competitive',
                        help='Directory to save checkpoints')
    parser.add_argument('--load_agent_a', default=None,
                        help='Path to pre-trained Agent A checkpoint (.pt). '
                        'If set, skip A training and load directly.')
    parser.add_argument('--ewc_lambda', type=float, default=100.0,
                        help='EWC penalty strength for Belief Net (P97, normalized Fisher, default: 100)')
    parser.add_argument('--no_belief_conditioned', action='store_true',
                        help='P100: Disable BCA (revert to 301-dim input). '
                        'By default, ALL agents use belief-conditioned actors (397-dim). '
                        'Use this flag for backward-compatible 301-dim experiments.')
    parser.add_argument('--belief_conditioned', action='store_true',
                        help='(Deprecated, now default) Kept for backward compatibility. '
                        'BCA is always on unless --no_belief_conditioned is set.')
    parser.add_argument('--kl_lambda', type=float, default=None,
                        help='KL anchor strength. Default: 0.3→0.0 anneal if --belief_conditioned, 0.3 fixed otherwise')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient for PPO (P98b: 0.01, was 1e-3 in P97d)')
    parser.add_argument('--eval_deals', type=int, default=1000,
                        help='Number of deals for Stage 3 evaluation (default 1000, use 5000 for paper)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip Stage 2 (RL training), load checkpoints from --save_dir, '
                             'run Stage 3 evaluation only. Requires prior training run.')

    # ── Eval-only mode (absorbed from eval_paired.py) ────────────────
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; run paired evaluation on saved checkpoints. '
                             'Requires --agent_a and --agent_b.')
    parser.add_argument('--agent_a', default=None,
                        help='[eval-only] Path to Agent A checkpoint (.pt)')
    parser.add_argument('--agent_b', default=None,
                        help='[eval-only] Path to Agent B checkpoint (.pt)')
    parser.add_argument('--num_deals', type=int, default=2000,
                        help='[eval-only] Number of deals for paired evaluation')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.eval_only:
        # Eval-only mode: requires --agent_a and --agent_b
        if not args.agent_a or not args.agent_b:
            import sys
            print("Error: --eval-only requires --agent_a and --agent_b", file=sys.stderr)
            sys.exit(1)
        run_eval_only(args)
    elif args.type == 'competitive':
        run_competitive(args)
    else:
        raise ValueError(f"Unknown subgame type: {args.type}")
