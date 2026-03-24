"""
Subgame Validation — Competitive Path
======================================

Training mode (default):
    python experiments/subgame_validation.py \\
        --type competitive \\
        --data path/to/1h1s_100k.npz \\
        --seed 42 --beta 0.05 --rounds 10 --quick

Eval-only mode (paired evaluation on saved checkpoints):
    python experiments/subgame_validation.py \\
        --eval-only \\
        --data data/competitive_500k.npz \\
        --sl_checkpoint results/sl_base.pt \\
        --agent_a results/competitive/agent_a_seed42.pt \\
        --agent_b results/competitive/agent_b_seed42.pt \\
        --num_deals 2000 --seed 42

Eval-only runs three paired matchups on identical deals (A vs SL, B vs SL, A vs B),
reports per-deal Wilcoxon p-values, and prints a paired A_vs_SL − B_vs_SL diff test.

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
from networks.policy_net import OBS_DIM, BELIEF_OBS_DIM, encode_obs_flat
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
    print(f"\n[Competitive Subgame] seed={args.seed}  device={device}")
    print(f"  beta={args.beta}  info_weight={args.info_weight}  rounds={args.rounds}  quick={args.quick}")
    print(f"  belief_conditioned={_bc}  kl_lambda={_kl}")

    # ── 环境 ────────────────────────────────────────────────────────────────
    print("\n[1] Initializing environment...")
    env = CompetitiveSubgameEnv(data_path=args.data)

    # ── 公共参数 ───────────────────────────────────────────────────────
    _bc = getattr(args, 'belief_conditioned', False)
    # P98: with belief_conditioned, KL=0 (Full Disclosure via convention card, not KL proxy)
    if args.kl_lambda is not None:
        _kl = args.kl_lambda
    else:
        _kl = 0.0 if _bc else 0.3

    # ── Agent A（控制组：纯 MAPPO，beta=0）──────────────────────────────────
    print("\n[2] Building Agent A (MAPPO, β=0)...")
    cfg_a = SubgameConfig(
        num_rounds       = args.rounds,
        steps_per_phase  = 10  if args.quick else 64,
        deals_per_step   = 32  if args.quick else 512,
        lr               = 3e-6,
        batch_size       = 256,
        use_info_bonus   = False,
        beta             = 0.0,
        fsp_pool_size    = 10,
        fsp_add_interval = 1,
        kl_lambda_start  = _kl,
        kl_lambda_end    = _kl,
        kl_anneal_frac   = 0.0,
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5    if args.quick else 20,
        device           = device,
        # P98: belief_conditioned — Actor uses belief features as input
        belief_conditioned = _bc,
        # A also needs a belief net when belief_conditioned (for obs encoding)
        # but does NOT use r_info
        freeze_belief    = False if _bc else True,
        belief_update_epochs = 1,
        belief_update_lr  = 1e-5,
    )
    reward_stats_a = RunningStats()
    trainer_a = SubgameTrainer(env, cfg_a, reward_stats=reward_stats_a)

    # ── Agent B（实验组：MAPPO + r_info，beta=args.beta）────────────────────
    print(f"\n[3] Building Agent B (MAPPO + r_info, β={args.beta}, w={args.info_weight})...")
    cfg_b = SubgameConfig(
        # 与A完全对称，唯一区别是use_info_bonus、beta、info_reward_weight
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
        kl_lambda_start  = _kl,
        kl_lambda_end    = _kl,
        kl_anneal_frac   = 0.0,
        bc_warmup_samples= 1000 if args.quick else 5000,
        bc_warmup_epochs = 5    if args.quick else 20,
        device           = device,
        # P98: belief_conditioned — Actor uses belief features as input
        belief_conditioned = _bc,
        freeze_belief    = False,
        use_ewc          = False,
        belief_update_epochs = 1,
        belief_update_lr  = 1e-5,
    )
    reward_stats_b = RunningStats()
    trainer_b = SubgameTrainer(env, cfg_b, reward_stats=reward_stats_b)

    # ── Stage 1: 初始化（SL checkpoint 优先，否则 rule-based BC）──────────────
    from env import NORTH as _N, EAST as _E, SOUTH as _S, WEST as _W
    sl_path = getattr(args, 'sl_checkpoint', None)

    # 确定哪些trainer需要SL初始化
    trainers_to_init = [trainer_b] if args.load_agent_a else [trainer_a, trainer_b]
    is_bca_ckpt = False  # default, updated below if BCA checkpoint detected

    if sl_path and os.path.exists(sl_path):
        print(f"\n[Stage 1] Loading SL checkpoint: {sl_path}")
        ckpt = torch.load(sl_path, map_location=device)
        sl_obs_dim = ckpt.get('obs_dim', OBS_DIM)
        player_key_map = [(_N, 'actor_n'), (_E, 'actor_e'),
                          (_S, 'actor_s'), (_W, 'actor_w')]

        # P98: BCA checkpoint (sl_base_bca.pt) has 349-dim actors + belief_net
        is_bca_ckpt = (sl_obs_dim == BELIEF_OBS_DIM)
        if _bc and is_bca_ckpt:
            print(f"  [P98] BCA checkpoint detected (obs_dim={sl_obs_dim})")

        for trainer in trainers_to_init:
            for player, key in player_key_map:
                if key not in ckpt:
                    continue
                sl_sd = {k: v.to(device) for k, v in ckpt[key].items()}
                actor = trainer.agent.get_actor(player)

                if not _bc:
                    # Standard 301→301 load
                    if is_bca_ckpt:
                        # BCA checkpoint into non-BCA agent: skip (dimension mismatch)
                        print(f"  ⚠️  Skipping {key}: BCA checkpoint incompatible with non-BCA agent")
                        continue
                    actor.load_state_dict(sl_sd)
                elif is_bca_ckpt:
                    # P98: BCA checkpoint → BCA agent (349→349, direct load)
                    actor.load_state_dict(sl_sd)
                else:
                    # P98: Standard checkpoint → BCA agent (301→349, zero-init extra)
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

            # P98: Load belief_net from BCA checkpoint if available
            if _bc and 'belief_net' in ckpt and trainer.belief_net is not None:
                trainer.belief_net.load_state_dict(
                    {k: v.to(device) for k, v in ckpt['belief_net'].items()})
                print(f"  [P98] Belief Net loaded from SL checkpoint for "
                      f"{'A' if trainer is trainer_a else 'B'}")

        init_names = "B only" if args.load_agent_a else "both agents"
        dim_note = f" [BCA {sl_obs_dim}-dim]" if is_bca_ckpt else (
            " [301→349 adapted]" if _bc else "")
        print(f"  [SL Init] Weights loaded for N/E/S/W actors ({init_names}).{dim_note}")
    else:
        print("\n[Stage 1] BC Warmup (rule-based, SL checkpoint not found)...")
        for trainer in trainers_to_init:
            trainer.run_bc_warmup()

    # Stage 1 结束：将当前 actor 设为 KL anchor
    for trainer in trainers_to_init:
        trainer.set_bc_anchor(trainer.agent)
    anchor_names = "Agent B" if args.load_agent_a else "both agents"
    print(f"  [KL Anchor] BC anchor set for {anchor_names}.")

    # ── Stage 1.5: Belief Net 独立预训练 ──────────────────────────────
    belief_pretrain_rounds = getattr(args, 'belief_pretrain_rounds', 5)
    deals = 200 if args.quick else 2000

    # P98: when belief_conditioned, BOTH agents need belief net pretrained
    # But skip if already loaded from BCA checkpoint
    trainers_for_belief = []
    if belief_pretrain_rounds > 0:
        _bn_from_ckpt = _bc and is_bca_ckpt and 'belief_net' in (ckpt if sl_path and os.path.exists(sl_path) else {})
        if trainer_b.belief_net is not None and not _bn_from_ckpt:
            trainers_for_belief.append(('B', trainer_b))
        if _bc and trainer_a.belief_net is not None and not args.load_agent_a and not _bn_from_ckpt:
            trainers_for_belief.append(('A', trainer_a))

    for name, trainer in trainers_for_belief:
        print(f"\n[Stage 1.5] Belief Net Pretrain (Agent {name}, "
              f"total {belief_pretrain_rounds * deals} deals, early stopping)...")
        trainer.pretrain_belief(
            num_rounds=belief_pretrain_rounds,
            deals_per_round=deals,
            epochs_per_round=5,
            max_epochs=getattr(args, 'belief_pretrain_max_epochs', 300),
        )

    # ── Build SL baseline trainer for mini eval during training ──────────────
    from algorithms.mappo import MAPPOAgent, MAPPOConfig
    from networks.policy_net import BELIEF_OBS_DIM as _BOD

    # P98: If BCA checkpoint, SL baseline also uses 349-dim with its own belief net
    _sl_bc = _bc and is_bca_ckpt if (sl_path and os.path.exists(sl_path)) else False
    _sl_obs_dim = _BOD if _sl_bc else OBS_DIM

    cfg_sl = SubgameConfig(
        device=device, belief_conditioned=_sl_bc, use_info_bonus=False,
        kl_lambda_start=0.0, kl_lambda_end=0.0,
    )
    sl_agent_eval = MAPPOAgent(MAPPOConfig(device=device, obs_dim=_sl_obs_dim))
    if sl_path and os.path.exists(sl_path):
        ckpt_sl = torch.load(sl_path, map_location=device)
        for player, key in [(_N, 'actor_n'), (_E, 'actor_e'),
                            (_S, 'actor_s'), (_W, 'actor_w')]:
            if key in ckpt_sl:
                sl_agent_eval.get_actor(player).load_state_dict(
                    {k: v.to(device) for k, v in ckpt_sl[key].items()})
    sl_trainer = SubgameTrainer(env, cfg_sl, reward_stats=RunningStats())
    sl_trainer.agent = sl_agent_eval

    # P98: Load SL's belief net for SL eval trainer
    if _sl_bc and 'belief_net' in ckpt_sl and sl_trainer.belief_net is not None:
        sl_trainer.belief_net.load_state_dict(
            {k: v.to(device) for k, v in ckpt_sl['belief_net'].items()})
        print(f"  [SL Eval] Belief-conditioned SL baseline ({_sl_obs_dim}-dim)")

    # ── Stage 2: RL 微调 ────────────────────────────────────────────────────
    print("\n[Stage 2] RL Fine-tuning...")

    if args.load_agent_a:
        print(f"  ── Agent A (loading from {args.load_agent_a}) ──")
        trainer_a.agent.load(args.load_agent_a)
        log_a = []
        print(f"  [Agent A] Loaded from checkpoint. Training skipped.")
    else:
        print("  ── Agent A ──")
        log_a = trainer_a.run(num_rounds=args.rounds, sl_trainer=sl_trainer)

    print("\n  ── Agent B ──")
    log_b = trainer_b.run(num_rounds=args.rounds, sl_trainer=sl_trainer)

    # ── Stage 3: 评估 ───────────────────────────────────────────────────────
    print("\n[Stage 3] Evaluation...")
    h2h_deals = 200 if args.quick else 1000

    # sl_trainer already built before Stage 2

    # ── A vs SL ────────────────────────────────────────────────────────────
    print("\n  → Agent A vs SL baseline")
    h2h_a_sl = trainer_a.evaluate_head_to_head(
        sl_trainer, num_deals=h2h_deals, label_self="A", label_other="SL")

    # ── B vs SL ────────────────────────────────────────────────────────────
    print("\n  → Agent B vs SL baseline")
    h2h_b_sl = trainer_b.evaluate_head_to_head(
        sl_trainer, num_deals=h2h_deals, label_self="B", label_other="SL")

    # ── A vs B ─────────────────────────────────────────────────────────────
    print("\n  → Agent A vs Agent B")
    h2h = trainer_a.evaluate_head_to_head(
        trainer_b, num_deals=h2h_deals, label_self="A", label_other="B")

    # Belief Net 评估（Agent B only）
    if cfg_b.use_info_bonus:
        print("\n  → Belief Network Evaluation (Agent B):")
        trainer_b.evaluate_belief(num_deals=50 if args.quick else 200)

    # ── P97c: Partner Info Gain diagnostic (A vs B, same belief net) ──
    if cfg_b.use_info_bonus and trainer_b.belief_net is not None:
        diag_deals = 100 if args.quick else 500
        print(f"\n  → Partner Info Gain Diagnostic ({diag_deals} deals, B's belief net as judge)")
        print("\n  Agent A (no r_info):")
        pig_a = trainer_a.evaluate_partner_info_gain(
            trainer_b.belief_net, num_deals=diag_deals)
        print("\n  Agent B (with r_info):")
        pig_b = trainer_b.evaluate_partner_info_gain(
            trainer_b.belief_net, num_deals=diag_deals)
        diff = pig_b['mean_partner_gain'] - pig_a['mean_partner_gain']
        print(f"\n  [Δ partner_gain] B - A = {diff:+.4f}"
              f"  ({'B communicates more' if diff > 0 else 'A communicates more'})")

    # ── 排雷诊断 ────────────────────────────────────────────────────────────
    print("\n[Diagnostics]")
    _print_diagnostics(log_a, log_b, cfg_b)

    # ── 最终摘要 ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY")
    print("═" * 60)
    print(f"  A vs SL:  {h2h_a_sl['mean_imp']:+.3f} ± {h2h_a_sl['std_imp']:.3f} IMP  "
          f"p={h2h_a_sl['p_value']:.3f} {'✅' if h2h_a_sl['significant'] else '(ns)'}")
    print(f"  B vs SL:  {h2h_b_sl['mean_imp']:+.3f} ± {h2h_b_sl['std_imp']:.3f} IMP  "
          f"p={h2h_b_sl['p_value']:.3f} {'✅' if h2h_b_sl['significant'] else '(ns)'}")
    print(f"  A vs B:   {h2h['mean_imp']:+.3f} ± {h2h['std_imp']:.3f} IMP  "
          f"p={h2h['p_value']:.3f} {'✅' if h2h['significant'] else '(ns)'}")
    # Conclusion
    if h2h['mean_imp'] < 0 and h2h['significant']:
        conclusion = "✅ B significantly better than A"
    elif h2h['mean_imp'] > 0 and h2h['significant']:
        conclusion = "⚠️  A outperforms B"
    else:
        conclusion = "❌ No significant difference"
    print(f"  Conclusion: {conclusion}")
    print("═" * 60)

    # ── 保存 checkpoint ─────────────────────────────────────────────────────
    if args.save_dir:
        import json
        os.makedirs(args.save_dir, exist_ok=True)
        trainer_a.agent.save(os.path.join(args.save_dir, f'agent_a_seed{args.seed}.pt'))
        trainer_b.agent.save(os.path.join(args.save_dir, f'agent_b_seed{args.seed}.pt'))
        report = {
            'seed': args.seed, 'beta': args.beta, 'rounds': args.rounds,
            'a_vs_sl': h2h_a_sl,
            'b_vs_sl': h2h_b_sl,
            'a_vs_b': h2h,
        }
        report_path = os.path.join(args.save_dir, f'report_seed{args.seed}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Checkpoints + report saved → {args.save_dir}")


def _print_diagnostics(log_a: list, log_b: list, cfg_b: SubgameConfig):
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

    ent_b   = _last_metric(log_b, SOUTH, 'entropy')
    vl_b    = _last_metric(log_b, SOUTH, 'value_loss')
    kl_b    = _last_metric(log_b, SOUTH, 'kl_loss')
    klam_b  = _last_metric(log_b, SOUTH, 'kl_lambda')
    ent_b_e = _last_metric(log_b, EAST, 'entropy')
    vl_b_e  = _last_metric(log_b, EAST, 'value_loss')
    print(f"  Agent B: NS entropy={ent_b:.3f} vl={vl_b:.4f} kl={kl_b:.5f}(λ={klam_b:.3f}) │ EW entropy={ent_b_e:.3f} vl={vl_b_e:.4f}")

    ok = True
    if ent_b < 0.5:
        print("  ⚠️  Entropy collapse detected! Consider higher entropy_coef.")
        ok = False
    if vl_b > 10:
        print("  ⚠️  Value loss too high! Consider more critic warmup.")
        ok = False
    if kl_b > 0.3:
        print("  ⚠️  KL too high! Consider higher kl_lambda_start.")
        ok = False

    if cfg_b.use_info_bonus:
        # ir 健康性: 从日志中找 info_ratio
        ir_vals = []
        for entry in log_b:
            if 'info_ratio' in entry:
                ir_vals.append(entry['info_ratio'])
        if ir_vals:
            mean_ir = np.mean(ir_vals)
            print(f"  Agent B: mean ir={mean_ir:.4f}")
            if mean_ir <= 0:
                print("  ⚠️  ir ≤ 0! Check Belief Net and r_info wiring.")
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
    """Load 4-actor SL checkpoint (sl_base.pt or sl_base_bca.pt) as a MAPPOAgent."""
    agent = MAPPOAgent(MAPPOConfig(device=device))
    ckpt = torch.load(sl_path, map_location=device)
    for player, key in [(0, 'actor_n'), (1, 'actor_e'),
                        (2, 'actor_s'), (3, 'actor_w')]:
        if key in ckpt:
            agent.get_actor(player).load_state_dict(
                {k: v.to(device) for k, v in ckpt[key].items()})
    return agent


def _make_eval_policy(agent: MAPPOAgent, env: CompetitiveSubgameEnv, device: str):
    """
    Wrap a MAPPOAgent as a 3-arg policy(obs, player, history_int) → action_int.
    Uses env.dealer for position encoding (set before each play_mixed call).
    """
    def policy(obs, player, history_int):
        flat = encode_obs_flat(obs, env.dealer, history_int)
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

    env.dealer must be set before each play_mixed call so the policy
    closures can encode position correctly.
    """
    imps = []
    for hands, dd_table, dealer, vul in deals:
        env.dealer = dealer
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
    parser.add_argument('--beta', type=float, default=0.05,
                        help='Internal β: r_info = I(partner) - β·I(opponent)')
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
    parser.add_argument('--sl_checkpoint', default='results/sl_base.pt',
                        help='SL pretrained checkpoint (4-actor format from sl_pretrain.py). '
                             'Standard: sl_base.pt (301-dim); BCA: sl_base_bca.pt (349-dim).')
    parser.add_argument('--save_dir', default='results/competitive',
                        help='Directory to save checkpoints')
    parser.add_argument('--load_agent_a', default=None,
                        help='Path to pre-trained Agent A checkpoint (.pt). '
                        'If set, skip A training and load directly.')
    parser.add_argument('--ewc_lambda', type=float, default=100.0,
                        help='EWC penalty strength for Belief Net (P97, normalized Fisher, default: 100)')
    parser.add_argument('--belief_conditioned', action='store_true',
                        help='P98: Belief-Conditioned Actor (349-dim input). '
                        'Actor receives belief features as input, closing the '
                        'sender→receiver→decision loop. Enables relaxed KL.')
    parser.add_argument('--kl_lambda', type=float, default=None,
                        help='KL anchor strength. Default: 0.0 if --belief_conditioned, 0.3 otherwise')

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
