"""
eval_paired.py — 配对评估：同一组牌上的 A vs B vs SL 三方对战
================================================================

从 competitive_500k.npz 中采样 N 副牌，三组对战 (A vs SL, B vs SL, A vs B)
全部使用同一组牌，支持配对检验。

使用 env.play_mixed() 保证 score 计算与训练时的 evaluate_head_to_head 完全一致。

用法:
    python eval_paired.py \
        --data data/competitive_500k.npz \
        --sl_checkpoint results/sl_base.pt \
        --agent_a results/competitive/agent_a_seed42.pt \
        --agent_b results/competitive/agent_b_seed42.pt \
        --num_deals 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from scipy.stats import wilcoxon

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from subgames.competitive_env import CompetitiveSubgameEnv
from networks.policy_net import encode_obs_flat
from utils.imp import score_to_imp
from env import NORTH, EAST, SOUTH, WEST


# ======================================================================
# Loading
# ======================================================================

def load_agent(path: str, device: str) -> MAPPOAgent:
    agent = MAPPOAgent(MAPPOConfig(device=device))
    agent.load(path)
    return agent


def load_sl_as_agent(sl_path: str, device: str) -> MAPPOAgent:
    agent = MAPPOAgent(MAPPOConfig(device=device))
    ckpt = torch.load(sl_path, map_location=device)
    for player, key in [(0, 'actor_n'), (1, 'actor_e'),
                        (2, 'actor_s'), (3, 'actor_w')]:
        if key in ckpt:
            agent.get_actor(player).load_state_dict(
                {k: v.to(device) for k, v in ckpt[key].items()})
    return agent


def make_policy(agent: MAPPOAgent, env: CompetitiveSubgameEnv, device: str):
    """
    policy(obs, player, history_int) -> action_int

    与 play_mixed 的 3 参数签名匹配。
    用 env.dealer 获取当前局 dealer（play_mixed 内部 reset 时会设置）。
    """
    def policy(obs, player, history_int):
        flat = encode_obs_flat(obs, env.dealer, history_int)
        flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
        legal = torch.tensor(
            obs['legal_actions'], dtype=torch.float32).unsqueeze(0).to(device)
        actor = agent.get_actor(player)
        with torch.no_grad():
            action, _, _ = actor.get_action(flat_t, legal, deterministic=True)
        return action.item()
    return policy


# ======================================================================
# Cross-eval on fixed deals
# ======================================================================

def cross_eval_fixed_deals(env, deals, pol_a, pol_b):
    """
    双桌复式赛，A 视角 IMP。直接调用 env.play_mixed()。
    桌1: A=opener(ns_policy), B=overcaller(ew_policy)
    桌2: B=opener(ns_policy), A=overcaller(ew_policy)
    IMP = score_to_imp(score_1 - score_2)  正=A赢

    关键：play_mixed 不会设置 env.dealer，但 policy 函数依赖 env.dealer
    做位置编码。必须在每局 play_mixed 之前手动设置 env.dealer。
    """
    imps = []
    for hands, dd_table, dealer, vul in deals:
        env.dealer = dealer  # policy 函数通过 env.dealer 获取 dealer
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


# ======================================================================
# Printing
# ======================================================================

def print_result(label_a, label_b, imps):
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


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/competitive_500k.npz')
    parser.add_argument('--sl_checkpoint', default='results/sl_base.pt')
    parser.add_argument('--agent_a', default='results/competitive/agent_a_seed42.pt')
    parser.add_argument('--agent_b', default='results/competitive/agent_b_seed42.pt')
    parser.add_argument('--num_deals', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Paired Eval] device={device}  n={args.num_deals}  seed={args.seed}")

    # ── Load ──────────────────────────────────────────────────────────
    env = CompetitiveSubgameEnv(data_path=args.data)

    print("\n[Loading]")
    sl_agent = load_sl_as_agent(args.sl_checkpoint, device)
    print(f"  SL:      {args.sl_checkpoint}")
    agent_a = load_agent(args.agent_a, device)
    print(f"  Agent A: {args.agent_a}")
    agent_b = load_agent(args.agent_b, device)
    print(f"  Agent B: {args.agent_b}")

    pol_sl = make_policy(sl_agent, env, device)
    pol_a = make_policy(agent_a, env, device)
    pol_b = make_policy(agent_b, env, device)

    # ── Sample deals from constrained pool ────────────────────────────
    print(f"\n[Sampling {args.num_deals} deals from {args.data}]")
    deals = []
    for _ in range(args.num_deals):
        hands, dd_table = env.generate_deal()
        dealer = env._sampled_dealer
        vul = [(False, False), (True, False),
               (False, True),  (True, True)][np.random.randint(4)]
        deals.append((hands.copy(), dd_table.copy(), dealer, vul))
    print(f"  {len(deals)} deals sampled.")

    # ── 3 matchups on SAME deals ──────────────────────────────────────
    print(f"\n[H2H on identical {args.num_deals} deals]")
    print("=" * 72)

    imps_a_sl = cross_eval_fixed_deals(env, deals, pol_a, pol_sl)
    m1, _, p1 = print_result("Agent_A", "SL", imps_a_sl)

    imps_b_sl = cross_eval_fixed_deals(env, deals, pol_b, pol_sl)
    m2, _, p2 = print_result("Agent_B", "SL", imps_b_sl)

    imps_a_b = cross_eval_fixed_deals(env, deals, pol_a, pol_b)
    m3, _, p3 = print_result("Agent_A", "Agent_B", imps_a_b)

    print("=" * 72)

    # ── Paired: is A stronger than B against SL? ──────────────────────
    print(f"\n[Paired test: A_vs_SL minus B_vs_SL, per-deal]")
    diff = imps_a_sl - imps_b_sl
    mean_d = diff.mean()
    std_d = diff.std()
    se_d = std_d / np.sqrt(len(diff))
    try:
        _, p_paired = wilcoxon(diff)
    except Exception:
        p_paired = 1.0

    print(f"  mean diff = {mean_d:+.3f}  std = {std_d:.3f}  SE = {se_d:.3f}")
    print(f"  Wilcoxon p = {p_paired:.4f}  {'✅ sig' if p_paired < 0.05 else '(ns)'}")
    if mean_d > 0 and p_paired < 0.05:
        print(f"  → A significantly stronger than B vs SL")
    elif mean_d < 0 and p_paired < 0.05:
        print(f"  → B significantly stronger than A vs SL")
    else:
        print(f"  → No significant difference in vs-SL strength")

    # ── Correlation ───────────────────────────────────────────────────
    corr = np.corrcoef(imps_a_sl, imps_b_sl)[0, 1]
    print(f"\n  Corr(A_vs_SL, B_vs_SL) = {corr:.3f}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  A vs SL:     {imps_a_sl.mean():+.3f} ± {imps_a_sl.std():.3f}  p={p1:.4f}")
    print(f"  B vs SL:     {imps_b_sl.mean():+.3f} ± {imps_b_sl.std():.3f}  p={p2:.4f}")
    print(f"  A vs B:      {imps_a_b.mean():+.3f} ± {imps_a_b.std():.3f}  p={p3:.4f}")
    print(f"  Paired diff: {mean_d:+.3f}  p={p_paired:.4f}")
    print(f"  Correlation: {corr:.3f}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
