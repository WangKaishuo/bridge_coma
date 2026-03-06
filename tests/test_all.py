#!/usr/bin/env python3
"""
Bridge-COMA Test Suite
======================

Phase 1 验收测试，覆盖：
1. 包导入验证
2. 得分计算（含边界情况）
3. IMP 转换
4. 叫牌环境（结束条件、合法动作、庄家判定）
5. 双桌环境（swap、IMP、reward 分配、dealer 轮转、vulnerability）
6. 神经网络（PolicyNet、BeliefNet、ActorCritic、ValueNet）
7. Agent（IPPO、MAPPO：动作采样 + store + update）
8. 端到端：随机策略打多副牌

运行：
    cd bridge-coma/
    python tests/test_all.py              # 全部测试
    python tests/test_all.py --no-torch   # 跳过 torch 相关测试
"""

import argparse
import sys
import os
import tempfile
import traceback
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ============================================================================
# Helpers
# ============================================================================

_passed = 0
_failed = 0
_skipped = 0


def run_test(name, fn):
    """Run a test function, catch and report errors."""
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  ✓ {name}")
    except Exception as e:
        _failed += 1
        print(f"  ✗ {name}: {e}")
        traceback.print_exc(limit=3)
        print()


def skip_test(name, reason):
    global _skipped
    _skipped += 1
    print(f"  ~ {name} (skipped: {reason})")


def make_test_dds_data(path, n=200):
    """Generate a small fake DDS dataset for testing."""
    np.random.seed(42)
    decks = np.zeros((n, 52), dtype=np.uint8)
    for i in range(n):
        perm = np.random.permutation(52)
        for j, card in enumerate(perm):
            decks[i, card] = j // 13
    tricks = np.random.randint(3, 12, size=(n, 5, 4)).astype(np.int8)
    np.savez_compressed(path, decks=decks, tricks=tricks)
    return path


# ============================================================================
# 1. Package Imports
# ============================================================================

def test_imports():
    print("\n[1] Package imports")

    def _env_imports():
        from env import (BridgeBiddingEnv, DualTableEnv, DualTableResult,
                         make_random_policy, NUM_PLAYERS, NUM_BIDS, NUM_SUITS,
                         NUM_LEVELS, BID_PASS, BID_DOUBLE, BID_REDOUBLE,
                         BID_1C, NORTH, EAST, SOUTH, WEST)
        assert NUM_PLAYERS == 4
        assert NUM_BIDS == 38
        assert NORTH == 0 and EAST == 1 and SOUTH == 2 and WEST == 3

    def _utils_imports():
        from utils import (Contract, calculate_score, score_to_imp, imp_to_vp,
                           DDSDataLoader, MultiFileLoader, create_loader,
                           deck_to_hands, RunningStats, EMAStats)

    run_test("env package", _env_imports)
    run_test("utils package", _utils_imports)


# ============================================================================
# 2. Scoring
# ============================================================================

def test_scoring():
    print("\n[2] Scoring")
    from utils.scoring import Contract, calculate_score

    def _basic():
        # 4S making 10 tricks, NV -> 420
        c = Contract(level=4, suit=3, doubled=0, declarer=0)
        assert calculate_score(c, 10, False) == 420

        # 3NT down 2, vul -> -200
        c = Contract(level=3, suit=4, doubled=0, declarer=0)
        assert calculate_score(c, 7, True) == -200

        # 1C making 7, NV -> 70 (partial)
        c = Contract(level=1, suit=0, doubled=0, declarer=0)
        assert calculate_score(c, 7, False) == 70

    def _game_bonus():
        # 3NT just making, NV -> 400 (100 trick pts + 300 game)
        c = Contract(level=3, suit=4, doubled=0, declarer=0)
        assert calculate_score(c, 9, False) == 400

        # 3NT just making, Vul -> 600 (100 trick pts + 500 game)
        assert calculate_score(c, 9, True) == 600

        # 4H making, NV -> 420
        c = Contract(level=4, suit=2, doubled=0, declarer=0)
        assert calculate_score(c, 10, False) == 420

    def _slams():
        # 6NT making 12, NV -> 990 (190 tricks + 300 game + 500 small slam)
        c = Contract(level=6, suit=4, doubled=0, declarer=0)
        assert calculate_score(c, 12, False) == 990

        # 6NT making 12, Vul -> 1440 (190 + 500 + 750)
        assert calculate_score(c, 12, True) == 1440

        # 7S making 13, NV -> 1510 (210 + 300 + 1000)
        c = Contract(level=7, suit=3, doubled=0, declarer=0)
        assert calculate_score(c, 13, False) == 1510

        # 7S making 13, Vul -> 2210 (210 + 500 + 1500)
        assert calculate_score(c, 13, True) == 2210

    def _doubled():
        # 2S doubled making, NV -> 470 (2*60=120 tricks + 300 game + 50 insult)
        c = Contract(level=2, suit=3, doubled=1, declarer=0)
        assert calculate_score(c, 8, False) == 470

        # 3NT doubled down 1, NV -> -100
        c = Contract(level=3, suit=4, doubled=1, declarer=0)
        assert calculate_score(c, 8, False) == -100

        # 3NT doubled down 1, Vul -> -200
        assert calculate_score(c, 8, True) == -200

    def _redoubled():
        # 2H redoubled making, NV -> 640 (4*60=240 tricks + 300 game + 100 insult)
        c = Contract(level=2, suit=2, doubled=2, declarer=0)
        assert calculate_score(c, 8, False) == 640

        # 1C redoubled down 2, NV -> -600 (2 * (100+200))
        c = Contract(level=1, suit=0, doubled=2, declarer=0)
        assert calculate_score(c, 5, False) == -600

    def _down_penalties():
        # Down 3, NV undoubled -> -150
        c = Contract(level=4, suit=3, doubled=0, declarer=0)
        assert calculate_score(c, 7, False) == -150

        # Down 3, Vul undoubled -> -300
        assert calculate_score(c, 7, True) == -300

        # Down 3, NV doubled -> -(100+200+200) = -500
        c = Contract(level=4, suit=3, doubled=1, declarer=0)
        assert calculate_score(c, 7, False) == -500

        # Down 3, Vul doubled -> -(200+300+300) = -800
        assert calculate_score(c, 7, True) == -800

    run_test("basic scores", _basic)
    run_test("game bonus", _game_bonus)
    run_test("slams", _slams)
    run_test("doubled", _doubled)
    run_test("redoubled", _redoubled)
    run_test("down penalties", _down_penalties)


# ============================================================================
# 3. IMP Conversion
# ============================================================================

def test_imp():
    print("\n[3] IMP conversion")
    from utils.imp import score_to_imp, imp_to_vp

    def _imp_table():
        assert score_to_imp(0) == 0
        assert score_to_imp(10) == 0      # < 20 -> 0
        assert score_to_imp(20) == 1      # 20-49 -> 1
        assert score_to_imp(90) == 3      # 90-129 -> 3
        assert score_to_imp(-90) == -3    # negative
        assert score_to_imp(4000) == 24   # max
        assert score_to_imp(-4000) == -24

    def _vp():
        vp = imp_to_vp(0, boards=16)
        assert 9.5 <= vp <= 10.5, f"Draw should be ~10 VP, got {vp}"

    run_test("IMP table", _imp_table)
    run_test("VP conversion", _vp)


# ============================================================================
# 4. Bidding Environment
# ============================================================================

def test_env():
    print("\n[4] BridgeBiddingEnv")
    from env.bridge_bidding_env import (
        BridgeBiddingEnv, BID_PASS, BID_DOUBLE, BID_REDOUBLE, BID_1C,
        bid_to_string, string_to_bid, NUM_BIDS
    )
    from utils.scoring import Contract

    def _obs_shape():
        env = BridgeBiddingEnv()
        obs = env.reset()
        assert obs['hand'].shape == (52,)
        assert obs['history'].shape == (60, NUM_BIDS)
        assert obs['legal_actions'].shape == (NUM_BIDS,)
        assert obs['position'].shape == (4,)
        assert obs['vulnerability'].shape == (2,)
        # hand should be one-hot with 13 cards
        assert abs(obs['hand'].sum() - 13.0) < 0.01

    def _four_passes():
        env = BridgeBiddingEnv()
        env.reset()
        for _ in range(4):
            obs, _, done, _ = env.step(BID_PASS)
        assert done, "Should be done after 4 passes"
        assert env.state.final_contract is None, "Should be passed out"

    def _bid_then_three_passes():
        env = BridgeBiddingEnv()
        env.reset()
        env.step(BID_1C)  # 1C
        env.step(BID_PASS)
        env.step(BID_PASS)
        _, _, done, _ = env.step(BID_PASS)
        assert done, "Should be done after 1C-P-P-P"
        c = env.state.final_contract
        assert c is not None
        assert c.level == 1 and c.suit == 0

    def _not_done_early():
        env = BridgeBiddingEnv()
        env.reset()
        env.step(BID_1C)
        _, _, done, _ = env.step(BID_PASS)
        assert not done, "Should NOT be done after 1C-P"
        _, _, done, _ = env.step(BID_PASS)
        assert not done, "Should NOT be done after 1C-P-P"

    def _legal_actions_basic():
        env = BridgeBiddingEnv()
        env.reset()
        legal = env._get_legal_actions()
        # Opening: Pass always legal, all bids legal, no X/XX
        assert legal[BID_PASS] == 1.0
        assert legal[BID_DOUBLE] == 0.0
        assert legal[BID_REDOUBLE] == 0.0
        assert legal[BID_1C] == 1.0
        assert legal[37] == 1.0  # 7NT

    def _legal_after_bid():
        env = BridgeBiddingEnv()
        env.reset()
        env.step(BID_1C)  # N bids 1C
        legal = env._get_legal_actions()
        # E: can pass, can bid >=1D, can double, cannot redouble
        assert legal[BID_PASS] == 1.0
        assert legal[BID_DOUBLE] == 1.0
        assert legal[BID_REDOUBLE] == 0.0
        assert legal[BID_1C] == 0.0  # can't bid same level
        assert legal[BID_1C + 1] == 1.0  # 1D ok

    def _declarer_logic():
        """Declarer = first player in winning pair to bid that denomination"""
        env = BridgeBiddingEnv(max_history_len=60)
        env.reset(dealer=0)  # N deals
        # N: 1H, E: P, S: 4H, W: P, N: P, P
        env.step(BID_1C + 2)  # N bids 1H (index 5)
        env.step(BID_PASS)    # E pass
        env.step(BID_1C + 17) # S bids 4H (index 20)
        env.step(BID_PASS)    # W pass
        env.step(BID_PASS)    # N pass
        _, _, done, _ = env.step(BID_PASS)  # E pass
        assert done
        c = env.state.final_contract
        assert c is not None
        # 4H by N (North bid hearts first in NS pair)
        assert c.level == 4 and c.suit == 2
        assert c.declarer == 0, f"Declarer should be N(0), got {c.declarer}"

    def _bid_string_roundtrip():
        assert bid_to_string(BID_PASS) == "Pass"
        assert bid_to_string(BID_DOUBLE) == "X"
        assert bid_to_string(BID_REDOUBLE) == "XX"
        assert bid_to_string(BID_1C) == "1♣"
        assert bid_to_string(37) == "7NT"
        # roundtrip
        for i in [BID_PASS, BID_1C, BID_1C + 4, 37]:
            assert string_to_bid(bid_to_string(i)) == i

    def _vulnerability_in_obs():
        env = BridgeBiddingEnv()
        obs = env.reset(vulnerability=(True, False))
        assert obs['vulnerability'][0] == 1.0  # NS vul
        assert obs['vulnerability'][1] == 0.0  # EW not vul

    def _dealer_rotation():
        env = BridgeBiddingEnv()
        for dealer in range(4):
            obs = env.reset(dealer=dealer)
            assert obs['position'][dealer] == 1.0, f"Dealer {dealer} should be current player"

    run_test("obs shape & hand size", _obs_shape)
    run_test("four passes = passed out", _four_passes)
    run_test("bid + 3 passes = contract", _bid_then_three_passes)
    run_test("not done early", _not_done_early)
    run_test("legal actions opening", _legal_actions_basic)
    run_test("legal actions after bid", _legal_after_bid)
    run_test("declarer logic", _declarer_logic)
    run_test("bid string roundtrip", _bid_string_roundtrip)
    run_test("vulnerability in obs", _vulnerability_in_obs)
    run_test("dealer rotation", _dealer_rotation)


# ============================================================================
# 5. Dual Table Environment
# ============================================================================

def test_dual_table(dds_path):
    print("\n[5] DualTableEnv")
    from env.dual_table_env import (
        DualTableEnv, DualTableResult, make_random_policy, VULNERABILITY_COMBOS
    )

    dual_env = DualTableEnv(dds_path)

    def _swap():
        hands = np.eye(4, 52, dtype=np.float32)
        dd = np.arange(20).reshape(5, 4).astype(np.int8)
        sh, sd = DualTableEnv._swap(hands, dd)
        assert np.array_equal(sh[0], hands[1])  # N gets E's cards
        assert np.array_equal(sh[1], hands[0])  # E gets N's cards
        assert np.array_equal(sh[2], hands[3])  # S gets W's cards
        assert np.array_equal(sh[3], hands[2])  # W gets S's cards
        assert np.array_equal(sd[:, 0], dd[:, 1])
        assert np.array_equal(sd[:, 1], dd[:, 0])

    def _play_deal():
        result = dual_env.play_deal(make_random_policy())
        assert isinstance(result, DualTableResult)
        assert result.imp_ns == -result.imp_ew
        assert isinstance(result.imp_ns, (int, np.integer))

    def _collect_episodes_structure():
        def policy(obs):
            legal = obs['legal_actions']
            action = int(np.random.choice(np.where(legal > 0.5)[0]))
            return action, {'log_prob': 0.0, 'value': 0.0}

        episodes = dual_env.collect_episodes(policy, num_deals=2, rotate_dealer=True)
        assert len(episodes) == 8, f"2 deals × 4 dealers = 8, got {len(episodes)}"

        ep = episodes[0]
        assert 'player_trajectories' in ep
        assert 'imp_ns' in ep
        assert 'dealer' in ep
        assert 'vulnerability' in ep
        assert 'contract' in ep

    def _reward_assignment():
        def policy(obs):
            legal = obs['legal_actions']
            action = int(np.random.choice(np.where(legal > 0.5)[0]))
            return action, {'log_prob': 0.0, 'value': 0.0}

        episodes = dual_env.collect_episodes(policy, num_deals=5, rotate_dealer=True)
        for ep in episodes:
            imp = ep['imp_ns']
            for player in range(4):
                traj = ep['player_trajectories'][player]
                if traj:
                    expected = imp if player % 2 == 0 else -imp
                    assert traj[-1]['reward'] == expected, (
                        f"Player {player}: expected {expected}, got {traj[-1]['reward']}"
                    )
                    for step in traj[:-1]:
                        assert step['reward'] == 0.0, "Non-terminal reward should be 0"

    def _dealer_rotation():
        def policy(obs):
            legal = obs['legal_actions']
            action = int(np.random.choice(np.where(legal > 0.5)[0]))
            return action, {'log_prob': 0.0, 'value': 0.0}

        episodes = dual_env.collect_episodes(policy, num_deals=1, rotate_dealer=True)
        dealers = [e['dealer'] for e in episodes]
        assert dealers == [0, 1, 2, 3], f"Expected [0,1,2,3], got {dealers}"

    def _vulnerability_random():
        def policy(obs):
            legal = obs['legal_actions']
            action = int(np.random.choice(np.where(legal > 0.5)[0]))
            return action, {'log_prob': 0.0, 'value': 0.0}

        # 收集足够多样本，应能覆盖多种 vulnerability
        episodes = dual_env.collect_episodes(policy, num_deals=20, rotate_dealer=False)
        vuls = set(tuple(e['vulnerability']) for e in episodes)
        assert len(vuls) >= 2, f"Expected >=2 unique vulnerabilities, got {vuls}"

    def _imp_symmetry():
        """Random vs random, average IMP should be near 0"""
        imps = []
        for _ in range(100):
            r = dual_env.play_deal(make_random_policy())
            imps.append(r.imp_ns)
        mean_imp = np.mean(imps)
        assert abs(mean_imp) < 5, f"Random self-play mean IMP should be ~0, got {mean_imp:.2f}"

    run_test("swap logic", _swap)
    run_test("play_deal", _play_deal)
    run_test("collect_episodes structure", _collect_episodes_structure)
    run_test("reward assignment (NS/EW)", _reward_assignment)
    run_test("dealer rotation", _dealer_rotation)
    run_test("vulnerability randomization", _vulnerability_random)
    run_test("IMP symmetry (random vs random)", _imp_symmetry)


# ============================================================================
# 6. DDS Data Loading
# ============================================================================

def test_dds_data(dds_path):
    print("\n[6] DDS data loading")
    from utils.dds_data import create_loader, deck_to_hands, DDSDataLoader

    def _single_loader():
        loader = create_loader(dds_path)
        assert isinstance(loader, DDSDataLoader)
        assert len(loader) == 200  # we created 200 samples

    def _sample():
        loader = create_loader(dds_path)
        hands, tricks = loader.sample(batch_size=8)
        assert hands.shape == (8, 4, 52)
        assert tricks.shape == (8, 5, 4)
        assert hands.dtype == np.float32
        # Each player should have 13 cards
        for b in range(8):
            for p in range(4):
                assert abs(hands[b, p].sum() - 13.0) < 0.01

    def _sample_one():
        loader = create_loader(dds_path)
        hands, tricks = loader.sample_one()
        assert hands.shape == (4, 52)
        assert tricks.shape == (5, 4)

    def _deck_to_hands():
        deck = np.array([i // 13 for i in range(52)], dtype=np.uint8)
        hands = deck_to_hands(deck)
        assert hands.shape == (4, 52)
        assert hands[0, :13].sum() == 13  # player 0 has first 13 cards
        assert hands[0, 13:].sum() == 0

    run_test("create_loader single file", _single_loader)
    run_test("sample batch", _sample)
    run_test("sample one", _sample_one)
    run_test("deck_to_hands", _deck_to_hands)


# ============================================================================
# 7. Running Stats
# ============================================================================

def test_running_stats():
    print("\n[7] Running stats")
    from utils.running_stats import RunningStats, EMAStats

    def _welford():
        rs = RunningStats()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        for x in data:
            rs.update(x)
        assert abs(rs.mean - 3.0) < 0.01
        assert abs(rs.variance - 2.5) < 0.01  # sample variance
        assert abs(rs.normalize(3.0)) < 0.01  # mean -> 0

    def _ema():
        ema = EMAStats(alpha=0.1)
        for _ in range(100):
            ema.update(5.0)
        assert abs(ema.mean - 5.0) < 0.1

    run_test("Welford RunningStats", _welford)
    run_test("EMAStats", _ema)


# ============================================================================
# 8. Networks (torch required)
# ============================================================================

def test_networks():
    print("\n[8] Networks")
    import torch
    from networks import PolicyNetwork, ValueNetwork, ActorCritic, BeliefNetwork
    from env import NUM_BIDS

    batch = 4
    obs = {
        'hand': torch.rand(batch, 52),
        'history': torch.rand(batch, 10, NUM_BIDS),
        'legal_actions': torch.ones(batch, NUM_BIDS),
        'position': torch.eye(4)[:batch],
        'vulnerability': torch.zeros(batch, 2),
    }

    def _policy_net():
        net = PolicyNetwork()
        logits = net(obs)
        assert logits.shape == (batch, NUM_BIDS)
        action, log_prob, entropy = net.get_action(obs)
        assert action.shape == (batch,)
        assert log_prob.shape == (batch,)

    def _value_net_local():
        net = ValueNetwork(centralized=False)
        val = net(obs)
        assert val.shape == (batch,)

    def _value_net_centralized():
        net = ValueNetwork(centralized=True)
        all_hands = torch.rand(batch, 4, 52)
        val = net(obs, all_hands)
        assert val.shape == (batch,)

    def _actor_critic():
        ac = ActorCritic(centralized_critic=False)
        action, log_prob, entropy, value = ac.get_action_and_value(obs)
        assert action.shape == (batch,)
        assert value.shape == (batch,)

    def _actor_critic_centralized():
        ac = ActorCritic(centralized_critic=True)
        all_hands = torch.rand(batch, 4, 52)
        action, log_prob, entropy, value = ac.get_action_and_value(obs, all_hands)
        assert action.shape == (batch,)
        assert value.shape == (batch,)

    def _belief_net():
        net = BeliefNetwork()
        belief = net(obs['hand'], obs['history'],
                     torch.tensor([0, 1, 2, 3]), torch.tensor([2, 3, 0, 1]))
        assert belief.shape == (batch, 52)
        assert (belief >= 0).all() and (belief <= 1).all(), "Belief should be in [0,1]"

    def _belief_loss():
        net = BeliefNetwork()
        target = torch.rand(batch, 52).round()  # binary target
        loss = net.compute_loss(obs['hand'], obs['history'],
                                torch.tensor([0, 1, 2, 3]),
                                torch.tensor([2, 3, 0, 1]),
                                target)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    run_test("PolicyNetwork", _policy_net)
    run_test("ValueNetwork (local)", _value_net_local)
    run_test("ValueNetwork (centralized)", _value_net_centralized)
    run_test("ActorCritic (local)", _actor_critic)
    run_test("ActorCritic (centralized)", _actor_critic_centralized)
    run_test("BeliefNetwork forward", _belief_net)
    run_test("BeliefNetwork loss", _belief_loss)


# ============================================================================
# 9. Agents (torch required)
# ============================================================================

def test_agents():
    print("\n[9] Agents")
    import torch
    from algorithms import IPPOAgent, PPOConfig, MAPPOAgent, MAPPOConfig

    def _make_obs():
        obs = {k: np.random.rand(*s).astype(np.float32) for k, s in [
            ('hand', (52,)), ('history', (60, 38)),
            ('legal_actions', (38,)), ('position', (4,)), ('vulnerability', (2,))
        ]}
        obs['legal_actions'][:] = 1
        obs['position'][:] = 0
        obs['position'][0] = 1
        return obs

    def _ippo_action():
        agent = IPPOAgent(PPOConfig(device='cpu'))
        obs = _make_obs()
        action, extra = agent.get_action(obs)
        assert 0 <= action < 38
        assert 'log_prob' in extra
        assert 'value' in extra

    def _ippo_store_and_update():
        agent = IPPOAgent(PPOConfig(device='cpu'))
        obs = _make_obs()
        for _ in range(10):
            action, extra = agent.get_action(obs)
            agent.store_transition(0, obs, action, extra['log_prob'], 1.0, extra['value'], False)
        action, extra = agent.get_action(obs)
        agent.store_transition(0, obs, action, extra['log_prob'], 5.0, extra['value'], True)
        stats = agent.update()
        assert 'loss' in stats
        assert 'policy_loss' in stats

    def _mappo_action():
        agent = MAPPOAgent(MAPPOConfig(device='cpu'))
        obs = _make_obs()
        all_hands = np.random.rand(4, 52).astype(np.float32)
        action, extra = agent.get_action(obs, all_hands=all_hands)
        assert 0 <= action < 38

    def _mappo_store_and_update():
        agent = MAPPOAgent(MAPPOConfig(device='cpu'))
        obs = _make_obs()
        all_hands = np.random.rand(4, 52).astype(np.float32)
        for _ in range(10):
            action, extra = agent.get_action(obs, all_hands=all_hands)
            agent.store_transition(0, obs, action, extra['log_prob'], 1.0,
                                   extra['value'], False, all_hands=all_hands)
        action, extra = agent.get_action(obs, all_hands=all_hands)
        agent.store_transition(0, obs, action, extra['log_prob'], 5.0,
                               extra['value'], True, all_hands=all_hands)
        stats = agent.update()
        assert 'loss' in stats

    run_test("IPPO get_action", _ippo_action)
    run_test("IPPO store + update", _ippo_store_and_update)
    run_test("MAPPO get_action", _mappo_action)
    run_test("MAPPO store + update", _mappo_store_and_update)


# ============================================================================
# 10. End-to-end
# ============================================================================

def test_e2e(dds_path):
    print("\n[10] End-to-end")
    from env.dual_table_env import DualTableEnv, make_random_policy

    def _random_10_deals():
        dual_env = DualTableEnv(dds_path)
        for i in range(10):
            result = dual_env.play_deal(
                make_random_policy(),
                dealer=i % 4,
                vulnerability=[(False, False), (True, False),
                               (False, True), (True, True)][i % 4],
            )
            # Should not crash; IMP should be an integer
            assert isinstance(result.imp_ns, (int, np.integer))

    def _random_collect_20_episodes():
        dual_env = DualTableEnv(dds_path)

        def policy(obs):
            legal = obs['legal_actions']
            action = int(np.random.choice(np.where(legal > 0.5)[0]))
            return action, {'log_prob': 0.0, 'value': 0.0}

        episodes = dual_env.collect_episodes(policy, num_deals=5, rotate_dealer=True)
        assert len(episodes) == 20
        # Every episode should have valid structure
        for ep in episodes:
            assert ep['dealer'] in [0, 1, 2, 3]
            assert ep['vulnerability'] in [
                (False, False), (True, False), (False, True), (True, True)
            ]
            total_bids = sum(len(t) for t in ep['player_trajectories'].values())
            assert total_bids >= 4, f"At least 4 bids needed, got {total_bids}"

    run_test("random policy 10 deals no crash", _random_10_deals)
    run_test("collect 20 episodes no crash", _random_collect_20_episodes)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-torch', action='store_true', help='Skip torch-dependent tests')
    args = parser.parse_args()

    has_torch = False
    if not args.no_torch:
        try:
            import torch
            has_torch = True
        except ImportError:
            pass

    # Create temp DDS data
    tmpdir = tempfile.mkdtemp()
    dds_path = make_test_dds_data(os.path.join(tmpdir, 'dds_test.npz'))

    print("=" * 60)
    print("Bridge-COMA Test Suite")
    print(f"torch available: {has_torch}")
    print("=" * 60)

    # Non-torch tests
    test_imports()
    test_scoring()
    test_imp()
    test_env()
    test_dual_table(dds_path)
    test_dds_data(dds_path)
    test_running_stats()

    # Torch tests
    if has_torch:
        test_networks()
        test_agents()
    else:
        print("\n[8] Networks")
        skip_test("all network tests", "torch not available")
        print("\n[9] Agents")
        skip_test("all agent tests", "torch not available")

    # E2E
    test_e2e(dds_path)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    print()
    print("=" * 60)
    total = _passed + _failed + _skipped
    print(f"Results: {_passed} passed, {_failed} failed, {_skipped} skipped / {total} total")
    if _failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"FAILURES: {_failed}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
