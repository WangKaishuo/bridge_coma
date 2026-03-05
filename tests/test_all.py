#!/usr/bin/env python3
"""Test Script"""
import numpy as np
import torch

def test_scoring():
    print("Testing scoring...")
    from utils.scoring import Contract, calculate_score
    
    # 4S making 10 tricks, not vul
    c = Contract(level=4, suit=3, doubled=0, declarer=0)
    score = calculate_score(c, 10, False)
    assert score == 420, f"Expected 420, got {score}"
    
    # 3NT down 2, vul
    c = Contract(level=3, suit=4, doubled=0, declarer=0)
    score = calculate_score(c, 7, True)
    assert score == -200, f"Expected -200, got {score}"
    
    print("  ✓ Scoring OK")

def test_env():
    print("Testing BridgeBiddingEnv...")
    from env import BridgeBiddingEnv, BID_PASS
    
    env = BridgeBiddingEnv()
    obs = env.reset()
    
    # Test 4 passes = passed out
    for _ in range(4):
        obs, _, done, _ = env.step(BID_PASS)
    assert done, "Should be done after 4 passes"
    assert env.state.final_contract is None, "Should be passed out"
    
    # Test real bid + 3 passes
    env.reset()
    env.step(3)  # 1C
    env.step(BID_PASS)
    env.step(BID_PASS)
    obs, _, done, _ = env.step(BID_PASS)
    assert done, "Should be done after 1C-P-P-P"
    assert env.state.final_contract is not None
    
    print("  ✓ BridgeBiddingEnv OK")

def test_networks():
    print("Testing networks...")
    from networks import PolicyNetwork, BeliefNetwork
    from env import NUM_BIDS
    
    obs = {
        'hand': torch.rand(2, 52),
        'history': torch.rand(2, 10, NUM_BIDS),
        'legal_actions': torch.ones(2, NUM_BIDS),
        'position': torch.eye(4)[:2],
        'vulnerability': torch.zeros(2, 2),
    }
    
    policy = PolicyNetwork()
    assert policy(obs).shape == (2, NUM_BIDS)
    
    belief = BeliefNetwork()
    pred = belief(obs['hand'], obs['history'], torch.tensor([0,1]), torch.tensor([1,2]))
    assert pred.shape == (2, 52)
    
    print("  ✓ Networks OK")

def test_agents():
    print("Testing agents...")
    from algorithms import IPPOAgent, PPOConfig, MAPPOAgent, MAPPOConfig
    
    obs = {k: np.random.rand(*s).astype(np.float32) 
           for k, s in [('hand', (52,)), ('history', (60, 38)), 
                        ('legal_actions', (38,)), ('position', (4,)), ('vulnerability', (2,))]}
    obs['legal_actions'][:] = 1
    obs['position'][:] = 0; obs['position'][0] = 1
    
    ippo = IPPOAgent(PPOConfig(device='cpu'))
    assert 0 <= ippo.get_action(obs)[0] < 38
    
    mappo = MAPPOAgent(MAPPOConfig(device='cpu'))
    assert 0 <= mappo.get_action(obs, np.random.rand(4, 52).astype(np.float32))[0] < 38
    
    print("  ✓ Agents OK")

if __name__ == "__main__":
    print("=" * 50)
    test_scoring()
    test_env()
    test_networks()
    test_agents()
    print("=" * 50)
    print("All tests passed! ✓")
