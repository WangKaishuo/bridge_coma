import torch
import torch.nn.functional as F
from env import string_to_bid, NORTH, SOUTH, bid_to_string
from subgames.stayman_env import StaymanSubgameEnv
from subgames.subgame_trainer import SubgameTrainer, SubgameConfig

def run_lstm_hearing_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading environment and model...")
    env = StaymanSubgameEnv("data/stayman_50k.npz", north_rule=False)
    cfg = SubgameConfig(device=device)
    trainer = SubgameTrainer(env, cfg)
    
    model = trainer.agent.model
    checkpoint = torch.load("results/s_base.pt", map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("\n=== LSTM Hearing Test for S_base ===")
    
    # 修复2：直接从数据集中捞一副保证符合要求的真牌！
    hands, _ = env.generate_deal()
    
    test_n_bids = {
        "N_bids_2D (No Fit)": "2D",
        "N_bids_2H (Heart Fit)": "2H",
        "N_bids_2S (Spade Fit)": "2S"
    }

    for case_name, n_bid in test_n_bids.items():
        # 重置环境
        obs = env.env.reset(hands, dealer=NORTH, vulnerability=(False, False))
        
        # 强制走完前面的序列
        for b in ["1NT", "Pass", "2C", "Pass"]:
            obs, _, _, _ = env.env.step(string_to_bid(b))
            
        # N 尝试不同的回答
        obs, _, _, _ = env.env.step(string_to_bid(n_bid))
        obs, _, _, _ = env.env.step(string_to_bid("Pass"))
        
        # 获取 S 的带合法动作掩码的观察
        obs['legal_actions'] = env._get_stayman_mask()
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in obs.items()}
        
        with torch.no_grad():
            try:
                logits = model.actor(obs_t)
            except TypeError:
                try:
                    logits = model.actor(obs_t['hand'], obs_t['history'], obs_t['history_lengths'])
                except TypeError:
                    hist_embed = model.actor.history_encoder(obs_t['history'], obs_t['history_lengths'])
                    logits = model.actor(obs_t['hand'], hist_embed)
            
            # 修复1：极其关键！在 Softmax 前必须用 Mask 屏蔽非法动作！
            legal_actions = obs_t['legal_actions']
            logits = logits.masked_fill(legal_actions == 0, -1e9)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        idx_3NT = string_to_bid("3NT")
        idx_4H = string_to_bid("4H")
        idx_4S = string_to_bid("4S")
        
        print(f"\nContext: {case_name}")
        print(f"  P(3NT): {probs[idx_3NT]:.4f}")
        print(f"  P(4H) : {probs[idx_4H]:.4f}")
        print(f"  P(4S) : {probs[idx_4S]:.4f}")
        
        # 打印 Top 2 看看它最想叫什么
        top_bids = probs.argsort()[-2:][::-1]
        print("  Top 2 choices:", ", ".join([f"{bid_to_string(b)} ({probs[b]:.1%})" for b in top_bids]))

if __name__ == "__main__":
    run_lstm_hearing_test()