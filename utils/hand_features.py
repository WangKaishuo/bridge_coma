"""
Hand Feature Extraction — 48-dim Belief Target
===============================================

将原始 52 维 one-hot 手牌转换为 48 维特征向量，
作为 BeliefNetwork 的预测目标。

特征结构
--------
[0  : 16]  荣誉牌归属 — 16 维
           每门花色 (♣♦♥♠) × AKQJ
           bit = 1 iff 该牌在目标玩家手中
           各维度相互独立

[16 : 48]  套长 one-hot — 32 维
           每门花色 × 8 档 (长度 0, 1, 2, 3, 4, 5, 6, 7+)
           每门花色恰好有一个 bit = 1，其余为 0
           8张及以上合并入"7+"档（极罕见，约0.3%）

设计理由
--------
- AKQJ 是叫牌大牌点的全部来源，T以下对叫牌决策几乎无贡献
- one-hot 套长避免了阶梯式编码的单调性冗余问题：
  "5张黑桃"只激活1个bit，r_info不会重复计算
- 每门花色内8档互斥，梯度信号干净，无跨维度干扰
- 全部 48 维均为 0/1，BCEWithLogitsLoss 与 r_info 公式无需修改

pos_weight
----------
统一设为 3.0。
荣誉牌: P(持有) = 1/4 → 理论值恰好 = 3.0
套长:   one-hot 互斥特性使类别不平衡影响较小，3.0 作为保守统一值
"""

import numpy as np
import torch

# ── 常量 ───────────────────────────────────────────────────────────────────────
BELIEF_DIM    = 48
HONOR_DIM     = 16          # [0:16]  AKQJ × 4门
LENGTH_DIM    = 32          # [16:48] one-hot套长 × 4门
NUM_SUITS     = 4
HONOR_RANKS   = [12, 11, 10, 9]    # A K Q J (rank index, 2=0, A=12)
LENGTH_BINS   = 8                   # 0,1,2,3,4,5,6,7+
POS_WEIGHT    = 3.0


def build_pos_weight() -> torch.Tensor:
    """构造 (48,) 统一 pos_weight tensor，传入 BCEWithLogitsLoss."""
    return torch.full((BELIEF_DIM,), POS_WEIGHT)


# ── 核心转换 ───────────────────────────────────────────────────────────────────

def hand_to_belief_target(hand: np.ndarray) -> np.ndarray:
    """
    单手牌转换: (52,) one-hot → (48,) binary features.

    Args:
        hand: (52,) float32, one-hot
              排列: suit * 13 + rank, rank ∈ [0,12] (2=0, A=12)

    Returns:
        features: (48,) float32, 值为 0.0 或 1.0
    """
    features = np.zeros(BELIEF_DIM, dtype=np.float32)

    # [0:16] AKQJ 归属
    idx = 0
    for suit in range(NUM_SUITS):
        for rank in HONOR_RANKS:
            features[idx] = hand[suit * 13 + rank]
            idx += 1

    # [16:48] 套长 one-hot（8档，7+合并）
    for suit in range(NUM_SUITS):
        suit_len = int(hand[suit * 13 : (suit + 1) * 13].sum())
        bin_idx  = min(suit_len, 7)          # 8张及以上 → 第7档
        features[HONOR_DIM + suit * LENGTH_BINS + bin_idx] = 1.0

    return features


def batch_hand_to_belief_target(hands: np.ndarray) -> np.ndarray:
    """
    批量转换: (B, 52) → (B, 48)，向量化实现。
    """
    B = hands.shape[0]
    features = np.zeros((B, BELIEF_DIM), dtype=np.float32)

    # [0:16] AKQJ 归属（向量化）
    idx = 0
    for suit in range(NUM_SUITS):
        for rank in HONOR_RANKS:
            features[:, idx] = hands[:, suit * 13 + rank]
            idx += 1

    # [16:48] 套长 one-hot（向量化）
    for suit in range(NUM_SUITS):
        suit_lens = hands[:, suit * 13 : (suit + 1) * 13].sum(axis=1).astype(int)
        bin_idxs  = np.minimum(suit_lens, 7)
        rows      = np.arange(B)
        features[rows, HONOR_DIM + suit * LENGTH_BINS + bin_idxs] = 1.0

    return features


# ── 评估指标 ───────────────────────────────────────────────────────────────────

def belief_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    评估 Belief Network 预测质量，替代旧版 top13_hit_rate。

    Args:
        probs:   (B, 48) sigmoid 概率
        targets: (B, 48) 0/1 ground truth

    Returns dict:
        honor_acc  : 荣誉牌 (前16维) 准确率，threshold=0.5
                     随机基线 ≈ 0.625（P=0.25时预测0，neg acc主导）
        length_acc : 套长 one-hot (后32维) 准确率
                     等价于: 每门花色预测的最高概率档位是否正确
                     随机基线 = 1/8 = 0.125（8档均匀）
        overall_acc: 全部48维准确率
    """
    # 荣誉牌：threshold=0.5
    honor_preds   = (probs[:, :HONOR_DIM] > 0.5).float()
    honor_acc     = float((honor_preds == targets[:, :HONOR_DIM]).float().mean())

    # 套长：每门花色取argmax，等价于one-hot分类准确率
    length_probs  = probs[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS)   # (B,4,8)
    length_target = targets[:, HONOR_DIM:].view(-1, NUM_SUITS, LENGTH_BINS) # (B,4,8)
    pred_bins     = length_probs.argmax(dim=-1)    # (B, 4)
    true_bins     = length_target.argmax(dim=-1)   # (B, 4)
    length_acc    = float((pred_bins == true_bins).float().mean())

    # 全维度 threshold acc（参考用）
    overall_preds = (probs > 0.5).float()
    overall_acc   = float((overall_preds == targets).float().mean())

    return {
        'honor_acc':  honor_acc,
        'length_acc': length_acc,
        'overall_acc': overall_acc,
    }


# ── 调试工具 ───────────────────────────────────────────────────────────────────

def decode_belief_target(features: np.ndarray) -> dict:
    """48 维特征 → 人类可读格式（调试用）."""
    suit_names  = ['♣', '♦', '♥', '♠']
    honor_names = ['A', 'K', 'Q', 'J']

    honors = {}
    idx = 0
    for suit in range(NUM_SUITS):
        honors[suit_names[suit]] = [
            honor_names[i] for i in range(4)
            if features[idx + i] > 0.5
        ]
        idx += 4

    lengths = {}
    for suit in range(NUM_SUITS):
        slot = features[HONOR_DIM + suit * LENGTH_BINS :
                        HONOR_DIM + (suit + 1) * LENGTH_BINS]
        bin_idx = int(slot.argmax())
        lengths[suit_names[suit]] = f"{bin_idx}+" if bin_idx == 7 else str(bin_idx)

    return {'honors': honors, 'lengths': lengths}


# ── 自测 ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    # 单手测试
    hand = np.zeros(52, dtype=np.float32)
    hand[np.random.choice(52, 13, replace=False)] = 1.0

    feat = hand_to_belief_target(hand)
    assert feat.shape == (BELIEF_DIM,), f"shape error: {feat.shape}"
    assert set(feat.tolist()) <= {0.0, 1.0}, "non-binary value found"

    # one-hot 完整性：每门花色恰好1个bit为1
    for suit in range(NUM_SUITS):
        slot = feat[HONOR_DIM + suit * LENGTH_BINS :
                    HONOR_DIM + (suit + 1) * LENGTH_BINS]
        assert slot.sum() == 1.0, f"suit {suit} one-hot invalid: {slot}"

    decoded = decode_belief_target(feat)
    print("Decoded:", decoded)

    # 验证套长与真实手牌一致
    for suit in range(NUM_SUITS):
        true_len  = int(hand[suit * 13 : (suit + 1) * 13].sum())
        slot      = feat[HONOR_DIM + suit * LENGTH_BINS :
                         HONOR_DIM + (suit + 1) * LENGTH_BINS]
        pred_bin  = int(slot.argmax())
        expected  = min(true_len, 7)
        assert pred_bin == expected, \
            f"suit {suit}: true_len={true_len}, expected_bin={expected}, got={pred_bin}"

    # 批量测试
    hands = np.zeros((32, 52), dtype=np.float32)
    for i in range(32):
        hands[i, np.random.choice(52, 13, replace=False)] = 1.0
    batch_feat = batch_hand_to_belief_target(hands)
    assert batch_feat.shape == (32, BELIEF_DIM)
    # 每门花色one-hot完整性
    lf = batch_feat[:, HONOR_DIM:].reshape(32, NUM_SUITS, LENGTH_BINS)
    assert (lf.sum(axis=-1) == 1).all(), "batch one-hot invalid"

    pw = build_pos_weight()
    assert pw.shape == (BELIEF_DIM,)
    assert (pw == POS_WEIGHT).all()

    print(f"pos_weight = {POS_WEIGHT} (uniform)")
    print(f"All checks passed. BELIEF_DIM={BELIEF_DIM} ✓")
