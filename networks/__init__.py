"""
Networks package (P52)
 
P52 变更: 删除 HandEncoder, HistoryEncoder, ActorCritic (LSTM架构已废弃).
保留: PolicyNetwork, ValueNetwork, BeliefNetwork.
"""
 
from networks.policy_net import (
    PolicyNetwork,
    ValueNetwork,
    encode_history_flat,
    BASE_INPUT_DIM,
)
 
from networks.belief_net import (
    BeliefNetwork,
    DualInfoComputer,
)