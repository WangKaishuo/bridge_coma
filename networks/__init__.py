"""
Networks package (P52)
 
P52 变更: 删除 HandEncoder, HistoryEncoder, ActorCritic (LSTM架构已废弃).
保留: MLPPolicyNetwork (alias: PolicyNetwork), MLPValueNetwork (alias: ValueNetwork),
      BeliefNetwork, DualInfoComputer.
"""
 
from networks.policy_net import (
    MLPPolicyNetwork,
    MLPValueNetwork,
    PolicyNetwork,       # alias for MLPPolicyNetwork
    ValueNetwork,        # alias for MLPValueNetwork
    encode_obs_flat,
    encode_history_flat,
    BASE_INPUT_DIM,
    OBS_DIM,
)
 
from networks.belief_net import (
    BeliefNetwork,
    DualInfoComputer,
)