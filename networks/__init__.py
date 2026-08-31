"""Policy, value, and belief networks used by Bridge-COMA."""
 
from networks.policy_net import (
    MLPPolicyNetwork,
    MLPValueNetwork,
    PolicyNetwork,       # alias for MLPPolicyNetwork
    ValueNetwork,        # alias for MLPValueNetwork
    ActorCritic,
    BASE_INPUT_DIM,
    OBS_DIM,
)
 
from networks.belief_net import (
    BeliefNetwork,
    DualInfoComputer,
)
