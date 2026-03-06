"""
Subgame Environments for Phase 2 Validation

Note: SubgameTrainer and torch-dependent modules are imported lazily.
Use direct imports when needed:
    from subgames.subgame_trainer import SubgameTrainer, SubgameConfig
"""

from subgames.action_mask import (
    count_hcp,
    count_suit_length,
    suit_lengths,
    is_balanced,
    get_legal_mask,
    get_soft_mask,
    get_combined_mask,
)

# Env classes only import numpy, not torch
from subgames.stayman_env import StaymanSubgameEnv
from subgames.competitive_env import (
    CompetitiveSubgameEnv,
    CrossEvalResult,
)
