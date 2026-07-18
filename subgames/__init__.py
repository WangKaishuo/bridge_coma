"""Competitive bridge-bidding subgame components.

Training modules are intentionally not imported here so importing an environment
does not initialize PyTorch.
"""

from subgames.competitive_env import (
    CompetitiveSubgameEnv,
)
from subgames.unrestricted_env import UnrestrictedBiddingEnv
