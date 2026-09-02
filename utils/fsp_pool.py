"""
FSP Pool - Fictitious Self-Play Checkpoint Pool
===============================================

BeliefNet is stored alongside actor weights in every snapshot because each
agent uses the opponent's BeliefNet to interpret opponent bids.

Pool entry format:
    {
        'actor_n': state_dict,
        'actor_s': state_dict,
        'actor_e': state_dict,
        'actor_w': state_dict,
        'belief_net': state_dict or None,
    }
"""

import copy
import os
import random
from pathlib import Path
from typing import Optional

import torch


class FSPPool:
    """
    Fixed-size first-in-first-out checkpoint pool.

    Permanent members, such as the supervised baseline, are never evicted.
    Each snapshot may carry a BeliefNet state dictionary.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._permanent: list[dict] = []
        self._pool: list[dict] = []

    @property
    def _fifo_capacity(self) -> int:
        return self.max_size - len(self._permanent)

    # ------------------------------------------------------------------
    # Write operations.
    # ------------------------------------------------------------------

    def add(self, agent, belief_net=None) -> None:
        """
        Store the agent's current actor weights and an optional BeliefNet snapshot.

        Args:
            agent:      MAPPOAgent
            belief_net: optional co-evolved BeliefNetwork instance
        """
        snapshot = {
            'actor_n': copy.deepcopy(agent.model.actor_n.state_dict()),
            'actor_s': copy.deepcopy(agent.model.actor_s.state_dict()),
            'actor_e': copy.deepcopy(
                getattr(agent.model, 'actor_e', agent.model.actor_n).state_dict()),
            'actor_w': copy.deepcopy(
                getattr(agent.model, 'actor_w', agent.model.actor_s).state_dict()),
            # BeliefNet snapshot
            'belief_net': copy.deepcopy(belief_net.state_dict()) if belief_net is not None else None,
        }
        snapshot = {
            role: ({k: v.cpu() for k, v in sd.items()} if isinstance(sd, dict) else sd)
            for role, sd in snapshot.items()
        }

        if self._fifo_capacity <= 0:
            return
        if len(self._pool) >= self._fifo_capacity:
            self._pool.pop(0)
        self._pool.append(snapshot)

    def add_state_dict(self, state_dict: dict, belief_net=None) -> None:
        """
        Store a prepared state dictionary, such as a pretrained checkpoint.

        Args:
            state_dict: {'actor_n': ..., 'actor_s': ..., 'actor_e': ..., 'actor_w': ...}
            belief_net: optional BeliefNetwork instance
        """
        sd_cpu = {
            role: ({k: v.cpu() for k, v in sd.items()} if isinstance(sd, dict) else sd)
            for role, sd in state_dict.items()
        }
        # Attach BeliefNet.
        sd_cpu['belief_net'] = (
            copy.deepcopy(belief_net.state_dict()) if belief_net is not None else None
        )
        if self._fifo_capacity <= 0:
            return
        if len(self._pool) >= self._fifo_capacity:
            self._pool.pop(0)
        self._pool.append(sd_cpu)

    def add_permanent(self, agent, belief_net=None) -> None:
        """
        Store an agent as a permanent member that FIFO eviction cannot remove.
        An optional BeliefNet snapshot is stored with the actor weights.
        """
        snapshot = {
            'actor_n': copy.deepcopy(agent.model.actor_n.state_dict()),
            'actor_s': copy.deepcopy(agent.model.actor_s.state_dict()),
            'actor_e': copy.deepcopy(
                getattr(agent.model, 'actor_e', agent.model.actor_n).state_dict()),
            'actor_w': copy.deepcopy(
                getattr(agent.model, 'actor_w', agent.model.actor_s).state_dict()),
            'belief_net': copy.deepcopy(belief_net.state_dict()) if belief_net is not None else None,
        }
        snapshot = {
            role: ({k: v.cpu() for k, v in sd.items()} if isinstance(sd, dict) else sd)
            for role, sd in snapshot.items()
        }
        self._permanent.append(snapshot)

    # ------------------------------------------------------------------
    # Read operations.
    # ------------------------------------------------------------------

    def sample(self) -> Optional[dict]:
        all_members = self._permanent + self._pool
        if not all_members:
            return None
        return random.choice(all_members)

    def latest(self) -> Optional[dict]:
        if not self._pool:
            return None
        return self._pool[-1]

    def __len__(self) -> int:
        return len(self._permanent) + len(self._pool)

    def is_empty(self) -> bool:
        return len(self._permanent) + len(self._pool) == 0

    # ------------------------------------------------------------------
    # Apply a snapshot to an agent.
    # ------------------------------------------------------------------

    def apply_to_agent(self, agent, state_dict: dict,
                       roles: tuple = ('actor_n', 'actor_s')) -> None:
        """Return actor weights; retrieve BeliefNet separately."""
        device = agent.device
        role_to_net = {
            'actor_n': agent.model.actor_n,
            'actor_s': agent.model.actor_s,
            'actor_e': getattr(agent.model, 'actor_e', agent.model.actor_n),
            'actor_w': getattr(agent.model, 'actor_w', agent.model.actor_s),
        }
        for role in roles:
            if role in state_dict and role in role_to_net:
                net = role_to_net[role]
                sd  = {k: v.to(device) for k, v in state_dict[role].items()}
                net.load_state_dict(sd)

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------

    def save_to_disk(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'pool': self._pool, 'max_size': self.max_size}, path)
        print(f"[FSPPool] Saved {len(self._pool)} snapshots -> {path}")

    def load_from_disk(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[FSPPool] No checkpoint at {path}, starting fresh.")
            return
        data = torch.load(path, map_location='cpu')
        self._pool    = data.get('pool', [])
        self.max_size = data.get('max_size', self.max_size)
        print(f"[FSPPool] Loaded {len(self._pool)} snapshots <- {path}")
