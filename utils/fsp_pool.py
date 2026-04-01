"""
FSP Pool — Fictitious Self-Play Checkpoint Pool
===============================================

P125: BeliefNet now stored alongside actor weights in every snapshot.
Full Disclosure: each agent uses opponent's BeliefNet to interpret opponent bids.

Pool entry format:
    {
        'actor_n': state_dict,
        'actor_s': state_dict,
        'actor_e': state_dict,
        'actor_w': state_dict,
        'belief_net': state_dict or None,   # ← P125 new
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
    固定大小的 checkpoint pool，先进先出（FIFO）.

    P90: 支持 permanent member（如 SL baseline），不被 FIFO 淘汰。
    P125: 每个 snapshot 携带 BeliefNet state_dict（Full Disclosure）。
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._permanent: list[dict] = []
        self._pool: list[dict] = []

    @property
    def _fifo_capacity(self) -> int:
        return self.max_size - len(self._permanent)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add(self, agent, belief_net=None) -> None:
        """
        将 agent 的当前 actor 权重 + BeliefNet 快照存入 pool.

        Args:
            agent:      MAPPOAgent
            belief_net: BeliefNetwork 实例（co-evolved，P125）
        """
        snapshot = {
            'actor_n': copy.deepcopy(agent.model.actor_n.state_dict()),
            'actor_s': copy.deepcopy(agent.model.actor_s.state_dict()),
            'actor_e': copy.deepcopy(
                getattr(agent.model, 'actor_e', agent.model.actor_n).state_dict()),
            'actor_w': copy.deepcopy(
                getattr(agent.model, 'actor_w', agent.model.actor_s).state_dict()),
            # P125: BeliefNet snapshot
            'belief_net': copy.deepcopy(belief_net.state_dict()) if belief_net is not None else None,
        }
        snapshot = {
            role: ({k: v.cpu() for k, v in sd.items()} if isinstance(sd, dict) else sd)
            for role, sd in snapshot.items()
        }

        if len(self._pool) >= self._fifo_capacity:
            self._pool.pop(0)
        self._pool.append(snapshot)

    def add_state_dict(self, state_dict: dict, belief_net=None) -> None:
        """
        直接存入预先准备好的 state_dict（BC 预训练 checkpoint 等）.

        Args:
            state_dict: {'actor_n': ..., 'actor_s': ..., 'actor_e': ..., 'actor_w': ...}
            belief_net: BeliefNetwork 实例（P125）
        """
        sd_cpu = {
            role: ({k: v.cpu() for k, v in sd.items()} if isinstance(sd, dict) else sd)
            for role, sd in state_dict.items()
        }
        # P125: attach BeliefNet
        sd_cpu['belief_net'] = (
            copy.deepcopy(belief_net.state_dict()) if belief_net is not None else None
        )
        if len(self._pool) >= self._fifo_capacity:
            self._pool.pop(0)
        self._pool.append(sd_cpu)

    def add_permanent(self, agent, belief_net=None) -> None:
        """
        P90: 将 agent 存为 permanent member（SL baseline 等），永不被 FIFO 淘汰。
        P125: 同时存入 BeliefNet（SL permanent 传入 sl_base_bca.pt 的 BeliefNet）。
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
    # 读取
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
    # 应用到 agent
    # ------------------------------------------------------------------

    def apply_to_agent(self, agent, state_dict: dict,
                       roles: tuple = ('actor_n', 'actor_s')) -> None:
        """actor weights only，BeliefNet 单独通过 get_fsp_belief_net() 取。"""
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
    # 持久化
    # ------------------------------------------------------------------

    def save_to_disk(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'pool': self._pool, 'max_size': self.max_size}, path)
        print(f"[FSPPool] Saved {len(self._pool)} snapshots → {path}")

    def load_from_disk(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[FSPPool] No checkpoint at {path}, starting fresh.")
            return
        data = torch.load(path, map_location='cpu')
        self._pool    = data.get('pool', [])
        self.max_size = data.get('max_size', self.max_size)
        print(f"[FSPPool] Loaded {len(self._pool)} snapshots ← {path}")
