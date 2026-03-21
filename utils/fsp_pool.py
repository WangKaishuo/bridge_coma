"""
FSP Pool — Fictitious Self-Play Checkpoint Pool
===============================================

Kita et al. 2024: 从历史 checkpoint pool 中随机采样对手，
防止 policy cycling（两个 agent 互相 adapt 形成的循环）。

Pool size = 10。每隔若干轮将当前 policy snapshot 存入 pool，
rollout 时从 pool 中随机选一个作为对手。

设计决策:
    - Pool 以 state_dict 形式存储（不保存 optimizer）
    - 采样时随机选一个，不做加权（Kita 消融证明等权最优）
    - 支持 save_to_disk / load_from_disk，便于跨会话恢复
    - 线程安全性: 不并发写，不需要锁

接口:
    pool = FSPPool(max_size=10)
    pool.add(agent)             → 存入当前快照
    state = pool.sample()       → 随机返回一个 state_dict
    pool.apply(agent, state)    → 将 state_dict 加载到 agent（仅覆盖 actor）
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
    pool = [permanent_0, ..., permanent_k, fifo_0, ..., fifo_m]
    max_size 包含 permanent members。

    Args:
        max_size : pool 容量，建议 10（Kita et al. 2024）
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._permanent: list[dict] = []  # 永不淘汰的 members (e.g. SL)
        self._pool: list[dict] = []       # FIFO 部分

    @property
    def _fifo_capacity(self) -> int:
        return self.max_size - len(self._permanent)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add(self, agent) -> None:
        """
        将 agent 的当前 actor 权重快照存入 pool.

        只存 actor（不存 critic / optimizer），节省内存。
        超过 max_size 时弹出最早的快照（FIFO）。

        Args:
            agent: MAPPOAgent，需有 .model.actor_n / .model.actor_s 属性
        """
        snapshot = {
            'actor_n': copy.deepcopy(agent.model.actor_n.state_dict()),
            'actor_s': copy.deepcopy(agent.model.actor_s.state_dict()),
            # EW 阵营也需要（competitive 子博弈中 EW 是对手）
            'actor_e': copy.deepcopy(
                getattr(agent.model, 'actor_e', agent.model.actor_n).state_dict()),
            'actor_w': copy.deepcopy(
                getattr(agent.model, 'actor_w', agent.model.actor_s).state_dict()),
        }
        # 将所有 tensor 移到 CPU，避免 GPU 显存占用
        snapshot = {
            role: {k: v.cpu() for k, v in sd.items()}
            for role, sd in snapshot.items()
        }

        if len(self._pool) >= self._fifo_capacity:
            self._pool.pop(0)   # 移除 FIFO 中最旧的（permanent 不受影响）
        self._pool.append(snapshot)

    def add_state_dict(self, state_dict: dict) -> None:
        """
        直接存入预先准备好的 state_dict（BC 预训练 checkpoint 等）.

        state_dict 格式: {'actor_n': ..., 'actor_s': ..., 'actor_e': ..., 'actor_w': ...}
        """
        sd_cpu = {
            role: {k: v.cpu() for k, v in sd.items()}
            for role, sd in state_dict.items()
        }
        if len(self._pool) >= self._fifo_capacity:
            self._pool.pop(0)
        self._pool.append(sd_cpu)

    def add_permanent(self, agent) -> None:
        """
        P90: 将 agent 存为 permanent member（SL baseline 等），永不被 FIFO 淘汰。
        """
        snapshot = {
            'actor_n': copy.deepcopy(agent.model.actor_n.state_dict()),
            'actor_s': copy.deepcopy(agent.model.actor_s.state_dict()),
            'actor_e': copy.deepcopy(
                getattr(agent.model, 'actor_e', agent.model.actor_n).state_dict()),
            'actor_w': copy.deepcopy(
                getattr(agent.model, 'actor_w', agent.model.actor_s).state_dict()),
        }
        snapshot = {
            role: {k: v.cpu() for k, v in sd.items()}
            for role, sd in snapshot.items()
        }
        self._permanent.append(snapshot)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def sample(self) -> Optional[dict]:
        """
        随机返回一个 state_dict（从 permanent + FIFO 合并池中均匀采样）。
        """
        all_members = self._permanent + self._pool
        if not all_members:
            return None
        return random.choice(all_members)

    def latest(self) -> Optional[dict]:
        """返回最新存入的 FIFO state_dict."""
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
        """
        将 pool 中的 state_dict 加载到 agent 的指定 actor 网络.

        只加载 actor（Critic 不参与 FSP，保持当前训练中的 Critic）。

        Args:
            agent      : MAPPOAgent
            state_dict : pool.sample() 返回的快照
            roles      : 要加载的角色列表，默认加载 NS（训练方的对手 = EW）
        """
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
        """将整个 pool 存到磁盘（用于断点续训）."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'pool': self._pool, 'max_size': self.max_size}, path)
        print(f"[FSPPool] Saved {len(self._pool)} snapshots → {path}")

    def load_from_disk(self, path: str) -> None:
        """从磁盘恢复 pool."""
        if not os.path.exists(path):
            print(f"[FSPPool] No checkpoint at {path}, starting fresh.")
            return
        data = torch.load(path, map_location='cpu')
        self._pool    = data.get('pool', [])
        self.max_size = data.get('max_size', self.max_size)
        print(f"[FSPPool] Loaded {len(self._pool)} snapshots ← {path}")
