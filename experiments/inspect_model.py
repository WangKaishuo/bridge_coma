"""Inspect and validate one inference-only Bridge-COMA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


REQUIRED_KEYS = {
    "actor_n",
    "actor_e",
    "actor_s",
    "actor_w",
    "obs_dim",
    "hidden_dim",
    "actor_belief_conditioned",
    "actor_belief_hidden_dim",
    "action_mapping_version",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing = REQUIRED_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(f"Missing required checkpoint keys: {sorted(missing)}")
    print(f"file: {args.checkpoint}")
    print(f"bytes: {args.checkpoint.stat().st_size}")
    print(f"sha256: {sha256(args.checkpoint)}")
    print(f"observation dimensions: {checkpoint['obs_dim']}")
    print(f"hidden dimensions: {checkpoint['hidden_dim']}")
    print(f"belief-conditioned actor: {checkpoint['actor_belief_conditioned']}")
    print(f"action mapping: {checkpoint['action_mapping_version']}")
    print("roles: N, E, S, W")


if __name__ == "__main__":
    main()

