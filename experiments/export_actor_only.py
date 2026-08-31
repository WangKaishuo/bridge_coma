"""Export a compact four-actor deployment checkpoint from a training artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ACTOR_KEYS = ("actor_n", "actor_e", "actor_s", "actor_w")
METADATA_KEYS = (
    "obs_dim",
    "hidden_dim",
    "actor_belief_conditioned",
    "actor_belief_hidden_dim",
    "action_mapping_version",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    checkpoint = source.get("agent", source)
    missing = [key for key in ACTOR_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"source checkpoint is missing actors: {missing}")
    compact = {
        key: {
            name: tensor.detach().cpu()
            for name, tensor in checkpoint[key].items()
        }
        for key in ACTOR_KEYS
    }
    for key in METADATA_KEYS:
        if key in checkpoint:
            compact[key] = checkpoint[key]

    required_metadata = set(METADATA_KEYS) - set(compact)
    if required_metadata:
        raise ValueError(
            f"source checkpoint is missing deployment metadata: "
            f"{sorted(required_metadata)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(compact, temporary)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
