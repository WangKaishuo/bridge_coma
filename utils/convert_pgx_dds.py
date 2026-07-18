#!/usr/bin/env python3
"""Convert PGX packed DDS lookup tables to bridge-coma ``dds_*.npz`` files.

The official PGX data stores four packed deal keys and four packed DDS values
per board.  bridge-coma uses an explicit owner for each of 52 cards and a
``(strain, declarer)`` trick table, so conversion is lossless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEY_SHIFTS = np.arange(24, -1, -2, dtype=np.uint32)
VALUE_SHIFTS = np.array([16, 12, 8, 4, 0], dtype=np.uint32)


def _destination_cards() -> np.ndarray:
    """Map PGX S,H,D,C / A,2,...,K order to project C,D,H,S / 2,...,A."""
    result = []
    for pgx_suit in range(4):
        project_suit = 3 - pgx_suit
        for pgx_rank in range(13):
            project_rank = 12 if pgx_rank == 0 else pgx_rank - 1
            result.append(project_suit * 13 + project_rank)
    return np.asarray(result, dtype=np.int64)


DESTINATION_CARDS = _destination_cards()


def decode_chunk(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode one vectorized chunk into the native compact representation."""
    keys = np.asarray(keys, dtype=np.uint32)
    values = np.asarray(values, dtype=np.uint32)
    if keys.ndim != 2 or keys.shape[1] != 4 or values.shape != keys.shape:
        raise ValueError(f"Expected keys/values shaped (N, 4), got {keys.shape}/{values.shape}")

    owners = ((keys[:, :, None] >> KEY_SHIFTS) & 3).astype(np.uint8)
    decks = np.empty((len(keys), 52), dtype=np.uint8)
    decks[:, DESTINATION_CARDS] = owners.reshape(len(keys), 52)

    # Packed values are player-major; native tables are strain-major.
    player_major = ((values[:, :, None] >> VALUE_SHIFTS) & 15).astype(np.int8)
    tricks = player_major.transpose(0, 2, 1)
    return decks, tricks


def validate_chunk(decks: np.ndarray, tricks: np.ndarray) -> None:
    if decks.shape[1:] != (52,) or tricks.shape[1:] != (5, 4):
        raise ValueError(f"Invalid decoded shapes: {decks.shape}, {tricks.shape}")
    if np.any(decks > 3):
        raise ValueError("Decoded deck contains an invalid owner")
    owner_counts = np.stack([(decks == player).sum(axis=1) for player in range(4)], axis=1)
    if np.any(owner_counts != 13):
        raise ValueError("Decoded deal does not give every player exactly 13 cards")
    if np.any(tricks < 0) or np.any(tricks > 13):
        raise ValueError("Decoded DDS table contains a trick count outside 0..13")


def convert(inputs: list[Path], output_dir: Path, batch_size: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_index = 0
    total = 0
    sources = []
    for source in inputs:
        packed = np.load(source, mmap_mode="r")
        if packed.ndim != 3 or packed.shape[0] != 2 or packed.shape[2] != 4:
            raise ValueError(f"Unexpected PGX array shape in {source}: {packed.shape}")
        count = int(packed.shape[1])
        sources.append({"path": str(source), "samples": count})
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            decks, tricks = decode_chunk(packed[0, start:stop], packed[1, start:stop])
            validate_chunk(decks, tricks)
            destination = output_dir / f"dds_{file_index:04d}.npz"
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, decks=decks, tricks=tricks)
            temporary.replace(destination)
            file_index += 1
            total += len(decks)
            print(f"[{total:,}] {destination.name}: {len(decks):,}")

    manifest = {
        "format": "bridge-coma-dds-v1",
        "source": "sotetsuk/dds_dataset",
        "samples": total,
        "batch_size": batch_size,
        "files": file_index,
        "inputs": sources,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(convert(args.inputs, args.output_dir, args.batch_size), indent=2))
