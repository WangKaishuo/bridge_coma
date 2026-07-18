"""Convert compressed DDS shards into one memory-mapped structured array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


DDS_DTYPE = np.dtype([
    ("decks", np.uint8, (52,)),
    ("tricks", np.int8, (5, 4)),
])


def flatten_dataset(source: Path, output: Path) -> Path:
    files = sorted(source.glob("dds_*.npz"))
    if not files:
        raise FileNotFoundError(f"No dds_*.npz files in {source}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "dds.npy"
    temporary = output / "dds.npy.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite existing output in {output}")

    counts = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            counts.append(len(archive["decks"]))
    total = sum(counts)
    records = open_memmap(temporary, mode="w+", dtype=DDS_DTYPE, shape=(total,))

    offset = 0
    for index, (path, count) in enumerate(zip(files, counts), 1):
        with np.load(path, allow_pickle=False) as archive:
            decks = np.asarray(archive["decks"], dtype=np.uint8)
            tricks = np.asarray(archive["tricks"], dtype=np.int8)
            if decks.shape != (count, 52) or tricks.shape != (count, 5, 4):
                raise ValueError(f"Unexpected DDS shapes in {path}")
            if decks.min() < 0 or decks.max() > 3:
                raise ValueError(f"Invalid card owner in {path}")
            records["decks"][offset:offset + count] = decks
            records["tricks"][offset:offset + count] = tricks
        offset += count
        print(f"[{index}/{len(files)}] {offset:,}/{total:,}", flush=True)

    records.flush()
    del records
    os.replace(temporary, target)
    manifest = {
        "format": "bridge_coma_dds_memmap_v1",
        "samples": total,
        "record_bytes": DDS_DTYPE.itemsize,
        "source": str(source),
        "source_files": len(files),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {target}: {total:,} samples, {target.stat().st_size:,} bytes")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    flatten_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
