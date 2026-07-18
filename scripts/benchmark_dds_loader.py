"""Measure DDS supply throughput independently from environment inference."""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dds_data import create_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()
    loader = create_loader(args.data)

    start = time.perf_counter()
    for _ in range(args.samples):
        loader.sample_one()
    elapsed = time.perf_counter() - start
    print(f"sample_one={args.samples / elapsed:,.0f}/s")

    batches = max(1, args.samples // 512)
    start = time.perf_counter()
    for _ in range(batches):
        loader.sample(512)
    elapsed = time.perf_counter() - start
    print(f"batch512={batches * 512 / elapsed:,.0f}/s")


if __name__ == "__main__":
    main()
