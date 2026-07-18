"""Benchmark the real bridge training pipeline without saving a checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil
import torch

from experiments.subgame_validation import (
    load_belief_checkpoint,
    load_policy_checkpoint,
)
from subgames.competitive_env import CompetitiveSubgameEnv
from subgames.subgame_trainer import SubgameConfig, SubgameTrainer
from utils.running_stats import RunningStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GpuSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    command, text=True, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).strip().splitlines()[0]
                utilization, memory = output.split(",")[:2]
                self.samples.append((
                    float(utilization),
                    float(memory),
                    float(self._process.cpu_percent(None)),
                    float(psutil.cpu_percent(None)),
                ))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        utilization = np.asarray([sample[0] for sample in self.samples])
        memory = np.asarray([sample[1] for sample in self.samples])
        process_cpu = np.asarray([sample[2] for sample in self.samples])
        system_cpu = np.asarray([sample[3] for sample in self.samples])
        return {
            "samples": int(len(self.samples)),
            "gpu_util_mean_pct": float(utilization.mean()),
            "gpu_util_median_pct": float(np.median(utilization)),
            "gpu_util_p95_pct": float(np.percentile(utilization, 95)),
            "gpu_memory_mean_mib": float(memory.mean()),
            "gpu_memory_max_mib": float(memory.max()),
            "process_cpu_mean_pct_one_core_100": float(process_cpu.mean()),
            "process_cpu_p95_pct_one_core_100": float(np.percentile(process_cpu, 95)),
            "system_cpu_mean_pct": float(system_cpu.mean()),
            "system_cpu_p95_pct": float(np.percentile(system_cpu, 95)),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
        }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def instrument(trainer: SubgameTrainer) -> dict[str, float]:
    timings: dict[str, float] = defaultdict(float)
    methods = {
        "collect": "_collect_episodes_batch",
        "info_reward": "_compute_info_bonus",
        "ppo_update": "_safe_update",
        "critic_warmup": "critic_warmup",
    }
    for label, method_name in methods.items():
        original = getattr(trainer, method_name)

        def wrapped(*args, _original=original, _label=label, **kwargs):
            synchronize()
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            synchronize()
            timings[_label] += time.perf_counter() - started
            return result

        setattr(trainer, method_name, wrapped)
    return timings


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is unavailable; pass --cpu only for diagnostics")
    set_seed(args.seed)
    device = "cpu" if args.cpu else "cuda"
    env = CompetitiveSubgameEnv(args.data)
    use_info = args.agent in ("B", "C")
    config = SubgameConfig(
        num_rounds=args.rounds,
        steps_per_phase=args.steps_per_phase,
        deals_per_step=args.deals_per_step,
        collector_workers=args.collector_workers,
        fast_observation_encoding=args.fast_observation_encoding,
        lr=3e-6,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        entropy_coef=0.01,
        hidden_dim=1024,
        use_info_bonus=use_info,
        beta=0.05 if args.agent == "C" else 0.0,
        info_reward_weight=0.05 if use_info else 0.0,
        info_scale_calibration_deals=args.calibration_deals,
        belief_conditioned=True,
        actor_belief_coef=0.1,
        freeze_belief=True,
        fsp_pool_size=10,
        fsp_add_interval=1,
        self_play=False,
        fsp_quality_gate=False,
        fsp_sl_sample_prob=0.30,
        critic_prewarm_deals=args.prewarm_deals,
        critic_prewarm_epochs=args.prewarm_epochs,
        device=device,
    )
    setup_started = time.perf_counter()
    trainer = SubgameTrainer(env, config, reward_stats=RunningStats())
    load_policy_checkpoint(trainer.agent, args.sl_checkpoint, device)
    load_belief_checkpoint(trainer, args.belief_checkpoint, device)
    trainer.initialize_actor_beliefs_from_judge()
    calibration_seconds = 0.0
    if use_info:
        synchronize()
        calibration_started = time.perf_counter()
        trainer.calibrate_info_scale(args.calibration_deals)
        synchronize()
        calibration_seconds = time.perf_counter() - calibration_started
    setup_seconds = time.perf_counter() - setup_started

    timings = instrument(trainer)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler(args.sample_interval)
    sampler.start()
    synchronize()
    training_started = time.perf_counter()
    trainer.run(num_rounds=args.rounds)
    synchronize()
    training_seconds = time.perf_counter() - training_started
    sampler.stop()

    total_deals = 2 * args.rounds * args.steps_per_phase * args.deals_per_step
    accounted = sum(timings.values())
    result = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "agent": args.agent,
        "total_deals": total_deals,
        "training_seconds": training_seconds,
        "deals_per_second": total_deals / training_seconds,
        "seconds_per_million_deals": training_seconds * 1_000_000 / total_deals,
        "setup_seconds_including_calibration": setup_seconds,
        "calibration_seconds": calibration_seconds,
        "timings_seconds": dict(timings),
        "unattributed_seconds": max(0.0, training_seconds - accounted),
        "torch_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0.0
        ),
        **sampler.summary(),
        "config": {
            "rounds": args.rounds,
            "steps_per_phase": args.steps_per_phase,
            "deals_per_step": args.deals_per_step,
            "collector_workers": args.collector_workers,
            "fast_observation_encoding": args.fast_observation_encoding,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "prewarm_deals": args.prewarm_deals,
            "prewarm_epochs": args.prewarm_epochs,
        },
    }
    print("\nBENCHMARK_RESULT")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/competitive_500k.npz")
    parser.add_argument("--sl-checkpoint", default="results/sl_base.pt")
    parser.add_argument("--belief-checkpoint", default="results/sl_base_bca.pt")
    parser.add_argument("--agent", choices=list("ABC"), default="B")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--steps-per-phase", type=int, default=4)
    parser.add_argument("--deals-per-step", type=int, default=512)
    parser.add_argument("--collector-workers", type=int, default=1)
    parser.add_argument("--fast-observation-encoding", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--prewarm-deals", type=int, default=256)
    parser.add_argument("--prewarm-epochs", type=int, default=1)
    parser.add_argument("--calibration-deals", type=int, default=256)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
