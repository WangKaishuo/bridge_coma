#!/usr/bin/env python3
"""
Drift Sweep Runner
==================

Run convention drift quantification experiments across multiple λ_KL values
and seeds. Wraps subgame_validation.py for batch execution.

P108: All observations use OpenSpiel-native 571-dim encoding.

Experiment 1 (571-dim): Measures drift advantage without BCA.
  → Prior work scenario (opponents can't interpret drifted bids)

Experiment 2 (667-dim BCA): Measures drift advantage with BCA.
  → Our framework (opponents use BeliefNet to interpret bids)

Usage:
  # Experiment 1: 571-dim drift sweep
  python drift_sweep.py \
      --mode 571 \
      --sl_checkpoint results/sl_base_571.pt \
      --data data/competitive_500k.npz \
      --rounds 10 --eval_deals 2000

  # Experiment 2: 667-dim BCA drift sweep
  python drift_sweep.py \
      --mode 667 \
      --sl_checkpoint results/sl_base_571.pt \
      --data data/competitive_500k.npz \
      --rounds 10 --eval_deals 2000

  # Quick smoke test (1 seed, 2 lambdas, 3 rounds)
  python drift_sweep.py \
      --mode 571 \
      --sl_checkpoint results/sl_base_571.pt \
      --data data/competitive_500k.npz \
      --lambdas 0.0 0.3 --seeds 42 --rounds 3 --quick

Output:
  results/drift_sweep_{mode}/
    ├── lambda0.0_seed42/      (per-run save_dir)
    ├── lambda0.0_seed123/
    ├── ...
    └── sweep_summary.json     (aggregated results)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0]
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def run_single(mode: str, lam: float, seed: int, args) -> dict:
    """Run a single (λ, seed) experiment via subgame_validation.py."""
    
    run_name = f"lambda{lam}_seed{seed}"
    save_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "experiments/subgame_validation.py",
        "--data", args.data,
        "--sl_checkpoint", args.sl_checkpoint,
        "--seed", str(seed),
        "--rounds", str(args.rounds),
        "--eval_deals", str(args.eval_deals),
        "--kl_lambda", str(lam),
        "--save_dir", save_dir,
    ]

    # Select which agent to train
    if getattr(args, "agent_b_only", False):
        cmd.append("--agent_b_only")
    else:
        cmd.append("--agent_a_only")
    
    if mode == "571":
        cmd.append("--no_belief_conditioned")
    else:
        # 667-dim BCA: pass standalone BeliefNet checkpoint if provided
        if getattr(args, "belief_checkpoint", None):
            cmd += ["--belief_checkpoint", args.belief_checkpoint]

    if args.quick:
        cmd.append("--quick")
    
    print(f"\n{'='*60}")
    print(f"  Running: mode={mode}  λ={lam}  seed={seed}")
    print(f"  Save: {save_dir}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=not args.verbose, text=True)
    elapsed = time.time() - t0
    
    # Try to load the report JSON
    report_path = os.path.join(save_dir, f"report_seed{seed}.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    
    run_result = {
        "mode": mode,
        "lambda": lam,
        "seed": seed,
        "elapsed_sec": round(elapsed, 1),
        "returncode": result.returncode,
        "report": report,
    }
    
    if result.returncode != 0:
        print(f"  ⚠️  FAILED (returncode={result.returncode})")
        if not args.verbose and result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line}")
    else:
        # Extract key metrics from report
        _b_only = getattr(args, "agent_b_only", False)
        if _b_only:
            vs_sl = report.get("b_vs_sl_imp", "?")
            print(f"  ✅ Done in {elapsed:.0f}s  B vs SL: {vs_sl} IMP")
        else:
            vs_sl = report.get("a_vs_sl_imp", "?")
            print(f"  ✅ Done in {elapsed:.0f}s  A vs SL: {vs_sl} IMP")
    
    return run_result


def main():
    parser = argparse.ArgumentParser(
        description="Convention Drift Sweep (multi-λ × multi-seed)")
    parser.add_argument("--mode", required=True, choices=["571", "667"],
                        help="571 = no BCA (prior work scenario), "
                             "667 = BCA (our framework)")
    parser.add_argument("--data", required=True,
                        help="Path to competitive data (npz)")
    parser.add_argument("--sl_checkpoint", required=True,
                        help="SL checkpoint (sl_base_571.pt for 571, sl_base_571.pt for 667)")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=DEFAULT_LAMBDAS,
                        help=f"λ_KL values (default: {DEFAULT_LAMBDAS})")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=DEFAULT_SEEDS,
                        help=f"Seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--eval_deals", type=int, default=3000)
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: results/drift_sweep_{mode})")
    parser.add_argument("--belief_checkpoint", default=None,
                        help="Standalone BeliefNet checkpoint (Stage A output). "
                             "Only used in --mode 667. Passed to subgame_validation.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for debugging")
    parser.add_argument("--verbose", action="store_true",
                        help="Show subprocess output")
    parser.add_argument("--agent_b_only", action="store_true",
                        help="Train Agent B (MAPPO+BCA+r_info) instead of Agent A. "
                             "For r_info experiments.")
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = f"results/drift_sweep_{args.mode}"
    os.makedirs(args.out_dir, exist_ok=True)
    
    n_total = len(args.lambdas) * len(args.seeds)
    print(f"[Drift Sweep] mode={args.mode}  "
          f"lambdas={args.lambdas}  seeds={args.seeds}")
    print(f"[Drift Sweep] P108 (OpenSpiel 571-dim obs)")
    print(f"[Drift Sweep] {n_total} runs total  → {args.out_dir}")
    
    # ── Run all experiments ─────────────────────────────────────────
    all_results = []
    t_start = time.time()
    
    for lam in args.lambdas:
        for seed in args.seeds:
            # Skip if already completed
            report_path = os.path.join(
                args.out_dir, f"lambda{lam}_seed{seed}",
                f"report_seed{seed}.json")
            if os.path.exists(report_path):
                print(f"\n  Skipping λ={lam} seed={seed} (already completed)")
                with open(report_path) as f:
                    report = json.load(f)
                all_results.append({
                    "mode": args.mode, "lambda": lam, "seed": seed,
                    "elapsed_sec": 0, "returncode": 0, "report": report,
                })
                continue
            
            result = run_single(args.mode, lam, seed, args)
            all_results.append(result)
    
    total_time = time.time() - t_start
    
    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DRIFT SWEEP SUMMARY ({args.mode}-dim)")
    print(f"{'='*60}")
    
    # Group by lambda
    from collections import defaultdict
    by_lambda = defaultdict(list)
    for r in all_results:
        by_lambda[r["lambda"]].append(r)
    
    _b_only = getattr(args, "agent_b_only", False)
    _imp_key = "b_vs_sl_imp" if _b_only else "a_vs_sl_imp"
    _agent_label = "B" if _b_only else "A"

    summary_table = []
    for lam in sorted(by_lambda.keys()):
        runs = by_lambda[lam]
        imps = []
        for r in runs:
            imp = r.get("report", {}).get(_imp_key)
            if imp is not None:
                imps.append(imp)
        
        if imps:
            import numpy as np
            mean_imp = np.mean(imps)
            std_imp = np.std(imps)
            entry = {
                "lambda": lam,
                "n_seeds": len(imps),
                "mean_imp": round(mean_imp, 3),
                "std_imp": round(std_imp, 3),
            }
        else:
            entry = {"lambda": lam, "n_seeds": 0,
                     "mean_imp": None, "std_imp": None}
        summary_table.append(entry)
        
        status = (f"{entry['mean_imp']:+.3f} ± {entry['std_imp']:.3f}"
                  if entry['mean_imp'] is not None else "NO DATA")
        print(f"  λ={lam:.1f}  n={entry['n_seeds']}  "
              f"{_agent_label} vs SL: {status} IMP")
    
    print(f"\n  Total time: {total_time/3600:.1f} hours")
    
    # Save summary
    summary_path = os.path.join(args.out_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "lambdas": args.lambdas,
            "seeds": args.seeds,
            "total_time_sec": round(total_time, 1),
            "summary": summary_table,
            "all_results": all_results,
        }, f, indent=2)
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
