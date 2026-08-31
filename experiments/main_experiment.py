"""Run the unrestricted complete-auction Bridge-COMA main experiment.

This entry point is deliberately separate from ``subgame_validation.py`` so a
formal run cannot accidentally inherit the controlled 1H-1S distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.subgame_validation import build_parser, run
from subgames.unrestricted_env import UnrestrictedBiddingEnv


def parse_args():
    parser = build_parser(
        default_data="data/pgx_train_10m",
        default_eval_data="data/pgx_eval_500k",
        default_output_dir="results/main_unrestricted",
    )
    parser.description = __doc__
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Run one short full-scale-shape round (real 512 rollout batches, "
            "small deal count and evaluation) before a long run"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot:
        args.rounds = 1
        args.steps_per_phase = 8
        args.deals_per_step = 512
        args.batch_size = 512
        args.info_calibration_deals = 512
        args.eval_deals = 500
        args.checkpoint_interval = 1
        if args.output_dir == "results/main_unrestricted":
            args.output_dir = "results/main_unrestricted_pilot"
    run(
        args,
        env_cls=UnrestrictedBiddingEnv,
        experiment_name="unrestricted_complete_auction_main",
    )


if __name__ == "__main__":
    main()
