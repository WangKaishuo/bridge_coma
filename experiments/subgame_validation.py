"""Run the competitive subgame validation experiment.

Formal design
-------------
Agent A: MAPPO + FSP, task reward only.
Agent B: MAPPO + FSP, task reward + partner information gain.
Agent C: MAPPO + FSP, task reward + partner gain - beta * opponent leakage.

All three agents expose the same 571-dimensional interface and contain the same
trainable partner/RHO belief decoder.  A separate frozen Judge exists only to
measure information reward during training; it is never shared at execution.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithms.mappo import MAPPOAgent, MAPPOConfig
from env import NUM_PLAYERS
from experiments.evaluation import (
    evaluate_match,
    evaluate_match_stratified,
    make_mappo_factory,
    sample_evaluation_deals,
)
from networks.belief_net import BeliefNetwork
from networks.policy_net import ACTION_MAPPING_VERSION, OBS_DIM
from subgames.competitive_env import CompetitiveSubgameEnv
from subgames.subgame_trainer import SubgameConfig, SubgameTrainer
from utils.running_stats import RunningStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args, label: str, device: str) -> SubgameConfig:
    if label == "A":
        use_info_bonus, beta = False, 0.0
    elif label == "B":
        use_info_bonus, beta = True, 0.0
    elif label == "C":
        use_info_bonus, beta = True, args.beta
    else:
        raise ValueError(f"Unknown agent label: {label}")

    use_help_bonus = False
    help_beta = 0.0

    return SubgameConfig(
        num_rounds=args.rounds,
        steps_per_phase=2 if args.quick else args.steps_per_phase,
        deals_per_step=32 if args.quick else args.deals_per_step,
        rollout_chunk_deals=(
            64 if args.quick else args.rollout_chunk_deals
        ),
        lr=args.learning_rate,
        batch_size=16 if args.quick else args.batch_size,
        num_epochs=1 if args.quick else args.num_epochs,
        entropy_coef=args.entropy_coef,
        hidden_dim=1024,
        use_info_bonus=use_info_bonus,
        beta=beta,
        info_reward_weight=args.info_weight if use_info_bonus else 0.0,
        info_scale_calibration_deals=(64 if args.quick else args.info_calibration_deals),
        info_potential_shaping=(
            use_info_bonus and bool(getattr(args, "info_potential_shaping", False))
        ),
        use_help_bonus=use_help_bonus,
        help_beta=help_beta,
        help_reward_weight=0.0,
        help_weight_clip=10.0,
        help_return_equivalent=False,
        help_receiver_value_baseline=False,
        help_all_action_q=False,
        help_task_q_lr=3e-4,
        help_task_q_batch_size=1024,
        help_task_q_epochs=1,
        help_task_q_min_samples=4096,
        help_task_q_hidden_dim=256,
        belief_conditioned=True,
        actor_belief_coef=args.actor_belief_coef,
        freeze_belief=True,
        belief_update_epochs=1,
        belief_update_lr=args.belief_learning_rate,
        # KL is an optional baseline regularizer, not part of the main method.
        kl_lambda_start=args.kl_lambda,
        kl_lambda_end=args.kl_lambda,
        kl_anneal_frac=0.0,
        fsp_pool_size=args.fsp_pool_size,
        fsp_add_interval=args.fsp_add_interval,
        self_play=False,
        fsp_quality_gate=args.fsp_quality_gate,
        fsp_gate_eval_deals=50 if args.quick else 200,
        fsp_sl_sample_prob=args.fsp_sl_sample_prob,
        critic_prewarm_deals=64 if args.quick else 2048,
        critic_prewarm_epochs=1 if args.quick else 10,
        bc_warmup_samples=128 if args.quick else 5000,
        bc_warmup_epochs=1 if args.quick else 20,
        device=device,
    )


def load_policy_checkpoint(agent: MAPPOAgent, path: str, device: str) -> dict:
    """Load either the shared SL checkpoint or a four-actor MAPPO checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("obs_dim", OBS_DIM) != OBS_DIM:
        raise ValueError(
            f"Expected a {OBS_DIM}-dimensional policy checkpoint, got "
            f"{checkpoint.get('obs_dim')}. Execution-time BCA checkpoints are "
            "not accepted by the formal experiment."
        )

    def _load_actor(actor, state):
        target = actor.state_dict()
        for key, value in state.items():
            if key not in target:
                continue
            if target[key].shape == value.shape:
                target[key] = value.to(target[key].device)
            elif (key == "net.0.weight"
                  and target[key].ndim == 2
                  and value.ndim == 2
                  and target[key].shape[0] == value.shape[0]
                  and target[key].shape[1] > value.shape[1]):
                expanded = torch.zeros_like(target[key])
                expanded[:, :value.shape[1]] = value.to(expanded.device)
                target[key] = expanded
            else:
                raise ValueError(
                    f"Unsupported actor tensor shape for {key}: "
                    f"checkpoint={tuple(value.shape)} target={tuple(target[key].shape)}"
                )
        actor.load_state_dict(target)

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        for player in range(NUM_PLAYERS):
            _load_actor(agent.get_actor(player), state)
    else:
        keys = ("actor_n", "actor_e", "actor_s", "actor_w")
        for player, key in enumerate(keys):
            if key in checkpoint:
                _load_actor(agent.get_actor(player), checkpoint[key])
    return checkpoint


def load_belief_checkpoint(trainer: SubgameTrainer, path: str, device: str) -> None:
    if trainer.belief_net is None:
        return
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("belief_net", checkpoint)
    hidden_dim = state["trunk.0.weight"].shape[0]
    if trainer.belief_net.trunk[0].out_features != hidden_dim:
        trainer.belief_net = BeliefNetwork(hidden_dim=hidden_dim).to(device)
        if trainer.dual_info is not None:
            trainer.dual_info.belief_net = trainer.belief_net
    trainer.belief_net.load_state_dict(state)
    trainer.belief_net.eval()
    for parameter in trainer.belief_net.parameters():
        parameter.requires_grad_(False)


def save_training_checkpoint(trainer: SubgameTrainer, path: Path, label: str) -> None:
    trainer.agent.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["agent_label"] = label
    checkpoint["execution_uses_belief"] = True
    checkpoint["belief_interface"] = "571_external_internal_partner_rho_v1"
    checkpoint["action_mapping_version"] = ACTION_MAPPING_VERSION
    checkpoint["info_scale_factor"] = trainer.info_scale_factor
    checkpoint["info_scale_metadata"] = trainer.info_scale_metadata
    checkpoint["help_scale_factor"] = trainer.help_scale_factor
    checkpoint["help_reward_metadata"] = trainer.help_reward_metadata
    if trainer.help_task_q is not None:
        checkpoint["help_task_q"] = {
            key: value.detach().cpu()
            for key, value in trainer.help_task_q.state_dict().items()
        }
        checkpoint["help_task_q_samples_seen"] = trainer.help_task_q_samples_seen
        checkpoint["help_task_q_updates"] = trainer.help_task_q_updates
    if trainer.belief_net is not None:
        checkpoint["belief_net"] = {
            key: value.detach().cpu()
            for key, value in trainer.belief_net.state_dict().items()
        }
    torch.save(checkpoint, path)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Resume checkpoints are loaded with ``map_location=trainer.device`` so a
    # CUDA trainer also moves the serialized CPU RNG tensor to CUDA.  PyTorch's
    # CPU and CUDA RNG restoration APIs both require CPU ByteTensors here.
    torch_state = state["torch"].detach().cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [value.detach().cpu() for value in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def save_resume_checkpoint(
    trainer: SubgameTrainer, path: Path, label: str, completed_rounds: int
) -> None:
    """Atomically save everything required to continue at a round boundary."""
    state = {
        "format_version": 1,
        "agent_label": label,
        "completed_rounds": completed_rounds,
        "agent": trainer.agent.checkpoint_dict(),
        "belief_net": (
            trainer.belief_net.state_dict() if trainer.belief_net is not None else None
        ),
        "belief_optimizer": (
            trainer.belief_optimizer.state_dict()
            if trainer.belief_optimizer is not None else None
        ),
        "fsp_pool": {
            "max_size": trainer.fsp_pool.max_size,
            "permanent": trainer.fsp_pool._permanent,
            "pool": trainer.fsp_pool._pool,
        },
        "fsp_seeded": trainer._fsp_seeded,
        "log": trainer.log,
        "global_step": trainer._global_step,
        "vl_history": trainer._vl_history,
        "reward_stats": {
            "n": trainer.reward_stats.n,
            "mean": trainer.reward_stats.mean,
            "M2": trainer.reward_stats.M2,
        },
        "info_scale_factor": trainer.info_scale_factor,
        "info_scale_metadata": trainer.info_scale_metadata,
        "help_scale_factor": getattr(trainer, "help_scale_factor", None),
        "help_reward_metadata": getattr(trainer, "help_reward_metadata", None),
        "help_task_q": (
            trainer.help_task_q.state_dict()
            if getattr(trainer, "help_task_q", None) is not None else None
        ),
        "help_task_q_optimizer": (
            trainer.help_task_q_optimizer.state_dict()
            if getattr(trainer, "help_task_q_optimizer", None) is not None else None
        ),
        "help_task_q_samples_seen": getattr(trainer, "help_task_q_samples_seen", 0),
        "help_task_q_updates": getattr(trainer, "help_task_q_updates", 0),
        "rng": _rng_state(),
    }
    # Direct-DRI actor-only updates are applied transactionally between task
    # rounds. Preserve their lifecycle/audit ledger across the next ordinary
    # task checkpoint instead of silently dropping unknown top-level fields.
    state.update(getattr(trainer, "_direct_auxiliary_resume_metadata", {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    print(f"  [Checkpoint] round {completed_rounds} -> {path}")


def load_resume_checkpoint(
    trainer: SubgameTrainer, path: Path, expected_label: str
) -> int:
    state = torch.load(path, map_location=trainer.device, weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError(f"Unsupported resume checkpoint format: {path}")
    if state.get("agent_label") != expected_label:
        raise ValueError(
            f"Resume checkpoint is Agent {state.get('agent_label')}, "
            f"but Agent {expected_label} was requested"
        )
    trainer.agent.load_checkpoint_dict(state["agent"])
    if trainer.belief_net is not None and state.get("belief_net") is not None:
        trainer.belief_net.load_state_dict(state["belief_net"])
    if trainer.belief_optimizer is not None and state.get("belief_optimizer") is not None:
        trainer.belief_optimizer.load_state_dict(state["belief_optimizer"])
    pool = state["fsp_pool"]
    trainer.fsp_pool._permanent = pool["permanent"]
    trainer.fsp_pool._pool = pool["pool"]
    configured_max = int(getattr(
        getattr(trainer, "config", None), "fsp_pool_size", pool["max_size"]
    ))
    if configured_max > 0:
        if configured_max < len(trainer.fsp_pool._permanent):
            raise ValueError(
                "fsp_pool_size is smaller than the permanent FSP membership"
            )
        trainer.fsp_pool.max_size = configured_max
        capacity = configured_max - len(trainer.fsp_pool._permanent)
        trainer.fsp_pool._pool = trainer.fsp_pool._pool[-capacity:] if capacity else []
    else:
        trainer.fsp_pool.max_size = pool["max_size"]
    trainer._fsp_seeded = state["fsp_seeded"]
    trainer._fsp_actor_cache.clear()
    trainer._fsp_cache_source = None
    trainer.log = state["log"]
    trainer._global_step = state["global_step"]
    trainer._vl_history = state["vl_history"]
    for key, value in state["reward_stats"].items():
        setattr(trainer.reward_stats, key, value)
    trainer.info_scale_factor = state["info_scale_factor"]
    trainer.info_scale_metadata = state["info_scale_metadata"]
    if hasattr(trainer, "help_scale_factor"):
        trainer.help_scale_factor = state.get("help_scale_factor")
    if hasattr(trainer, "help_reward_metadata"):
        trainer.help_reward_metadata = state.get("help_reward_metadata")
    if (getattr(trainer, "help_task_q", None) is not None
            and state.get("help_task_q") is not None):
        trainer.help_task_q.load_state_dict(state["help_task_q"])
    if (getattr(trainer, "help_task_q_optimizer", None) is not None
            and state.get("help_task_q_optimizer") is not None):
        trainer.help_task_q_optimizer.load_state_dict(state["help_task_q_optimizer"])
    if hasattr(trainer, "help_task_q_samples_seen"):
        trainer.help_task_q_samples_seen = int(
            state.get("help_task_q_samples_seen", 0)
        )
    if hasattr(trainer, "help_task_q_updates"):
        trainer.help_task_q_updates = int(state.get("help_task_q_updates", 0))
    trainer._direct_auxiliary_resume_metadata = {
        key: state[key]
        for key in (
            "direct_auxiliary_lifecycle",
            "direct_auxiliary_audits",
            "direct_auxiliary_protocol",
        )
        if key in state
    }
    _restore_rng_state(state["rng"])
    return int(state["completed_rounds"])


def load_deployment_agent(path: Path, device: str) -> MAPPOAgent:
    """Load a trained 571-dimensional agent without constructing a trainer."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    obs_dim = checkpoint.get("obs_dim", OBS_DIM)
    if obs_dim != OBS_DIM:
        raise ValueError(f"Expected {OBS_DIM}-dimensional agent, got {obs_dim}: {path}")
    mapping_version = checkpoint.get("action_mapping_version")
    if mapping_version != ACTION_MAPPING_VERSION:
        raise ValueError(
            f"Checkpoint uses an unknown or legacy action mapping: {path}. "
            "Retrain it with the corrected OpenSpiel 52..89 mapping."
        )
    agent = MAPPOAgent(MAPPOConfig(
        device=device,
        obs_dim=OBS_DIM,
        hidden_dim=checkpoint.get("hidden_dim", 1024),
        actor_belief_conditioned=checkpoint.get(
            "actor_belief_conditioned", False
        ),
        actor_belief_hidden_dim=checkpoint.get(
            "actor_belief_hidden_dim", checkpoint.get("hidden_dim", 1024)
        ),
    ))
    agent.load(str(path))
    return agent


def format_result(row: str, column: str, result) -> str:
    return (
        f"{row} vs {column}: {result.mean_imp:+.3f} +/- {result.std_imp:.3f} IMP "
        f"(95% CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}]); "
        f"{result.wins}/{result.losses}/{result.ties}"
    )


def run(
    args,
    env_cls=CompetitiveSubgameEnv,
    experiment_name: str = "controlled_1h_1s",
) -> None:
    pipeline_started = time.perf_counter()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env_started = time.perf_counter()
    env = env_cls(args.data)
    print(f"[Timing] environment_init={time.perf_counter() - env_started:.1f}s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [] if args.eval_only else [
        label for label in args.train_agents if label not in args.skip_agent
    ]
    if not args.eval_only and not labels:
        raise ValueError("At least one training agent is required")
    if args.resume and len(labels) != 1:
        raise ValueError("--resume requires exactly one --train-agents label")
    trainers: dict[str, SubgameTrainer] = {}
    trainer_init_started = time.perf_counter()
    for label in labels:
        config = build_config(args, label, device)
        trainers[label] = SubgameTrainer(env, config, reward_stats=RunningStats())
    print(f"[Timing] trainer_init={time.perf_counter() - trainer_init_started:.1f}s")

    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"Training data: {args.data}")
    print(f"Policy observation: {OBS_DIM} dimensions")
    print("Policy interface: 571 dimensions; internal partner/RHO belief head enabled")
    print("Frozen Judge at execution: disabled")

    start_rounds = {label: 0 for label in labels}
    if args.resume:
        label = labels[0]
        start_rounds[label] = load_resume_checkpoint(
            trainers[label], Path(args.resume), label
        )
        print(f"Resumed Agent {label} after round {start_rounds[label]}")

    if getattr(args, "capture_task_gradients_output", None):
        for trainer in trainers.values():
            trainer.enable_task_actor_gradient_capture(True)

    if labels and not args.resume:
        reference_label = labels[0]
        reference = trainers[reference_label]
        if args.sl_checkpoint and os.path.exists(args.sl_checkpoint):
            load_policy_checkpoint(reference.agent, args.sl_checkpoint, device)
        else:
            print(
                f"Agent {reference_label}: no SL checkpoint; running one shared "
                "rule-based warmup"
            )
            reference.run_bc_warmup()

        shared_initial_state = {
            key: value.detach().clone()
            for key, value in reference.agent.state_dict().items()
        }
        for label, trainer in trainers.items():
            if label != reference_label:
                trainer.agent.load_state_dict(shared_initial_state)
            if args.kl_lambda > 0:
                trainer.set_bc_anchor(trainer.agent)
        print(f"Shared initial Actor+Critic parameters copied to: {', '.join(labels)}")

    belief_trainers = [trainers[label] for label in labels]
    info_trainers = [
        trainers[label] for label in ("B", "C")
        if label in trainers and trainers[label].config.use_info_bonus
    ]
    if belief_trainers and not args.resume:
        if args.belief_checkpoint and os.path.exists(args.belief_checkpoint):
            for trainer in belief_trainers:
                load_belief_checkpoint(trainer, args.belief_checkpoint, device)
        elif args.belief_pretrain_rounds > 0:
            if not info_trainers:
                raise RuntimeError(
                    "Belief pretraining without a checkpoint requires Agent B or C"
                )
            set_seed(args.seed + 10_000)
            source = info_trainers[0]
            source.pretrain_belief(
                num_rounds=args.belief_pretrain_rounds,
                deals_per_round=200 if args.quick else args.belief_pretrain_deals,
                epochs_per_round=2 if args.quick else 5,
                max_epochs=args.belief_pretrain_max_epochs,
            )
            for trainer in belief_trainers:
                if trainer is source:
                    continue
                trainer.belief_net.load_state_dict(source.belief_net.state_dict())
                if hasattr(source, "_pretrain_replay"):
                    trainer._pretrain_replay = {
                        key: value.clone()
                        for key, value in source._pretrain_replay.items()
                    }
        else:
            raise FileNotFoundError(
                "A frozen Judge checkpoint is required for the internal belief "
                "architecture and information calibration"
            )

        for trainer in belief_trainers:
            trainer.initialize_actor_beliefs_from_judge()

    if info_trainers and not args.resume:
        # The calibration seed is independent of training and is identical in
        # split B/C Colab runs.  Calibration always uses partner-only deltas.
        set_seed(args.seed + 20_000)
        scale_source = info_trainers[0]
        calibration_started = time.perf_counter()
        scale_source.calibrate_info_scale(
            scale_source.config.info_scale_calibration_deals
        )
        print(
            f"[Timing] information_calibration="
            f"{time.perf_counter() - calibration_started:.1f}s"
        )
        for trainer in info_trainers[1:]:
            trainer.copy_info_scale_from(scale_source)

    logs = {}
    for label, trainer in trainers.items():
        print(f"\n=== Training Agent {label} ===")
        # Common random numbers reduce variance in A/B/C treatment comparisons.
        if not args.resume:
            set_seed(args.seed)
        resume_path = (
            Path(args.resume) if args.resume else
            output_dir / f"agent_{label.lower()}_seed{args.seed}.resume.pt"
        )

        def checkpoint_callback(current_trainer, completed_rounds, _log_entry):
            if getattr(args, "capture_task_gradients_output", None):
                gradients = {
                    seat: current_trainer.captured_task_actor_gradient(seat)
                    for seat in range(4)
                }
                missing = [seat for seat, value in gradients.items() if value is None]
                if missing:
                    raise RuntimeError(
                        f"task gradient capture missing seats: {missing}"
                    )
                gradient_path = Path(args.capture_task_gradients_output)
                gradient_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_gradient_path = gradient_path.with_suffix(
                    gradient_path.suffix + ".tmp"
                )
                torch.save({
                    "schema_version": "adjacent-task-policy-gradients-v2",
                    "gradient_definition": (
                        "pure_task_policy_loss_pre_clipping_first_minibatch"
                    ),
                    "source_resume": str(args.resume),
                    "completed_round": completed_rounds,
                    "gradients": gradients,
                    "seat_metadata": {
                        seat: current_trainer.
                        captured_task_actor_gradient_metadata(seat)
                        for seat in range(4)
                    },
                }, temporary_gradient_path)
                temporary_gradient_path.replace(gradient_path)
            if (completed_rounds % args.checkpoint_interval == 0
                    or completed_rounds == args.rounds):
                checkpoint_started = time.perf_counter()
                save_resume_checkpoint(
                    current_trainer, resume_path, label, completed_rounds
                )
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                _log_entry.setdefault("timing_seconds", {})[
                    "checkpoint"
                ] = checkpoint_seconds
                print(f"  [Timing] checkpoint={checkpoint_seconds:.1f}s")

        logs[label] = trainer.run(
            num_rounds=args.rounds,
            start_round=start_rounds[label],
            skip_warmup=bool(args.resume),
            round_callback=checkpoint_callback,
        )
        final_save_started = time.perf_counter()
        save_training_checkpoint(
            trainer, output_dir / f"agent_{label.lower()}_seed{args.seed}.pt", label
        )
        print(f"  [Timing] final_checkpoint={time.perf_counter() - final_save_started:.1f}s")

    should_evaluate = (
        args.eval_only or args.evaluate or set(labels) == set("ABC")
    )
    if not should_evaluate:
        print(f"[Timing] process_total={time.perf_counter() - pipeline_started:.1f}s")
        print("Training complete. Cross-agent evaluation deferred to an eval-only run.")
        return

    checkpoint_overrides = {
        "A": args.agent_a,
        "B": args.agent_b,
        "C": args.agent_c,
    }
    deployment_agents = {
        label: trainer.agent for label, trainer in trainers.items()
    }
    for label in "ABC":
        if label in deployment_agents:
            continue
        path = Path(checkpoint_overrides[label]) if checkpoint_overrides[label] else (
            output_dir / f"agent_{label.lower()}_seed{args.seed}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Agent {label} checkpoint for evaluation: {path}"
            )
        deployment_agents[label] = load_deployment_agent(path, device)

    eval_data = args.eval_data or args.data
    eval_env = env if eval_data == args.data else env_cls(eval_data)
    print(f"Evaluation data: {eval_data}")
    deals = sample_evaluation_deals(eval_env, args.eval_deals, args.eval_seed)
    factories = {
        label: make_mappo_factory(agent)
        for label, agent in deployment_agents.items()
    }
    published_labels = []
    for item in args.published_agent:
        if "=" not in item:
            raise ValueError("--published-agent must be NAME=module:builder")
        name, spec = item.split("=", 1)
        if not name or name in factories:
            raise ValueError(f"Invalid or duplicate published-agent name: {name}")
        from experiments.published_agent_adapter import load_published_factory
        factories[name] = load_published_factory(spec)
        published_labels.append(name)
    results = {}
    for row, column in (("B", "A"), ("C", "A"), ("C", "B")):
        if row not in factories or column not in factories:
            continue
        stratified = evaluate_match_stratified(
            eval_env, deals, factories[row], factories[column]
        )
        result = stratified.overall
        overall_summary = {
            key: value for key, value in asdict(result).items() if key != "imps"
        }
        results[f"{row}_vs_{column}"] = {
            **overall_summary,
            "strata": {
                name: {
                    key: value
                    for key, value in asdict(getattr(stratified, name)).items()
                    if key != "imps"
                }
                for name in ("competitive", "non_competitive", "mixed")
            },
        }
        print(format_result(row, column, result))

    for baseline in published_labels:
        for label in "ABC":
            if label not in factories:
                continue
            stratified = evaluate_match_stratified(
                eval_env, deals, factories[label], factories[baseline]
            )
            result = stratified.overall
            results[f"{label}_vs_{baseline}"] = {
                **{
                    key: value
                    for key, value in asdict(result).items()
                    if key != "imps"
                },
                "strata": {
                    name: {
                        key: value
                        for key, value in asdict(getattr(stratified, name)).items()
                        if key != "imps"
                    }
                    for name in ("competitive", "non_competitive", "mixed")
                },
            }
            print(format_result(label, baseline, result))

    summary = {
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "experiment": experiment_name,
        "training_distribution": type(env).__name__,
        "evaluation_distribution": type(eval_env).__name__,
        "execution_uses_belief": True,
        "belief_interface": "571_external_internal_partner_rho_v1",
        "info_scales": {
            label: trainer.info_scale_metadata
            for label, trainer in trainers.items()
            if trainer.info_scale_metadata is not None
        },
        "config": vars(args),
        "matchups": results,
    }
    with open(output_dir / f"summary_seed{args.seed}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def build_parser(
    default_data: str = "data/competitive_100k.npz",
    default_eval_data: str | None = None,
    default_output_dir: str = "results/competitive_v2",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=default_data)
    parser.add_argument("--eval-data", default=default_eval_data)
    parser.add_argument("--sl-checkpoint", default="data/sl_base.pt")
    parser.add_argument("--belief-checkpoint", default="data/sl_base_bca.pt")
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--steps-per-phase", type=int, default=64)
    parser.add_argument("--deals-per-step", type=int, default=512)
    parser.add_argument(
        "--rollout-chunk-deals", type=int, default=8192,
        help="Maximum deals retained as episode objects at once",
    )
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--eval-deals", type=int, default=5000)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--info-weight", type=float, default=0.05)
    parser.add_argument("--info-calibration-deals", type=int, default=2048)
    parser.add_argument(
        "--info-potential-shaping",
        action="store_true",
        help=(
            "Replace legacy immediate information deltas with strict per-seat "
            "gamma*Phi(next own turn)-Phi(now) shaping"
        ),
    )
    parser.add_argument("--actor-belief-coef", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--belief-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-lambda", type=float, default=0.0)
    parser.add_argument("--belief-pretrain-rounds", type=int, default=0)
    parser.add_argument("--belief-pretrain-deals", type=int, default=2000)
    parser.add_argument("--belief-pretrain-max-epochs", type=int, default=50)
    parser.add_argument("--fsp-sl-sample-prob", type=float, default=0.30)
    parser.add_argument(
        "--fsp-pool-size", type=int, default=10,
        help="Total FSP members including the permanent BC baseline",
    )
    parser.add_argument("--fsp-add-interval", type=int, default=1)
    parser.add_argument("--fsp-quality-gate", action="store_true")
    parser.add_argument(
        "--resume", help="Resume-state checkpoint; requires one training agent"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--capture-task-gradients-output",
        help=(
            "diagnostic-only path for one pre-clipping task actor gradient per "
            "seat from the adjacent round"
        ),
    )
    parser.add_argument(
        "--train-agents", nargs="+", choices=list("ABC"), default=list("ABC"),
        help="Agents to train in this Colab session, e.g. --train-agents B",
    )
    parser.add_argument("--skip-agent", choices=list("ABC"), action="append", default=[])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate after a partial training run by loading missing checkpoints",
    )
    parser.add_argument("--agent-a", help="Agent A checkpoint for evaluation")
    parser.add_argument("--agent-b", help="Agent B checkpoint for evaluation")
    parser.add_argument("--agent-c", help="Agent C checkpoint for evaluation")
    parser.add_argument(
        "--published-agent",
        action="append",
        default=[],
        metavar="NAME=MODULE:BUILDER",
        help="Observation-only published baseline adapter; may be repeated",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def parse_args(**parser_defaults) -> argparse.Namespace:
    return build_parser(**parser_defaults).parse_args()


if __name__ == "__main__":
    run(parse_args())
