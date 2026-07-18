import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.subgame_validation import (
    load_resume_checkpoint,
    save_resume_checkpoint,
)
from utils.convert_pgx_dds import decode_chunk
from utils.dds_data import MemmapDDSLoader
from utils.flatten_dds_dataset import flatten_dataset
from utils.running_stats import RunningStats
from subgames.subgame_trainer import FlatRolloutBuffer


class _DummyAgent:
    def __init__(self, value):
        self.value = torch.tensor([value], dtype=torch.float32)

    def checkpoint_dict(self):
        return {"value": self.value.clone()}

    def load_checkpoint_dict(self, state):
        self.value = state["value"].clone()


class _DummyTrainer:
    def __init__(self, value=1.0):
        self.device = "cpu"
        self.agent = _DummyAgent(value)
        self.belief_net = None
        self.belief_optimizer = None
        self.fsp_pool = SimpleNamespace(
            max_size=3,
            _permanent=[{"actor_n": {"w": torch.tensor([1.0])}}],
            _pool=[{"actor_n": {"w": torch.tensor([2.0])}}],
        )
        self._fsp_seeded = True
        self._fsp_actor_cache = {"stale": object()}
        self._fsp_cache_source = object()
        self.log = [{"round": 1}]
        self._global_step = 17
        self._vl_history = [0.5]
        self.reward_stats = RunningStats()
        self.reward_stats.update_batch(np.array([1.0, 3.0]))
        self.info_scale_factor = 0.25
        self.info_scale_metadata = {"source": "test"}


class DDSConversionTest(unittest.TestCase):
    def test_packed_round_trip(self):
        deck = np.arange(52, dtype=np.uint8) % 4
        tricks = (np.arange(20, dtype=np.int8).reshape(5, 4) % 14)

        keys = np.zeros(4, dtype=np.uint32)
        for pgx_suit in range(4):
            project_suit = 3 - pgx_suit
            for pgx_rank, shift in enumerate(range(24, -1, -2)):
                project_rank = 12 if pgx_rank == 0 else pgx_rank - 1
                owner = deck[project_suit * 13 + project_rank]
                keys[pgx_suit] |= np.uint32(owner) << np.uint32(shift)

        values = np.zeros(4, dtype=np.uint32)
        for player in range(4):
            for strain, shift in enumerate((16, 12, 8, 4, 0)):
                values[player] |= np.uint32(tricks[strain, player]) << np.uint32(shift)

        decoded_decks, decoded_tricks = decode_chunk(keys[None], values[None])
        np.testing.assert_array_equal(decoded_decks[0], deck)
        np.testing.assert_array_equal(decoded_tricks[0], tricks)

    def test_flattened_memmap_preserves_every_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shards"
            output = root / "flat"
            source.mkdir()
            expected_decks = []
            expected_tricks = []
            for index, count in enumerate((3, 2)):
                decks = np.tile(np.arange(52, dtype=np.uint8) % 4, (count, 1))
                decks = (decks + index) % 4
                tricks = np.full((count, 5, 4), index + 6, dtype=np.int8)
                np.savez_compressed(
                    source / f"dds_{index:04d}.npz", decks=decks, tricks=tricks
                )
                expected_decks.append(decks)
                expected_tricks.append(tricks)

            target = flatten_dataset(source, output)
            records = np.load(target, mmap_mode="r", allow_pickle=False)
            np.testing.assert_array_equal(
                records["decks"], np.concatenate(expected_decks)
            )
            np.testing.assert_array_equal(
                records["tricks"], np.concatenate(expected_tricks)
            )
            loader = MemmapDDSLoader(str(output))
            self.assertEqual(len(loader), 5)
            hands, tricks = loader.sample(4)
            self.assertEqual(hands.shape, (4, 4, 52))
            self.assertEqual(tricks.shape, (4, 5, 4))
            loader.close()
            records._mmap.close()


class ResumeCheckpointTest(unittest.TestCase):
    def test_complete_state_and_rng_are_restored(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        source = _DummyTrainer()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_resume_checkpoint(source, path, "B", completed_rounds=4)
            expected_rng = (random.random(), np.random.rand(), torch.rand(1).item())

            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored = _DummyTrainer(value=99.0)
            completed = load_resume_checkpoint(restored, path, "B")

        self.assertEqual(completed, 4)
        self.assertEqual(restored.agent.value.item(), 1.0)
        self.assertEqual(restored._global_step, 17)
        self.assertEqual(restored.log, [{"round": 1}])
        self.assertEqual(len(restored.fsp_pool._permanent), 1)
        self.assertEqual(len(restored.fsp_pool._pool), 1)
        self.assertEqual(restored._fsp_actor_cache, {})
        self.assertIsNone(restored._fsp_cache_source)
        self.assertEqual(restored.reward_stats.n, 2)
        actual_rng = (random.random(), np.random.rand(), torch.rand(1).item())
        np.testing.assert_allclose(actual_rng, expected_rng)


class PackedRolloutBufferTest(unittest.TestCase):
    def test_chunks_are_compacted_without_changing_returns(self):
        buffer = FlatRolloutBuffer("cpu")
        for chunk in range(2):
            for index in range(3):
                terminal = index == 2
                buffer.add(
                    flat_obs=np.full(5, chunk * 3 + index, dtype=np.float32),
                    legal_actions=np.ones(4, dtype=np.float32),
                    action=index % 4, log_prob=torch.tensor(-0.5),
                    reward=1.0 if terminal else 0.0,
                    value=torch.tensor(0.25), done=terminal,
                    all_hands=np.zeros((4, 52), dtype=np.float32),
                )
            buffer.pack_pending()
            self.assertEqual(len(buffer.actions), 0)
        self.assertEqual(len(buffer), 6)
        buffer.compute_returns_and_advantages(0.0, gamma=0.99, gae_lambda=0.95)
        batches = list(buffer.get_batches(4))
        self.assertEqual(sum(len(batch["actions"]) for batch in batches), 6)
        self.assertTrue(torch.isfinite(buffer.returns).all())

    def test_direct_step_batch_has_expected_dense_shapes(self):
        buffer = FlatRolloutBuffer("cpu")
        steps = []
        for index in range(7):
            steps.append({
                "flat_obs": np.full(5, index, dtype=np.float32),
                "legal_actions": np.ones(4, dtype=np.float32),
                "action": index % 4,
                "log_prob": torch.tensor(-0.5),
                "reward": float(index == 6),
                "value": torch.tensor(0.25),
                "done": index == 6,
                "all_hands": np.zeros((4, 52), dtype=np.float32),
            })
        buffer.add_steps(steps)
        self.assertEqual(len(buffer), 7)
        data = buffer._materialize()
        self.assertEqual(tuple(data["flat_obs"].shape), (7, 5))
        self.assertEqual(tuple(data["all_hands"].shape), (7, 4, 52))


if __name__ == "__main__":
    unittest.main()
