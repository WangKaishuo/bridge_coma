#!/usr/bin/env python3
"""
DDS Data - Generation and Loading
==================================

预生成 DDS 训练数据，训练时直接加载使用。

存储格式（紧凑）：
    - decks: uint8 (N, 52)，deck[card] = 持有该牌的玩家 (0-3)
    - tricks: int8 (N, 5, 4)，tricks[suit, player] = DD 墩数
    
    每样本仅 52 + 20 = 72 bytes（对比原 float32 one-hot 的 832 bytes）

生成数据:
    python -m utils.dds_data --num_samples 1000000 --num_workers 8
    python -m utils.dds_data --num_samples 1000000 --seed_offset 1000000  # 第二批
    python -m utils.dds_data --resume
    python -m utils.dds_data --analyze data/

加载数据:
    from utils.dds_data import create_loader
    loader = create_loader('data/')
    hands, tricks = loader.sample(batch_size=256)  # hands: float32 (B,4,52)
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
from datetime import timedelta
from typing import Tuple, List
import os

import numpy as np
from tqdm import tqdm


# ==============================================================================
# Data Conversion
# ==============================================================================

def deck_to_hands(deck: np.ndarray) -> np.ndarray:
    """
    紧凑存储 -> one-hot
    
    Args:
        deck: (52,) 或 (N, 52) uint8
    Returns:
        hands: (4, 52) 或 (N, 4, 52) float32
    """
    if deck.ndim == 1:
        hands = np.zeros((4, 52), dtype=np.float32)
        hands[deck.astype(np.intp), np.arange(52)] = 1.0
        return hands
    else:
        batch_size = deck.shape[0]
        hands = np.zeros((batch_size, 4, 52), dtype=np.float32)
        hands[
            np.arange(batch_size)[:, None],
            deck.astype(np.intp),
            np.arange(52)[None, :],
        ] = 1.0
        return hands


# ==============================================================================
# Data Loader
# ==============================================================================

class DDSDataLoader:
    """单文件加载器 (Memory-mapped)"""
    
    def __init__(self, data_path: str, preload: bool = False):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"DDS data not found: {data_path}")
        
        mmap_mode = None if preload else 'r'
        data = np.load(self.data_path, mmap_mode=mmap_mode)
        
        self.decks = data['decks']
        self.tricks = data['tricks']
        self.num_samples = len(self.decks)
        
        size_mb = os.path.getsize(self.data_path) / 1024 / 1024
        print(f"  {self.data_path.name}: {self.num_samples:,} samples, {size_mb:.1f} MB")
    
    def sample(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.num_samples, size=batch_size)
        return deck_to_hands(np.array(self.decks[indices])), np.array(self.tricks[indices])
    
    def sample_one(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.num_samples)
        return deck_to_hands(np.array(self.decks[idx])), np.array(self.tricks[idx])
    
    def __len__(self) -> int:
        return self.num_samples


class MultiFileLoader:
    """多文件加载器"""
    
    def __init__(self, data_paths: List[str], preload: bool = False):
        print(f"Loading {len(data_paths)} files...")
        mmap_mode = None if preload else 'r'
        
        self.files = []
        for p in data_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Not found: {p}")
            self.files.append(np.load(path, mmap_mode=mmap_mode))
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {path.name}: {len(self.files[-1]['decks']):,} samples, {size_mb:.1f} MB")
        
        self.sizes = [len(f['decks']) for f in self.files]
        self.cumsum = np.cumsum(self.sizes)
        self.num_samples = self.cumsum[-1]
        print(f"Total: {self.num_samples:,} samples")
    
    def _locate(self, idx: int) -> Tuple[int, int]:
        file_idx = np.searchsorted(self.cumsum, idx, side='right')
        local_idx = idx - (self.cumsum[file_idx - 1] if file_idx > 0 else 0)
        return file_idx, local_idx
    
    def sample(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.num_samples, size=batch_size)
        
        decks = np.empty((batch_size, 52), dtype=np.uint8)
        tricks = np.empty((batch_size, 5, 4), dtype=np.int8)
        
        for i, idx in enumerate(indices):
            fi, li = self._locate(idx)
            decks[i] = self.files[fi]['decks'][li]
            tricks[i] = self.files[fi]['tricks'][li]
        
        return deck_to_hands(decks), tricks
    
    def sample_one(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.num_samples)
        fi, li = self._locate(idx)
        return deck_to_hands(np.array(self.files[fi]['decks'][li])), np.array(self.files[fi]['tricks'][li])
    
    def __len__(self) -> int:
        return self.num_samples


class MemmapDDSLoader:
    """Single-file, zero-decompression DDS loader.

    ``dds.npy`` is a structured array with ``decks`` and ``tricks`` fields.
    Multiple training processes map the same file and share the operating
    system page cache instead of each holding or repeatedly decompressing a
    private copy.
    """

    def __init__(self, data_path: str):
        path = Path(data_path)
        if path.is_dir():
            path = path / "dds.npy"
        if not path.exists():
            raise FileNotFoundError(f"Memmap DDS data not found: {path}")
        self.data_path = path
        self.records = np.load(path, mmap_mode="r", allow_pickle=False)
        required = {"decks", "tricks"}
        names = set(self.records.dtype.names or ())
        if not required.issubset(names):
            raise ValueError(f"Expected structured fields {required}, got {names}")
        if self.records.dtype["decks"].shape != (52,):
            raise ValueError("Memmap decks field must have shape (52,)")
        if self.records.dtype["tricks"].shape != (5, 4):
            raise ValueError("Memmap tricks field must have shape (5, 4)")
        self.num_samples = len(self.records)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path.name}: {self.num_samples:,} samples, {size_mb:.1f} MB (memmap)")

    def sample(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.num_samples, size=batch_size)
        records = self.records[indices]
        decks = np.asarray(records["decks"])
        tricks = np.asarray(records["tricks"])
        return deck_to_hands(decks), tricks.copy()

    def sample_one(self) -> Tuple[np.ndarray, np.ndarray]:
        record = self.records[np.random.randint(0, self.num_samples)]
        return (
            deck_to_hands(np.asarray(record["decks"])),
            np.asarray(record["tricks"]).copy(),
        )

    def __len__(self) -> int:
        return self.num_samples

    def close(self) -> None:
        mmap = getattr(self.records, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_loader(data_path: str, preload: bool = False):
    """
    智能创建加载器
    - 文件 -> DDSDataLoader
    - 目录 -> MultiFileLoader
    """
    path = Path(data_path)
    
    if path.is_file() and path.suffix == ".npy":
        return MemmapDDSLoader(str(path))
    if path.is_file():
        return DDSDataLoader(str(path), preload)
    elif path.is_dir():
        if (path / "dds.npy").exists():
            return MemmapDDSLoader(str(path / "dds.npy"))
        files = sorted(path.glob("dds_*.npz"))
        if not files:
            raise FileNotFoundError(f"No dds_*.npz in {path}")
        if len(files) == 1:
            return DDSDataLoader(str(files[0]), preload)
        return MultiFileLoader([str(f) for f in files], preload)
    else:
        raise FileNotFoundError(f"Not found: {path}")


# ==============================================================================
# Data Generation
# ==============================================================================

def cards_to_pbn(cards: np.ndarray) -> str:
    """洗牌结果 -> PBN (无中间 one-hot)"""
    rank_chars = "23456789TJQKA"
    suit_order = [3, 2, 1, 0]  # SHDC
    
    hands = []
    for p in range(4):
        player_cards = cards[p * 13 : (p + 1) * 13]
        suits = [[] for _ in range(4)]
        for c in player_cards:
            suits[c // 13].append(rank_chars[c % 13])
        
        hand_str = '.'.join(
            ''.join(sorted(suits[s], key=lambda x: rank_chars.index(x), reverse=True))
            for s in suit_order
        )
        hands.append(hand_str)
    
    return f"N:{hands[0]} {hands[1]} {hands[2]} {hands[3]}"


def process_deal(seed: int):
    """生成一副牌 + DD table"""
    try:
        from endplay.dds import calc_dd_table
        from endplay.types import Deal
        
        np.random.seed(seed)
        
        cards = np.arange(52, dtype=np.uint8)
        np.random.shuffle(cards)
        
        # 紧凑存储
        deck = np.empty(52, dtype=np.uint8)
        for i, card in enumerate(cards):
            deck[card] = i // 13
        
        # DD table
        dd = calc_dd_table(Deal(cards_to_pbn(cards))).to_list()
        tricks = np.array([
            [dd[3][p] for p in range(4)],  # C
            [dd[2][p] for p in range(4)],  # D
            [dd[1][p] for p in range(4)],  # H
            [dd[0][p] for p in range(4)],  # S
            [dd[4][p] for p in range(4)],  # NT
        ], dtype=np.int8)
        
        return deck, tricks
    except:
        return None


def generate_batch(start_seed: int, count: int, num_workers: int):
    """生成一批"""
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_deal, range(start_seed, start_seed + count), chunksize=100),
            total=count, leave=False
        ))
    
    valid = [r for r in results if r is not None]
    if not valid:
        return None, None
    
    return np.stack([r[0] for r in valid]), np.stack([r[1] for r in valid])


class BatchManager:
    """批次管理"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
    
    def load_progress(self) -> dict:
        if self.progress_file.exists():
            return json.load(open(self.progress_file))
        return {'done': [], 'total': 0, 'seed_offset': 0}
    
    def save_progress(self, progress: dict):
        json.dump(progress, open(self.progress_file, 'w'))
    
    def save_batch(self, batch_id: int, decks: np.ndarray, tricks: np.ndarray):
        if decks is None:
            return None
        path = self.output_dir / f"dds_{batch_id:04d}.npz"
        np.savez_compressed(path, decks=decks, tricks=tricks)
        return path


def generate_all(num_samples: int, batch_size: int, num_workers: int,
                 output_dir: str, seed_offset: int = 0, resume: bool = False) -> int:
    """生成所有数据"""
    manager = BatchManager(output_dir)
    
    if resume:
        progress = manager.load_progress()
        seed_offset = progress.get('seed_offset', seed_offset)
    else:
        progress = {'done': [], 'total': 0, 'seed_offset': seed_offset}
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    done = set(progress['done'])
    total = progress['total']
    
    if resume and done:
        print(f"Resuming: {len(done)}/{num_batches} batches, {total:,} samples")
    
    t0 = time.time()
    
    for bid in range(num_batches):
        if bid in done:
            continue
        
        size = min(batch_size, num_samples - bid * batch_size)
        elapsed = time.time() - t0
        
        if total > 0 and elapsed > 0:
            eta = timedelta(seconds=int((num_samples - total) / (total / elapsed)))
        else:
            eta = "..."
        
        print(f"[{bid+1}/{num_batches}] {total:,}/{num_samples:,} ({100*total/num_samples:.1f}%) ETA: {eta}")
        
        decks, tricks = generate_batch(seed_offset + bid * batch_size, size, num_workers)
        path = manager.save_batch(bid, decks, tricks)
        
        if decks is not None:
            done.add(bid)
            total += len(decks)
            manager.save_progress({'done': list(done), 'total': total, 'seed_offset': seed_offset})
            print(f"  -> {path.name}: {len(decks):,} samples")
    
    return total


def analyze(data_path: str):
    """分析数据"""
    path = Path(data_path)
    files = sorted(path.glob("dds_*.npz")) if path.is_dir() else [path]
    
    if not files:
        print(f"No data found: {data_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"DDS Data: {data_path}")
    print(f"{'='*60}")
    
    total_samples = 0
    total_size = 0
    all_tricks = []
    
    for f in files:
        d = np.load(f)
        n, size = len(d['decks']), os.path.getsize(f)
        total_samples += n
        total_size += size
        all_tricks.append(d['tricks'])
        print(f"  {f.name}: {n:,} samples, {size/1024/1024:.1f} MB")
    
    print(f"\nTotal: {total_samples:,} samples, {total_size/1024/1024:.1f} MB")
    print(f"Storage: {total_size/total_samples:.1f} bytes/sample")
    
    tricks = np.concatenate(all_tricks)
    print(f"\nMean DD tricks (NS):")
    for suit, name in enumerate(['♣', '♦', '♥', '♠', 'NT']):
        print(f"  {name}: {(tricks[:, suit, 0] + tricks[:, suit, 2]).mean() / 2:.2f}")


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def main():
    default_output = str(get_project_root() / 'data')
    
    p = argparse.ArgumentParser(description="Generate/analyze DDS data")
    p.add_argument('--num_samples', type=int, default=1000000,
                   help='Total samples to generate (default: 1000000)')
    p.add_argument('--batch_size', type=int, default=100000,
                   help='Samples per file (default: 100000)')
    p.add_argument('--num_workers', type=int, default=8,
                   help='Parallel workers (default: 8)')
    p.add_argument('--output_dir', type=str, default=default_output,
                   help=f'Output directory (default: {default_output})')
    p.add_argument('--seed_offset', type=int, default=0,
                   help='Seed offset to avoid duplicates')
    p.add_argument('--resume', action='store_true',
                   help='Resume interrupted generation')
    p.add_argument('--analyze', type=str, metavar='PATH',
                   help='Analyze data (file or directory)')
    args = p.parse_args()
    
    if args.analyze:
        analyze(args.analyze)
        return
    
    try:
        from endplay.dds import calc_dd_table
        print("✓ endplay")
    except ImportError:
        print("✗ pip install endplay")
        return
    
    print(f"\n{'='*60}")
    print(f"Samples: {args.num_samples:,}, Batch: {args.batch_size:,}")
    print(f"Workers: {args.num_workers}, Seed offset: {args.seed_offset:,}")
    print(f"Output: {args.output_dir}/")
    print(f"{'='*60}\n")
    
    t0 = time.time()
    total = generate_all(args.num_samples, args.batch_size, args.num_workers,
                         args.output_dir, args.seed_offset, args.resume)
    
    print(f"\n{'='*60}")
    print(f"Done! {total:,} in {(time.time()-t0)/3600:.2f}h ({total/(time.time()-t0):.0f}/s)")
    print(f"{'='*60}")
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
