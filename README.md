# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**

## 项目结构

```
bridge-coma/
├── env/                     # 环境
│   ├── bridge_bidding_env.py
│   └── dual_table_env.py
├── networks/                # 神经网络
│   ├── policy_net.py
│   └── belief_net.py
├── utils/                   # 工具
│   ├── scoring.py           # 得分计算 (SSOT)
│   ├── imp.py               # IMP 转换
│   ├── dds_data.py          # DDS 数据生成/加载
│   └── running_stats.py
├── algorithms/              # 算法
│   ├── ippo.py
│   └── mappo.py
├── experiments/
│   └── train.py
└── data/                    # DDS 数据
```

## 快速开始

```bash
# 安装
pip install torch numpy tqdm endplay

# 生成 DDS 数据（输出到 data/ 目录）
python -m utils.dds_data --num_samples 100000 --num_workers 8

# 生成更多数据（使用 seed_offset 避免重复）
python -m utils.dds_data --num_samples 100000 --seed_offset 100000 --output_name dds_data_1.npz

# 训练（支持单文件或目录）
python experiments/train.py --algorithm mappo --data_path data/dds_data.npz
python experiments/train.py --algorithm mappo --data_path data/  # 加载目录下所有 .npz

# 测试
python test_all.py

# 分析数据
python -m utils.dds_data --analyze data/dds_data.npz
```

## DDS 数据说明

### 内存优化
- 使用 `mmap_mode='r'` 实现内存映射，千万级数据不会撑爆内存
- OS 自动管理缓存，热点数据留在内存

### 多文件支持
```python
from utils.dds_data import DDSDataLoader, MultiFileLoader, create_loader

# 单文件
loader = DDSDataLoader('data/dds_data.npz')

# 多文件
loader = MultiFileLoader(['data/part_0.npz', 'data/part_1.npz'])

# 智能加载（推荐）
loader = create_loader('data/')  # 自动加载目录下所有 .npz

hands, tricks = loader.sample(batch_size=256)
```

### 避免数据重复
```bash
# 第一批：seed 0-99999
python -m utils.dds_data --num_samples 100000 --seed_offset 0

# 第二批：seed 100000-199999
python -m utils.dds_data --num_samples 100000 --seed_offset 100000 --output_name dds_data_1.npz
```

## 修复记录

- **P0**: 结束条件 Bug - 四家全Pass流局 / 有实质叫品后三家Pass
- **P1**: 得分计算统一到 `utils/scoring.py`
- **P2**: PBN 转换统一到 `utils/dds_data.py`
- **内存优化**: Memory-mapped 加载，支持大规模数据集
- **多文件支持**: MultiFileLoader + create_loader
