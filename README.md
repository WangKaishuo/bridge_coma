# Bridge-COMA

**Dual-Information Credit Assignment for Cooperative-Competitive Multi-Agent Coordination**

MSc Research Project — 在合作-对抗混合博弈（桥牌叫牌）中，利用先验不对称性进行信息论 reward shaping。

## 项目进度

### Phase 1: 环境与基础设施 ✅ 完成

| 工作项 | 状态 |
|--------|------|
| 包结构 + `setup_project.py` 自动组装 | ✅ |
| 单桌叫牌环境 (`BridgeBiddingEnv`) | ✅ |
| 双桌 IMP 环境 (`DualTableEnv`) | ✅ |
| 得分计算 SSOT (`scoring.py`) | ✅ |
| IMP 转换 (`imp.py`) | ✅ |
| DDS 数据生成与加载（100 万条已生成） | ✅ |
| IPPO / MAPPO 算法 | ✅ |
| 训练脚本（双桌 IMP reward + dealer 轮转 + vulnerability 随机） | ✅ |
| 测试套件（35 项测试覆盖全模块） | ✅ |

### Phase 2: 子博弈验证 — 待开始

### Phase 3: Belief + DualInfo — 待开始

### Phase 4: 完整训练与实验 — 待开始

## 项目结构

源文件在 Projects files 中以扁平方式存放。运行 `setup_project.py` 自动组装为以下包结构：

```
bridge-coma/
├── env/                          # 环境
│   ├── bridge_bidding_env.py     # 单桌叫牌环境 (Dec-POMDP)
│   └── dual_table_env.py         # 双桌 IMP 环境（训练 + 评估）
├── networks/                     # 神经网络
│   ├── policy_net.py             # PolicyNetwork, ValueNetwork, ActorCritic
│   └── belief_net.py             # BeliefNetwork, DualInfoComputer
├── utils/                        # 工具
│   ├── scoring.py                # 得分计算 (SSOT)
│   ├── imp.py                    # IMP 转换
│   ├── dds_data.py               # DDS 数据生成/加载
│   └── running_stats.py          # Welford / EMA 在线统计
├── algorithms/                   # 算法
│   ├── ippo.py                   # Independent PPO
│   └── mappo.py                  # Multi-Agent PPO (CTDE)
├── experiments/
│   └── train.py                  # 训练入口
├── tests/
│   └── test_all.py               # 测试套件 (35 tests)
├── data/                         # DDS 生成数据 (.npz)
├── checkpoints/                  # 模型存档
├── setup_project.py              # 扁平文件 → 包结构组装
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
pip install torch numpy tqdm endplay pyyaml
```

### 2. 组装项目（每次新对话/新环境必做）

```bash
python setup_project.py
```

### 3. 生成 DDS 数据

```bash
cd bridge-coma/

# 生成 10 万条（约 15-40 分钟，取决于 CPU 核数）
python -m utils.dds_data --num_samples 100000 --num_workers 4

# 追加生成（用 seed_offset 避免重复）
python -m utils.dds_data --num_samples 100000 --seed_offset 100000

# 分析已有数据
python -m utils.dds_data --analyze data/
```

### 4. 训练

```bash
cd bridge-coma/

# MAPPO（默认，推荐）
python experiments/train.py --algorithm mappo --data_path data/

# IPPO
python experiments/train.py --algorithm ippo --data_path data/

# 完整参数
python experiments/train.py \
    --algorithm mappo \
    --data_path data/ \
    --num_iterations 2500 \
    --deals_per_collect 4 \
    --eval_interval 50 \
    --save_interval 200 \
    --device cuda
```

训练参数说明：
- `--deals_per_collect 4`：每次采样 4 副牌 × 4 dealer = 16 episodes，然后做一次 PPO update
- `--no_rotate`：关闭 dealer 轮转（一副牌只用 1 次而非 4 次）
- `--device cuda`：使用 GPU（默认自动检测）

### 5. 测试

```bash
cd bridge-coma/

# 运行全部测试
python tests/test_all.py

# 跳过 torch 相关测试（快速验证环境逻辑）
python tests/test_all.py --no-torch
```

### Colab 使用

```python
# Cell 1: 挂载 Drive 并安装依赖
from google.colab import drive
drive.mount('/content/drive')
!pip install torch numpy tqdm pyyaml endplay

# Cell 2: 组装项目
%cd /content/drive/MyDrive/bridge-coma
!python setup_project.py

# Cell 3: 训练（checkpoint 保存到 Drive 防断连）
%cd /home/claude/bridge-coma
!python experiments/train.py \
    --algorithm mappo \
    --data_path data/ \
    --checkpoint_dir /content/drive/MyDrive/bridge-coma/checkpoints
```

## 训练机制

### 双桌 IMP Reward

每个训练 episode 打双桌：同一副牌在正常位置和互换位置（N↔E, S↔W）各叫一次，用分差转 IMP 作为终局 reward。NS 玩家得 +IMP，EW 玩家得 -IMP。

### Dealer 轮转

一副牌使用 4 次（dealer = N/E/S/W），4× 数据利用率。同一副牌的 4 次使用分散在不同 batch 中。

### Vulnerability 随机化

每副牌随机从 4 种局况（双无/NS有局/EW有局/双有）中采样。

## DDS 数据

### 存储格式（紧凑）
- `decks`: uint8 (N, 52)，每张牌由哪个玩家持有 (0-3)
- `tricks`: int8 (N, 5, 4)，DD 墩数 [suit, player]
- 每样本 72 bytes

### 内存映射
使用 `mmap_mode='r'` 加载，百万级数据不会撑爆内存。

### 多文件支持
```python
from utils.dds_data import create_loader

loader = create_loader('data/')  # 自动加载目录下所有 dds_*.npz
hands, tricks = loader.sample(batch_size=256)
```

## 测试覆盖

| 测试组 | 数量 | 内容 |
|--------|------|------|
| 包导入 | 2 | env, utils 所有导出 |
| 得分计算 | 6 | 基本分、成局、满贯、加倍、再加倍、宕墩 |
| IMP 转换 | 2 | IMP 表、VP 转换 |
| 叫牌环境 | 10 | obs shape、结束条件、合法动作、庄家判定、bid 转换 |
| 双桌环境 | 7 | swap、play_deal、reward 分配、dealer 轮转、vulnerability |
| DDS 数据 | 4 | 加载、采样、deck→hands |
| 运行统计 | 2 | Welford、EMA |
| 网络 | 7 | Policy/Value/ActorCritic/Belief 前向传播与 loss |
| Agent | 4 | IPPO/MAPPO 采样 + store + update |
| 端到端 | 2 | 随机策略多副牌无崩溃 |

## 修复记录

- **P0 包结构**: `setup_project.py` 自动从扁平文件组装正确的包层级
- **P1 结束条件**: 四家全 Pass 流局 / 有实质叫品后三家 Pass
- **P2 Reward 分配**: NS 阵营 +IMP, EW 阵营 -IMP（修复了旧版所有玩家相同 reward 的 bug）
- **P3 评估逻辑**: 使用双桌 IMP 评估（修复了旧版只用原始分的问题）
- **P4 得分计算**: 统一到 `utils/scoring.py` (SSOT)，修复 `bridge_bidding_env.py` 中的延迟 import
- **P5 Vulnerability**: 随机化（4 种局况组合）
- **P6 Dealer 轮转**: 一副牌 ×4 dealer
- **内存优化**: Memory-mapped 加载
- **多文件支持**: MultiFileLoader + create_loader
