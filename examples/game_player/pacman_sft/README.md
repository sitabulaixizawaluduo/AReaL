# PlayJev Pacman soft-label SFT on AReaL

第一阶段：迁移 PlayJev 的 **Pacman teacher 采集、选项随机化、四选项软标签 SFT**。 复用 AReaL
`SFTTrainer → MegatronLMEngine`，不需要启动 SGLang。 数据与 prompt 直接复用指定的 PlayJev checkout；没有
DAgger 或通用数据混训。

## 训练链路

```text
PlayJev 游戏内部 state → 原 Pacman teacher → 四个动作的 teacher_probs
游戏截图 + 四个选项 → PlayJev SFTDataset 随机重排选项和对应概率
                    → 原 build_plain_prompt → VLM
                    → A/B/C/D 对应 token logits → 四选项 softmax → soft CE
```

模型只接收单帧截图、原任务指令和随机顺序的选项描述。内部坐标、teacher_action、 taken_action、游戏分数和 teacher_probs 都不进入
prompt。监督是完整的四项软标签， 不是 argmax 的硬标签，也没有 game-score reward。

若原动作顺序是 `[up, down, left, right]`，teacher 分布是 `[0.70, 0.10, 0.15, 0.05]`， 本次排列是
`[right, up, left, down]`，那么 A/B/C/D 的 target 就是 `[0.05, 0.70, 0.15, 0.10]`。因此 teacher
偏向 up 不等于 label 总偏向 A。 保留 teacher 的真实偏好，不人为把 q 拉平。

`p = softmax(logits[A, B, C, D])`，每个 decision 的 `loss = -Σ q[i] log p[i]`。
`candidate_id_0..3` 是 tokenizer 编码 `" A"`、`" B"`、`" C"`、`" D"` 的单 token ID。
`candidate_target_0..3` 保存重排后的 q，`candidate_count` 在监督位置为 4。 adapter 在 prompt 后追加一个 A
token 作为对齐载体；loss 读取它前一位置的 logits， 载体的 token 身份不作为硬标签。每张截图产生一条训练样本。

## 环境

在 CUDA 机器准备 AReaL 的 Megatron 运行环境，以及 PlayJev 的原有依赖和 Chromium。 本补丁没有增加 AReaL 的依赖。采集脚本复用
PlayJev 的 `python -m playjev.collect`， 需要该 Python 环境能导入 PlayJev 的依赖并启动 Playwright。

```bash
export PLAYJEV_ROOT=/path/to/PlayJev
export PLAYJEV_DATA_ROOT="$PLAYJEV_ROOT/data"
export MODEL_PATH=/path/to/Qwen3.5-0.8B-Base
export AREAL_OUTPUT_ROOT=/path/to/outputs
# 可选：指定已配置好的 Python 环境；默认 python3。
export PYTHON_BIN=python3
```

`MODEL_PATH` 建议使用与原 PlayJev 一致的 Qwen3.5-0.8B-Base 本地 checkpoint； 需要当前 Megatron Bridge
支持该 VLM。其他模型必须满足同一图像 placeholder 与四个 space-prefixed letter 单 token 的要求，不能直接假设所有 VLM 通用。

## 收集数据

在 AReaL worktree 根目录执行：

```bash
SHARD=pacman-sft-s0 bash examples/game_player/pacman_sft/collect_data.sh
```

默认采集 100,000 个 decision、8 个页面、epsilon=0.1、seed0=1000、每局最多 3000 步。 截图来自原采集器；原生 Pacman
每个动作执行 6 个游戏帧，单帧 JPEG 默认长边 448。 输出在
`$PLAYJEV_ROOT/data/pacman/$SHARD/records.jsonl`，frame 路径相对于 shard 目录。 脚本拒绝覆盖现有
shard。原采集器按 pages 批量推进，实际条数可能略超过 steps。

```bash
SHARD=pacman-small COLLECT_STEPS=1000 COLLECT_PAGES=8 \
  bash examples/game_player/pacman_sft/collect_data.sh
```

可通过 `COLLECT_EPSILON`、`COLLECT_SEED0`、`COLLECT_MAX_STEPS` 调整对应采集参数。 训练读取所有 shard；如需只读某个
shard，用后面的 `dataset_kwargs.shards` override。

原始记录保留 teacher_probs 等采集字段；adapter 仅将其放入监督通道。直接读取这些 JSONL 和 JPEG，不必先 export 成重复的 prompt
数据。按原 PlayJev 的 seed 划分： `seed % 10 == 0` 是验证集，其余是训练集；两个 split 都必须非空。

## 训练 recipe 与入口

在当前 Workspace 布局下，可直接使用启动脚本：

```bash
bash examples/game_player/pacman_sft/run_train.sh
```

`run_train.sh` 顶部的 `MODEL_PATH`、`PLAYJEV_DATA_ROOT` 是主要需要修改的两项。 默认分别指向 Workspace 下的
`Data/Qwen__Qwen3.5-2B` 和刚采集数据所在的 `Data/playjev`；数据路径要指向包含 `pacman/` 的目录，而不是某个 shard 或
JSONL 文件。 默认模型是本地已下载的 2B checkpoint，与原 PlayJev 的 0.8B-Base 不同。 脚本自动定位 AReaL、相邻的 PlayJev
checkout，并生成新的 trial 名。 在已有 CUDA/Megatron Python 环境中运行；也可通过 `PYTHON_BIN` 指定解释器。

```bash
MODEL_PATH=/path/to/model PLAYJEV_DATA_ROOT=/path/to/data \
  bash examples/game_player/pacman_sft/run_train.sh

# 额外参数直接透传给已有 recipe。
bash examples/game_player/pacman_sft/run_train.sh total_train_epochs=2
```

启动脚本已做 Shell 语法检查，未在本地启动 GPU 训练。 若希望直接调用入口，使用：

```bash
"$PYTHON_BIN" -m areal.infra.launcher.local \
  examples/game_player/pacman_sft/train.py \
  --config examples/game_player/pacman_sft/megatron.yaml
```

`train.py` 直接构建两个 dataset 并调用 `SFTTrainer.train()`，沿用已有配置。 PlayJev 专用数据参数使用已有的
`dataset_kwargs`，没有修改 `cli_args.py` 或 launcher。 默认配置针对当前 Qwen3.5-2B、单机 8 张 H200，使用
`megatron:d8p1t1`（DP=8，TP=PP=CP=1）。每卡保留完整模型，分片 optimizer； 对这个小模型优先用数据并行扩大
batch，减少模型并行通信。以下是吞吐优先的起始配置，最大吞吐与显存占用仍需在目标机器实测。

| 参数         | Recipe                                                   |
| ------------ | -------------------------------------------------------- |
| 输入         | 单帧，原 plain prompt，四个方向全部保留并随机排列        |
| 标签         | 原 teacher_probs，保持原采集器的四位小数                 |
| 划分         | 原 PlayJev 的 seed 隔离 train/validation                 |
| 更新         | 全参数 SFT，一轮，global batch 512，每个 DP rank 64 条   |
| 优化器       | Adam，lr 2e-5，betas 0.9/0.95，eps 1e-8，weight decay 0  |
| 调度         | cosine，warmup 3%，gradient clipping 1.0                 |
| 精度         | BF16，关闭 activation recompute，候选 softmax/CE 用 FP32 |
| Token budget | 单样本上限 4096；每卡 microbatch 总 token 上限 32768     |

`max_tokens_per_mb` 是每个 DP rank 单次 forward 的总 token 预算，包含图像 token， 不是样本数量；packing 可能把每卡
64 条拆成多个 microbatch。`n_mbs=1` 不强制梯度累积。 本地用当前 2B processor 检查了 32 条采集样本，长度都是 269 tokens；
按这一长度，每卡 64 条约 17,216 tokens，packing 会分配为一个 microbatch。 这只是输入长度检查，没有执行 GPU forward
或吞吐测试。 当前实现仍计算完整词表 LM head，再提取四项 logits；按 248,320 词表和 FP32 输出估算， 17,216 tokens 的单份
logits 约 15.9 GiB，且还需考虑 logits 副本、反向梯度和模型激活。 四项 softmax 不代表仅有四项输出的显存成本，因此显存不能只按 2B 权重估算。

开启 distributed optimizer、gradient reduce/parameter gather overlap，关闭 offload。 Megatron
Bridge 会读取 `recompute_*`，因此同时将这些字段设为 `null`，确保全层重计算关闭。 每个 DP rank 使用 4 个数据加载 worker，共 32
个；若 CPU/JPEG 处理成为瓶颈，再按实际 CPU 资源调整。 保持原始图片处理与 prompt，不通过降低分辨率改变监督任务。

当前本地采集的训练 split 是 7,597 条。默认 `drop_last=true`，global batch 从 64 改到 512 后， 每轮更新次数从 118 降至
14，并丢弃尾部 429 条；这是吞吐优先带来的优化行为变化。 学习率保持 2e-5，不按 batch 比例放大。需要更多更新时增加数据量或训练轮数。

在 H200 上比较稳定阶段的 samples/s、有效 tokens/s 和峰值显存；排除模型加载、首步、验证和 checkpoint 时间。 更大 batch
不保证更快。可通过已有启动脚本比较下面两档；同一预算下，batch 变大可能只增加梯度累积次数：

```bash
# 较小的每卡 microbatch：32 条，约 8,608 tokens（按上述样本长度）。
bash examples/game_player/pacman_sft/run_train.sh \
  train_dataset.batch_size=256 valid_dataset.batch_size=256 \
  actor.mb_spec.max_tokens_per_mb=16384

# 更大的每卡 microbatch：128 条，约 34,432 tokens，需要实测显存与吞吐。
bash examples/game_player/pacman_sft/run_train.sh \
  train_dataset.batch_size=1024 valid_dataset.batch_size=1024 \
  actor.mb_spec.max_tokens_per_mb=65536
```

若默认档 OOM，先将 `actor.mb_spec.max_tokens_per_mb=16384`，保持 global batch 512； 这会拆出更多
microbatch。仍不够时开启重计算：

```bash
bash examples/game_player/pacman_sft/run_train.sh \
  actor.mb_spec.max_tokens_per_mb=16384 actor.gradient_checkpointing=true \
  actor.megatron.recompute_granularity=full actor.megatron.recompute_method=uniform \
  actor.megatron.recompute_num_layers=1
```

TP/PP/DP 可通过现有 allocation 参数调整，例如双卡 TP：

```bash
"$PYTHON_BIN" -m areal.infra.launcher.local \
  examples/game_player/pacman_sft/train.py \
  --config examples/game_player/pacman_sft/megatron.yaml \
  allocation_mode=megatron:d1p1t2 cluster.n_gpus_per_node=2 \
  '+train_dataset.dataset_kwargs.shards=[pacman-sft-s0]' \
  '+valid_dataset.dataset_kwargs.shards=[pacman-sft-s0]'
```

首版要求 CP=1、temperature=1，关闭 tree training、MTP training 与 chunked LM head。 使用 Megatron 现有
LM head，然后仅对四项归一化；没有全词表 CE。 TP 仅通信四个候选 logits，梯度回到拥有相应词表分片的 rank。 其他普通 SFT 数据仍走原来的 token
loss。

复用的是原采集、排列、prompt、标签及目标函数。原 PlayJev 在 final hidden state 上做 FP32 四行投影，而这里沿用 Megatron LM
head；数值精度、batch packing、调度实现和 worker 随机种子与原单进程训练可能不同，不能据此声称训练结果逐位一致。 选项排列沿用 upstream
`SFTDataset` 的实现：训练依赖 `torch.initial_seed()+index`， 同一 worker seed 下重复访问同一 index
的排列会相同；验证由 seed/index 固定。

日志记录 `sft/soft_ce`、`sft/candidate_entropy`，验证记录对应的 `sft-eval` 指标。 检查 loss
有限、候选熵在合理范围，并与未训练模型的验证 CE 比较。 checkpoint 使用 AReaL saver 的 Hugging Face 导出格式，写入
`AREAL_OUTPUT_ROOT`。 CE 下降说明 teacher imitation 改善，游戏成绩需要另行闭环评估。

## 短局末尾样本加权（boost-last）

通过已有 `dataset_kwargs` 开启 PlayJev 的 `boost_last`，默认不重复样本：

```bash
bash examples/game_player/pacman_sft/run_train.sh \
  '+train_dataset.dataset_kwargs.boost_last=[10,4]'
```

也可在 recipe 的 `train_dataset.dataset_kwargs` 下添加 `boost_last: [10, 4]`。 参数必须是两个正整数
`[last_k, times]`，`times=1` 表示不增加副本。 加载器直接复用上游 `load_records(boost_last=(10, 4))`：按每个
shard 内的 `(seed, episode)` 分组，对记录的 `max(step)+1 < 500` 的训练局，最后 10 条 记录各补 3 个副本，总共出现 4
次；`seed % 10 == 0` 的验证样本不重复。 这是对现有截图与 teacher 软标签的重采样，随后仍由上游随机排列选项，再计算四项 soft CE。
短局可能是死亡、成功或采集截断，不能将其全部视为死亡局。

该参数作用于所有选中的 shard，包括 cloning 和 DAgger shard。请通过 `dataset_kwargs.shards`
显式选择本轮混训数据；如果同时设置 `limit`，它在重复和 train/validation 划分后截取样本，可能截掉追加的副本。上游在列表末尾追加副本，因此不要 用很小的
`limit` 来衡量实际加权效果。重复后每轮样本量和更新次数可能增加。

以下最小和端到端验证命令**未在本地运行**；预期完成训练、验证与 checkpoint 导出， 日志中的候选 soft CE 有限，验证集样本量保持不变：

```bash
# 单卡最小验证；关闭默认的训练 limit，让追加的副本保留在数据集中。
bash examples/game_player/pacman_sft/run_minimal.sh \
  '+train_dataset.dataset_kwargs.boost_last=[10,4]' \
  train_dataset.dataset_kwargs.limit=0

# 新建 teacher shard → 按同一规则重复末尾样本 → SFT → 验证 → HF checkpoint。
COLLECT_STEPS=1000 bash examples/game_player/pacman_sft/run_e2e.sh \
  '+train_dataset.dataset_kwargs.boost_last=[10,4]' \
  train_dataset.batch_size=8 valid_dataset.batch_size=8
```

## 验证脚本

下面的 UT、GPU smoke 和端到端脚本均**未在本地运行**。

```bash
# 已有包含 train/validation 的采集数据：单卡、两次 optimizer update。
bash examples/game_player/pacman_sft/run_minimal.sh

# 新建 shard → teacher 采集 → 单轮 SFT → 验证 → HF checkpoint。
bash examples/game_player/pacman_sft/run_e2e.sh

# 小批次端到端试跑；采集仍必须包含两个 split。
COLLECT_STEPS=1000 bash examples/game_player/pacman_sft/run_e2e.sh \
  train_dataset.batch_size=8 valid_dataset.batch_size=8

# 用户或 CI 执行单元测试；two-GPU TP 测试在没有两卡时跳过。
pytest tests/test_candidate_soft_sft.py tests/test_playjev_dataset.py
```

最小脚本预期完成两次更新和验证，产生有限的 soft CE/entropy 与 checkpoint。 端到端脚本预期产生新的 JSONL/JPEG
shard、完成训练和验证并导出 checkpoint； 采集或训练失败会以非零状态退出。它验证训练链路，不包含模型实际打游戏的 benchmark。
