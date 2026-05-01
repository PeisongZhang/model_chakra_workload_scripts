# STG 配置参数与现有 workload 脚本梳理

本文件总结 `symbolic_tensor_graph/main.py` 暴露的全部命令行参数、参数之间的约束，
以及 `llama3_8b / llama3_70b / megatron_gpt_39b / megatron_gpt_76b / qwen_32b / qwen_35b`
六份现有 workload 生成脚本的缺陷。所有路径相对仓库根 `dnn_workload/`。

---

## 一、STG 全部命令行参数（`symbolic_tensor_graph/main.py:175-310`）

### 1. 输出
| 参数 | 含义 |
|---|---|
| `--output_dir` | 输出 `.et` / `workload.json` 的目录（必填） |
| `--output_name` | 文件模板，必须含 `%d` 表示 rank（必填） |
| `--chakra_schema_version` | 仅保留 `v0.0.4`，readout backend 已写死 |

### 2. 并行度（总 NPU 数 = `dp · tp · pp · sp · ep`）
| 参数 | 含义 |
|---|---|
| `--dp` | data-parallel 度 |
| `--tp` | tensor-parallel 度（行/列分片 GEMM） |
| `--pp` | pipeline-parallel 度（按 transformer block 切阶段） |
| `--sp` | context/sequence parallel 度（在 STG 内部用 `cp` 符号） |
| `--ep` | expert parallel 度（仅 MoE 用；dense/gpt 路径在 `main.py:450/541` 把 `tp *= ep` 折叠回 TP 通信组） |
| `--weight_sharded` | True 时把 `fsdp` 符号映射成 `dp`（FSDP/ZeRO-3 全分片）；否则置 1 |

### 3. 模型形状
| 参数 | 含义 |
|---|---|
| `--dvocal` | 词表大小 |
| `--dmodel` | hidden size |
| `--dff` | FFN 中间维度（dense FFN 或 MoE 单 expert 的 intermediate） |
| `--head` / `--kvhead` | Q-head 数 / KV-head 数（GQA：`kvhead < head`） |
| `--num_stacks` | transformer block 数（layer 数） |
| `--seq` | sequence 长度 |
| `--experts` / `--kexperts` | MoE 总专家数 / top-k 路由数 |
| `--model_type` | `llama` / `dense` / `gpt` / `moe` / `debug` |

### 4. Batch 与训练循环
| 参数 | 含义 |
|---|---|
| `--batch` | **全局** batch size（跨所有 DP rank） |
| `--micro_batch` | **每 rank** 的 micro-batch（Megatron 习惯）；微批数 = `batch/(micro_batch·dp)`；默认 -1 表示无累积 |
| `--num_iterations` | 在同一 trace 内连续生成的 iteration 数 |
| `--dp_local_sgd_interval` | 每 K 个 iteration 才插入 DP all-reduce；K=1 即同步 DP，K>1 为 LocalSGD |

### 5. Pipeline 调度
| 参数 | 含义 |
|---|---|
| `--pipeline_schedule` | `natural` / `gpipe` / `1f1b` / `1f1b-interleaved`（`pipeline_schedule.py:397`） |
| `--pipeline_virtual_stages` | 每设备的虚拟段数 v；v>1 即 Megatron 交错调度 |

### 6. 显存 / 通信优化
| 参数 | 含义 |
|---|---|
| `--scatter_gather_optimization` | PP 跨段时 TP rank 只发 1/t 切片，对端 all-gather 还原（仅 tp>1 生效） |
| `--activation_recompute` | backward 抬升 1× forward FLOP；对应 Megatron §3.5 |
| `--mixed_precision` | 切到 `llama_model.py` mixprecision 路径；同时影响 pipeline tensor map 的实现分支 |

### 7. Attention 后端
| 参数 | 含义 |
|---|---|
| `--attention_backend` | `auto` / `standard` / `fused` / `flash` |
| `--flash_attention` | 旧布尔 flag；与 `attention_backend=auto` 兼容 |
| `--tpsp` | True 表示 TP+SP；MoE 路径强制要求 True（`main.py:596`） |

### 8. 报告
| 参数 | 含义 |
|---|---|
| `--print_gpu_vram` | 输出 per-GPU 显存（params/acts/grads） |

---

## 二、参数之间的约束

`main.py` 中显式断言或隐式要求：

1. **批量整除**（`main.py:362-373`）
   - `batch % (micro_batch · dp) == 0`
   - 默认 `micro_batch=-1` 时还要 `batch % dp == 0`

2. **Pipeline 划分**（`main.py:316-328`、`_create_pipeline_tensor_map`）
   - `num_stacks ≥ pipeline_virtual_stages · pp`
   - **v>1 必须**：`num_stacks % (v · pp) == 0`；v=1 允许余数

3. **Pipeline schedule**（`pipeline_schedule.py:397,426`）
   - schedule 必须在 `VALID_SCHEDULES` 内；非 `natural` 仅在 `pp>1` 时生效（`main.py:154`）

4. **Iteration / LocalSGD**（`main.py:312-315`）
   - `num_iterations ≥ 1`，`dp_local_sgd_interval ≥ 1`

5. **Attention 后端**（`main.py:329-337`）
   - `--flash_attention=true` 不能配 `--attention_backend standard|fused`

6. **GQA**
   - `head % kvhead == 0`，且 TP ≤ kvhead，否则跨 TP rank 复制 KV 头（脚本注释里强调，代码并未自动检查）

7. **EP 与 TP 混合（dense/gpt 路径）**（`main.py:450,541`）
   - 进 GraphDistributer 前 `tp ← tp · ep`，意味着 dense 路径**对 EP 实际是当成额外的 TP** 处理；只有 `model_type=moe` 才走真正的 expert-parallel 路径

8. **MoE 强制项**
   - `tpsp=True`（`main.py:596`）
   - `experts % ep == 0`（注释里要求；STG 内部按 `experts/ep` 拆分）

9. **MicroBatch 符号**
   - `MicroBatch_symbol = micro_batch · dp`，依赖 `MicroBatchReplicator` 后图里 `Batch/dp → micro_batch`
   - 走 `STAGE_MICROBATCH_OPTIMIZE=1` shortcut 时要小心 op_attr 字符串替换冲突（见下文）

---

## 三、现有 workload 脚本的缺陷

### 共性问题

1. **缺乏并行度合法性校验**。脚本只声明 `total = DP·TP·PP·SP`（`llama3_8b.sh:18` 等），但不检查
   - `head % TP == 0`、`kvhead ≥ TP`
   - `LAYER % (PP·PP_VIRTUAL) == 0`（`PP_VIRTUAL>1` 时 STG 直接报错，提前拦下来更友好）
   - `BATCH % (MICROBATCH·DP) == 0`
   - 错误目前要进入 `main.py` 才暴露，调试链路长。

2. **`OUTPUT_DIR` 命名维度不全**。当前命名只覆盖 `attention/sgd/layer/iter/batch/mb/seq/sched/v/sgo/ar`（70b/76b），**没有 dp/tp/pp/sp**。改并行度后会覆盖之前的产物（`llama3_70b.sh:46`、`megatron_gpt_*.sh`、`qwen_3?b.sh`）。这是当下最容易“静默踩坑”的缺陷。

3. **不同脚本默认值/可调项不一致**：
   - `llama3_8b.sh` 没有 `PP_SCHEDULE` / `PP_VIRTUAL` / `SGO` / `ACTIVATION_RECOMPUTE` 钩子（默认值由 `main.py` 决定，等于固化 `natural / v=1 / off / off`）。
   - `qwen_3?b.sh` 默认 `ATTENTION=standard`、`SGO=false`、`PP_SCHEDULE=natural`；而 `llama3_70b/megatron_gpt_*` 默认 `ATTENTION=fused`、`SGO=1`、`PP_SCHEDULE=1f1b`、`AR=1`。同一套基础设施跑出的报告口径不同。

4. **EP 仅在 `qwen_35b.sh` 暴露**。其他脚本没有 `--ep`，但 `main.py:450/541` 默认 `ep=1` 会被乘到 `tp`，行为等价；问题是 dense 路径把 EP 当 TP 用（`tp *= ep`），脚本没注释这一点，用户若误认为 EP 在 dense 模型里能开真 expert 并行会被坑。

5. **Python 解释器路径硬编码**。`llama3_70b.sh`、`megatron_gpt_*.sh` 默认 `PY=/home/ps/sow/part2/astra-sim/.venv/bin/python`（`llama3_70b.sh:12`），换机器/环境就会报错；`llama3_8b.sh`、`qwen_*.sh` 用的是 `python3`，行为不统一。

6. **`SGO` 在不同脚本里取值不同**：`llama3_70b/megatron_gpt_*` 用 `0/1` 拼路径再展开成 `--scatter_gather_optimization true`；`qwen_*` 直接传 `true/false` 字串（路径里也长成 `sgofalse`）。同一字段出现两套表示，分析脚本 / 后续 grep 容易踩。

7. **VRAM 报告未默认开启**。`--print_gpu_vram` 是诊断 OOM / FSDP 决策最便宜的工具，但所有驱动脚本都未暴露。

8. **`weight_sharded` 强制 0**。所有脚本都写死 `--weight_sharded 0`，导致 FSDP / ZeRO-3 这条 STG 已实现的路径根本没法通过驱动脚本跑。

### 单个脚本特有的缺陷

- **`llama3_8b/llama3_8b.sh`**
  - 没有 PP/AR/SGO/调度的环境变量入口（功能阉割版）。
  - 默认 `DP=4 TP=1 PP=4 SP=1`，`LAYER=32`、`PP=4` 对 `PP_VIRTUAL` 升级时直接报错：32 不被 v>1·4 整除（v=8 才行），脚本无提示。

- **`qwen_32b/qwen_32b.sh`**
  - `--dmodel 5210`：注释说取自 `qwen_32b_config.json`，但配置里 `hidden_size=5120`（`qwen_32b_config.json:9`）。**这是一处事实性 typo，会让 GEMM/通信量算错 ~1.7%**。
  - 默认 `LAYER=4`，真实模型 64 层；属于 "smoke 默认"，脚本没标注，跑出来的数据若被当成 32B 全模型结果会误导。
  - GQA：`head=40, kvhead=8`，默认 `TP=8` 刚好等于 kvhead；若用户改 `TP=16` 没有报错保护。

- **`qwen_35b/qwen_35b.sh`**
  - `head_dim` 失真：脚本注释承认（`qwen_35b.sh:67-71`、`docs/qwen_35b.md`），attention FLOP 被 ×½；网络/系统层实验可用，做算力 / TFLOPS 报告会偏低。
  - 没有暴露 `--shared_expert_intermediate_size`、`mtp_num_hidden_layers`、layer-mix（每 4 层 full_attention）等真实结构；当前一律按 dense transformer block 处理。
  - 默认 `EXPERTS=256, KEXPERTS=8, EP=8` → `experts/ep=32`，但脚本不校验 `EXPERTS % EP == 0`。

- **`llama3_70b/llama3_70b.sh`**
  - 注释写“对标 megatron_gpt_76b 的 t=8,p=4,d=32 → 1024 GPUs”，但 70B 真实层数 80、`MICROBATCH=2 BATCH=1792`，`num_micro_batches = 1792/(2·32)=28`，与 1f1b warmup steady 计算一致；但脚本没体现“这是 mixed-precision recipe”，与 llama3 真实训练栈口径有差距。
  - `kvhead=8`，TP 上限 8（注释里写明）但脚本不强制。

- **`megatron_gpt_39b/megatron_gpt_39b.sh`**
  - 用 `--dff 32768`，论文 `dff = 4·dmodel = 4·8192 = 32768` 是对的；不过这值并未来自任何 json，纯靠注释，缺一份跟脚本同目录的 `*.json` 让数据可追溯。
  - `head=64, kvhead=64`（MHA），TP=8 时每 rank 8 head 没问题；但脚本没有 GQA 注释提示读者它是 MHA。

- **`megatron_gpt_76b/megatron_gpt_76b.sh`**
  - 与 39b 同样的 “无 config json” 问题。
  - 目录里已有两份产物 `fused_standard_60_1_1792_2_2048_1f1b_v1_sgo1_ar1` 和 `..._ar0` —— 印证了 “OUTPUT_DIR 不含并行度” 的隐患：换并行度跑会直接覆盖。

- **MoE 路径的潜在 bug（已修但脆弱）**：`main.py` 的 `ReplicateGraph` 对 `op_attr` 做字符串替换，会把 `(MicroBatch)` 二次替换出 `Micro(MicroBatch)`。`docs/qwen_35b.md` §3 已记录修复，但根因（字符串替换而非 sympy 替换）尚未根治；`STAGE_MICROBATCH_OPTIMIZE=1` 路径仍是隐患。

---

## 四、可优先修的 3 件事

1. 把 `DP/TP/PP/SP/EP` 加入所有脚本的 `OUTPUT_DIR`，避免静默覆盖。
2. 在脚本里加 `LAYER % (PP·PP_VIRTUAL)`、`BATCH % (MICROBATCH·DP)`、`HEAD % TP` 三个前置校验。
3. 修 `qwen_32b.sh` 的 `--dmodel 5210 → 5120`，并补一份 `llama3_405b / megatron_gpt_*` 缺失的 `*_config.json` 让输入可追溯。

---

## 五、MoE 模型中 TP / EP / 总 Rank 的关系

### 5.1 坐标轴关系：TP 与 EP 是正交并行轴

`main.py:343` 一次性声明 5 个并行度符号：
```python
dp, tp, pp, spp, ep, fsdp = sp.symbols("dp tp pp cp ep fsdp")
```

走 `model_type=moe` 时（`main.py:593-628`）：

```python
spatial_parallel_dims_moe = [dp, tp, spp, ep]   # 空间维：DP, TP, SP(cp), EP
temporal_parallel_dims    = [pp]                # 时间维：PP
```

**总 rank 数 = `DP · TP · PP · SP · EP`**，每一维都吃掉一个 NPU 轴。

> ⚠️ MoE 路径独有。dense / gpt 路径在 `main.py:450,541` 会执行
> `symbol_map_value[tp] *= symbol_map_value[ep]`，把 EP 折叠回 TP。
> 也就是 dense 模型上传 `--ep` 只是当成"放大一倍 TP"，**不会真正出现 expert 并行**。

### 5.2 EP 的语义：专家分组

`moe_model.py:46-52`：

```python
experts_each_group = experts / ep
assert experts_each_group == int(experts_each_group)
```

- 全局 `Experts` 个专家 → 切成 EP 组，每组 `Experts/EP` 个本地专家。
- 每个 EP rank 在本地构建 `experts_each_group` 个 expert 子图（`moe_model.py:62-63`）。
- 路由权重 `wrouter` 沿 EP 切 `Dmodel`：`Dmodel/(ep*1), KExperts`（`moe_frame.csv:3`）。

### 5.3 TP 的语义：Hidden / FFN / Attention 内部分片

在 MoE expert 内部（`llama_feed_forward_network.csv`）：

| 张量 | 形状 | 说明 |
|---|---|---|
| `wgate` | `Dmodel, Dff/tp` | FFN 列并行 |
| `wdown` | `Dff/tp, Dmodel` | FFN 行并行 |
| `xdown` | `Batch/dp, (Seq/cp)/tp, Dff` | TP+SP（输出沿 Seq 切回 1/tp） |

加上 attention 部分仍是常规 GQA 的 `Head/tp`。

### 5.4 TP 与 EP 在路由后联动

`expert_wrapper.csv:2-3`：

```
x          : Batch/dp, (Seq/cp)*KExperts/(tp*ep),         Dmodel/1
x_expert   : Batch/dp, (Seq/cp)*KExperts/(Experts*tp*ep), Dmodel/1
```

路由后的 token 序列长度被 **TP × EP × Experts** 一起切：

- 进入 router 后 token 总数 per rank = `(Seq · KExperts)/(tp · ep)`
- 每个本地 expert 处理 token = `(Seq · KExperts)/(Experts · tp · ep)`

### 5.5 必须满足的整除约束

| 约束 | 来源 | 触发后果 |
|---|---|---|
| `Experts % EP == 0` | `moe_model.py:51` assert | 直接 AssertionError |
| `(Seq · KExperts) % (TP · EP) == 0` | `moe_frame.csv` | sympy 表达式无法整型化 |
| `(Seq · KExperts) % (Experts · TP · EP) == 0` | `expert_wrapper.csv` 的 `x_expert` 形状 | 本地 expert 拿不到完整 token 切片 |
| `Dmodel % EP == 0` | `wrouter` 形状 | 路由权重无法切分 |
| `Dff % TP == 0` | FFN | FFN 列/行并行无法切 |
| `Head % TP == 0` 且 `KVHead ≥ TP` | GQA | KV head 需被复制 |

### 5.6 取值折中

- 形状里 `tp · ep` 始终成对出现，**token 维度的总切分系数 = TP·EP**。
- TP 增大 → expert 内部 all-reduce / reduce-scatter 量上升。
- EP 增大 → 路由 all-to-all 数据量上升。
- GQA 上限：TP ≤ KVHead；qwen_35b 的 `KVHead=2` 把 TP 钉死在 1，所以 `qwen_35b.sh` 默认 `TP=1 EP=8`。

> 一句话：MoE 路径下总 rank = DP·TP·PP·SP·EP，TP 与 EP 正交，二者通过路由后 token 维 `(Seq·KExperts)/(tp·ep)` 联动，必须同时满足 `Experts % EP == 0` 与 `(Seq·KExperts) % (Experts·TP·EP) == 0`。

---

## 六、`--weight_sharded` 的具体作用

一句话：把模型构建时**已经埋好的 FSDP 桥接图**激活成真实的 ZeRO-3 / FSDP 行为——沿 `dp` 轴把权重和梯度切成 1/dp，把 DP 的 `all-reduce` 替换成 `all-gather` + `reduce-scatter`。**不增加并行度轴，只是把现有 DP 轴语义从"权重复制"改成"权重分片"。**

### 6.1 模型构建时无条件埋好的 FSDP 桥接

三个模型都在 decoder block 末尾**无条件**调用 `FSDPWeightGradManager.apply(...)`：

- `llama_model.py:152` / `gpt_model.py:178` / `moe_model.py:221`

`grad_updater.py:205` 对每个 require_grads 的权重做的手术：

| 新增张量 | 形状 | 角色 |
|---|---|---|
| `sharded_weight` | `total_size / fsdp` | 持久化的本地权重分片 |
| `assembled_weight` | `total_size` | forward 前临时拼装的完整权重（瞬时） |
| `assembled_weight_backward` | `total_size` | backward 前再 all-gather 的完整权重 |
| `assembled_grad` | `weight.y_shape` | 局部计算出的完整梯度 |
| `sharded_grad` | `total_size / fsdp` | reduce-scatter 之后的本地梯度分片 |

`grad_updater.py:171` 写死 `reduce_expr = 1/(cp·dp)`：sharded_grad 沿 **CP·DP** 做 reduce-scatter。

### 6.2 控制开关

`main.py:397-402`：
```python
if args.weight_sharded:
    symbol_map_value[fsdp] = args.dp     # 沿 DP 轴切 dp 份
else:
    symbol_map_value[fsdp] = 1           # 不切，等价于"复制"
```

`main.py:433-442 / 524-533 / 614-623` 再做一次符号替换：
```python
if args.weight_sharded:
    ReplicateGraph.apply(graph, old_symbol_map_new_symbol={"fsdp": "dp"})
else:
    ReplicateGraph.apply(graph, old_symbol_map_new_symbol={"fsdp": 1})
```

### 6.3 两种状态对比

| | `weight_sharded=False`（默认） | `weight_sharded=True` |
|---|---|---|
| `sharded_weight.shape` | `total_size` | `total_size / dp` |
| `assembled_weight.shape` | `total_size` | `total_size` |
| sharded ↔ assembled 的 shape 差 | 相等 → 不发 collective | 差 dp 倍 → `ALL_GATHER` |
| sharded_grad.shape | `total_size` | `total_size / dp` |
| assembled_grad → sharded_grad | shape 一致 → `ALL_REDUCE` | shape 差 dp 倍 → `REDUCE_SCATTER` |
| 持久 weight VRAM/rank | 1× | 1/dp |
| 持久 grad VRAM/rank | 1× | 1/dp |
| 通信总量 | 一轮 `ALL_REDUCE`(grad) | `ALL_GATHER`(fwd) + `ALL_GATHER`(bwd) + `REDUCE_SCATTER`(grad) |

通信模式由 `coll_comm_matcher.py` 根据张量两端 shape 自动推导。

### 6.4 VRAM 报告影响

`convert_chakra.py:38-42` + `vram_counting.py:90`：
```python
fsdp_enabled = symbol_map_value.get('fsdp', 0) > 1
```
只有 `fsdp > 1` 时 VRAM 统计才会忽略 `_assembled_weight*` / `_assembled_grad` 等瞬时缓冲；否则它们会被全量计入显存。

### 6.5 与其它并行度的关系

- 不开新轴，**就是把 DP 这个轴的语义从"复制"换成"分片"**。
- 与 TP/PP/EP 正交。
- grad reduce-scatter 沿 `cp·dp` 做，所以 SP rank 也参与权重同步。
- 与 LocalSGD 的联合行为未充分测试。

### 6.6 驱动脚本的相关缺陷

六份 `*.sh` 脚本**全部硬编码 `--weight_sharded 0`**：

1. ZeRO-3 路径在驱动脚本层走不到。
2. 想跑 FSDP workload 必须直接调 `main.py`。
3. `--print_gpu_vram` 在驱动脚本下永远是"权重全量复制"视角。

**最简修法**：
```bash
WEIGHT_SHARDED=${WEIGHT_SHARDED:-false}
... --weight_sharded "${WEIGHT_SHARDED}"
```
并在 `OUTPUT_DIR` 拼接里加 `_fsdp${WEIGHT_SHARDED}` 防覆盖。

---

## 七、有效的 `--model_type` 取值

5 个有效字符串、3 条独立代码路径。

| 取值 | 走哪条分支 | 用什么模型构造器 | 备注 |
|---|---|---|---|
| `llama` | dense 分支（`main.py:409`） | `llama_model.py:llama`（mixprecision 开时）<br>否则 fallback 到 `gpt_model.py:gpt` | 与 `dense` 共用同一段代码 |
| `dense` | 同上 | 同上 | 与 `llama` 完全等价 |
| `gpt` | gpt 分支（`main.py:503`） | `gpt_model.py:gpt` | 不带 mixprecision 切换 |
| `moe` | MoE 分支（`main.py:593`） | `moe_model.py:transformer` | 唯一让 EP 占独立 rank 轴的路径，强制 `tpsp=True` |
| `debug` | debug 分支（`main.py:680`） | 直接加载 `embedding.csv` | 仅 embedding 单算子，强制 `pp == 1` |

> argparse 没设 `choices`，传别的字符串不会报错，所有分支都走不到，main 函数静默退出，**输出目录里不会生成任何 `.et`**——这是个隐藏陷阱。

实质三类路径：dense / llama（mixprecision 决定具体 kernel）；gpt（与 dense 同源但单写）；moe（唯一真正用 EP 轴）。外加 debug 仅供最小子图调试。

---

## 八、`--tpsp` 的具体作用

一句话：在 **`--model_type gpt`** 路径上选两套不同的 sharding CSV ——TP+SP（TPSP，Megatron-LM §3.4）还是纯 TP；其它路径基本是噱头。

### 8.1 选择不同的 sharding CSV

| 模块 | `tpsp=True` 用的 CSV | `tpsp=False` 用的 CSV |
|---|---|---|
| GQA kernel | `tpsp_gpt/group_query_attention_kernel*.csv` | `tp_gpt/group_query_attention_kernel*.csv` |
| GQA surrounding | `tpsp_gpt/group_query_attention_surrounding.csv` | `tp_gpt/group_query_attention_surrounding.csv` |
| FFN | `tpsp_gpt/llama_feed_forward_network.csv` | `tp_gpt/llama_feed_forward_network.csv` |
| LayerNorm | `tpsp_gpt/layer_norm.csv` | `tp_gpt/layer_norm.csv` |
| Residual | `tpsp_gpt/residual.csv` | `tp_gpt/residual.csv` |
| Embedding | `tpsp_gpt/embedding.csv` | `tp_gpt/embedding.csv` |
| Loss | `tpsp_gpt/loss.csv` | `tp_gpt/loss.csv` |

### 8.2 真正差别：Seq 维是否再被 TP 切

```
# tpsp_gpt/llama_feed_forward_network.csv: x0 -> Batch/dp, (Seq/cp)/tp, Dmodel/1
# tp_gpt/llama_feed_forward_network.csv  : x0 -> Batch/dp, (Seq/cp),    Dmodel/1
```

```
# tpsp_gpt/layer_norm.csv: x -> Batch/dp, (Seq/cp)/tp, Dmodel/1
# tp_gpt/layer_norm.csv  : x -> Batch/dp, (Seq/cp),    Dmodel
```

- **TP-only**：LayerNorm/Dropout/Residual 在每个 TP rank 上完整复制（`[B, S, D]`），TP 通信只发生在 attention/FFN 两端的 `all-reduce`。
- **TP+SP（TPSP）**：LayerNorm/Dropout/Residual 沿 Seq 维再切 1/tp（`[B, S/tp, D]`），attention/FFN 边界把 `all-reduce` 拆成 `reduce-scatter` + `all-gather`，**通信总量相同但 LN/Residual activation 显存降到 1/tp**。

### 8.3 各 model_type 对 `--tpsp` 的实际响应

| `model_type` | 是否真正读 `--tpsp` |
|---|---|
| `gpt` | **是**：唯一双轨实现 |
| `llama` / `dense`（mixprecision） | **否**：`llama_model.py:175` 的 `tpsp` 形参从未传到 CSV 选择处，硬编码读 `tpsp/`，等同永远 `tpsp=True` |
| `llama` / `dense`（非 mixprecision） | 走 gpt 路径，`--tpsp` 生效 |
| `moe` | **否**：`main.py:596` 写死 `assert args.tpsp`，硬编码读 `tpsp_moe/` |
| `debug` | **否**：路径里写死 `tpsp/embedding.csv` |

> 隐藏陷阱：用户在 `qwen_*.sh` / `llama3_*.sh`（model_type=llama，默认 mixprecision=True）传 `--tpsp false` **不报错**，但实际仍是 TPSP 形状——开关被静默忽略。

### 8.4 与 SP（cp）的关系

`--sp` 控制 `Seq/cp` 切分（context parallel）；`--tpsp` 控制 Seq 维**还要不要再被 TP 切一次**。两者正交：

| 配置 | Seq 维实际形状 |
|---|---|
| `sp=1, tpsp=true` | `Seq/tp` |
| `sp=4, tpsp=true` | `(Seq/4)/tp` |
| `sp=4, tpsp=false` | `Seq/4` |
| `sp=1, tpsp=false` | `Seq` |

### 8.5 副作用

- TPSP 让 LN/Residual/Dropout 中间张量从 `BSD` 缩成 `BSD/tp`。
- 通信总字节数与 TP-only 相同，但拆成 `RS+AG` 后能与计算 overlap。
- `gpt(...)` 把 `tpsp` 写进 cache key（`gpt_model.py:213`），切换不会跨用缓存。

### 8.6 驱动脚本相关问题

六份脚本**全部不暴露 `--tpsp`**，等价于固定 `tpsp=True`：

- `megatron_gpt_39b/76b` 永远走 TPSP，与 Megatron 论文 §5.1 的 "纯 TP" 基线对不上。
- `llama_model.py:175` 的形参其实不影响 CSV 选择，建议要么修复、要么删除并在帮助里注明 llama 路径只支持 TPSP。

---

## 九、`--pipeline_virtual_stages` 的具体作用

一句话：`v` 是**每个 PP 设备上承载的 chunk 数**——把 `num_stacks` 切成 `v · pp` 段、round-robin 到 `pp` 个设备，从而支持 **Megatron-LM 的 interleaved 1F1B（VPP）调度**。`v=1` 即标准连续映射。

### 9.1 切 chunk 与 round-robin 分配（`main.py:27-92`）

```python
num_chunks  = virtual_stages * range_      # range_ = pp
sizes       = [num_stacks // num_chunks] * num_chunks
device      = chunk_idx % range_           # round-robin
chunk_local = chunk_idx // range_          # chunk 在本设备上的序号 (0..v-1)
```

举例 `num_stacks=8, pp=4, v=2`：

```
chunk_idx :  0  1  2  3  4  5  6  7
device(%4):  0  1  2  3  0  1  2  3       ← 每设备 v=2 个 chunk
chunk_local: 0  0  0  0  1  1  1  1

block 0 → dev 0 chunk0
block 4 → dev 0 chunk1     ← 同一设备上"非连续"两段 layer
```

`v=1` 时每设备拿到一段连续的 layer。`out_emb`/`loss` 始终钉在最后一个 block 的设备。

### 9.2 喂给 `1f1b-interleaved` 调度器（`pipeline_schedule.py:249-308`）

按 Megatron 的 VPP 公式排队：

```python
microbatch_id_in_group = k % (p * v)
model_chunk_id         = microbatch_id_in_group // p
microbatch_id          = (k // (p * v)) * p + (k % p)
warmup                 = (p - rank - 1) * 2 + (v - 1) * p
```

每个 microbatch 在每个设备上前后各访问 v 次 chunk，warmup 比纯 1F1B 多 `(v-1)·p` 步。

### 9.3 对图的具体影响

| | `v=1`（默认） | `v>1`（interleaved） |
|---|---|---|
| 设备 ↔ layer | `dev k` 持有 layer `[k·L/p, (k+1)·L/p)` 连续段 | `dev k` 持有 v 段非连续 layer |
| `block_to_chunk_local` | 一律 `0` | 取值 `0..v-1` |
| 跨设备 SEND/RECV / iter | `2(p-1) m` | `2(p-1) m + 2(v-1) m` |
| pipeline bubble | `(p-1)/m` | `(p-1)/(v·m)` |
| 调度切换 | 任选 | 只有 `1f1b-interleaved` 利用 v |

> 改 v 不改并行度轴数也不改总 NPU；只改"layer→设备"拓扑和 1F1B 内部排序。

### 9.4 必须满足的约束

1. `num_stacks ≥ v · pp`（v=1 也要满足）。
2. `v > 1` 时强制 `num_stacks % (v · pp) == 0`；v=1 容忍余数。
3. `v > 1` 且 `1f1b-interleaved` 时强制 `num_micro_batches % pp == 0`。
4. 默认 `v=1`，`≥ 1`。

举例 `llama3_70b` 默认 `LAYER=80, PP=4`：v∈{1,2,4,5,8,10}（被 4 整除且 80 被 v·4 整除）。

### 9.5 与其它参数的耦合

- `--pipeline_schedule`：v 只在 `1f1b-interleaved` 下产生**调度收益**；其它 schedule 即使 v>1 也只改了张量映射，bubble 不变。
- `--pp`：v 与 pp 共同决定 `num_chunks = v·pp`；增大 v 等价于"虚拟"增大 pp，但不消耗额外 NPU。
- `--micro_batch / --batch / --dp`：决定 `num_micro_batches`，进而决定整除约束。
- `--mixed_precision / --activation_recompute / model_type`：与 v 正交。

### 9.6 bubble 实际下降

`bubble ratio = (p - 1) / (v · m)`

`llama3_70b` 默认 `pp=4, m=28`：

| v | bubble |
|---|---|
| 1 | ≈ 10.7% |
| 2 | ≈ 5.4% |
| 4 | ≈ 2.7% |

代价：每个 micro-batch 跨设备 SEND/RECV 次数从 `2(p-1)` 涨到 `2(p-1) + 2(v-1)`，链路紧张时收益会被抵消。

### 9.7 驱动脚本的相关情况

- `llama3_70b/megatron_gpt_39b/megatron_gpt_76b.sh` 暴露 `PP_VIRTUAL`（默认 1），写入 `OUTPUT_DIR`。
- `qwen_32b/qwen_35b.sh` 也暴露但默认 1。
- **`llama3_8b.sh` 不暴露**，固化 v=1。
- 没有任何脚本校验 `num_micro_batches % pp == 0`：用户改 `BATCH/MICROBATCH/DP` 后开 `v>1 + 1f1b-interleaved` 会在 STG 内部直接 `ValueError`。
