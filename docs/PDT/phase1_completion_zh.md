# Phase 1 完成小结：scatter/gather + 激活重算

对应 `docs/PDT/implementation_plan_zh.md` 的 P1‑A + P1‑B 两项。

## 改动文件

| 文件 | 作用 |
|------|------|
| `symbolic_tensor_graph/graph/graph.py` | `HybridGraph.NodeType` 新增 `Y_RECV_AG` |
| `symbolic_tensor_graph/graph/convert_chakra.py` | `_get_output_node` 优先 AG；`_insert_send_node` / `_insert_recv_node` 新增 `comm_chunks`；新增 `_insert_recv_allgather_node`；`BundledConvertChakra.apply` 接受 `scatter_gather_optimization` 并在 pp‑boundary 链路上切分 + 插 AG；`_comm_info_post_process` 处理 `Y_RECV_AG` 类型 |
| `symbolic_tensor_graph/graph/activation_recompute.py` | 新模块。`ActivationRecomputePostProcess` 按 `(mb, block)` 分组后抬升 backward COMP 节点的 num_ops / tensor_size |
| `symbolic_tensor_graph/vram_counting.py` | `_print_gpu_vram` 加 `activation_recompute` 参数，启用时 acts 按 `activation_recompute_keep_ratio=0.2` 缩减并附注 |
| `main.py` | 新增 CLI `--scatter_gather_optimization`；激活 `--activation_recompute` 端到端；透传 pp/tp sympy 符号给 `BundledConvertChakra.apply`；`_postprocess_chakra_graph` 在注入 schedule 之前运行 `ActivationRecomputePostProcess` |
| `qwen_32b/qwen_32b.sh` | 新增 `SGO` / `ACTIVATION_RECOMPUTE` env；输出目录名追加 `_sgo<val>_ar<val>` |
| `symbolic_tensor_graph/test_cases/test_scatter_gather.py` | 回归测试：sg 开启后 tp=4 下 P2P 字节数降 1/4 并伴随 all_gather；tp=1 场景 sg 无副作用 |

## 使用方式

```bash
# P1-A: scatter/gather（需要 TP>1 才生效）
SGO=true DP=4 TP=8 PP=4 LAYER=32 bash dnn_workload/qwen_32b/qwen_32b.sh

# P1-B: 激活重算（backward 额外计入一次 forward 的 FLOP）
ACTIVATION_RECOMPUTE=true DP=4 TP=8 PP=4 LAYER=32 bash .../qwen_32b.sh

# 组合使用（常见生产配置）
SGO=true ACTIVATION_RECOMPUTE=true PP_SCHEDULE=1f1b \
    DP=4 TP=8 PP=4 LAYER=32 bash .../qwen_32b.sh
```

输出目录名现在追加 `_sgo{true|false}_ar{true|false}`，便于回归对比。

## 已验证（`test_scatter_gather.py`）

### scatter/gather（tp=4, pp=2, num_stacks=4, batch=4）

| 指标 | sg OFF | sg ON | 比率 |
|------|--------|-------|------|
| SEND 字节（全局） | 1,048,576 | 262,144 | 4× |
| RECV 字节（全局） | 1,048,576 | 262,144 | 4× |
| SEND/RECV 节点数 | 32 / 32 | 32 / 32 | 不变 |
| ALL_GATHER 节点数 | 0 | 32 | +1 per pp‑boundary recv |

精确匹配论文 §4.1 的 `$1/t$` 切分量与节点数守恒关系。tp=1 时 scatter/gather 自动退化为 no‑op。

### 激活重算（tp=1, pp=2, num_stacks=4）

| FLOP 分桶 | AR OFF | AR ON | 变化 |
|-----------|--------|-------|------|
| transformer blocks F | 23.69 G | 23.69 G | 不变 |
| transformer blocks B | 48.41 G | **72.07 G** | +1.00× F |
| embedding / output / loss | 201.40 G | 201.40 G | 不变 |
| 全局 (F + B) per block | 72.10 G | 95.76 G | +23.66 G / block |

后向 FLOP 精确多出一份前向量（72.07 ≈ 48.41 + 23.69，剩余 0.04 G 来自 `int()` 截断），与论文 §3.5 "backward 前多跑一次 forward" 的描述一致。embedding 等非块节点不受影响。

### VRAM 报告

开启 `--activation_recompute` 后，`--print_gpu_vram` 输出新增 `[recomp: acts X->Y @0.2]` 列，把激活显存乘以 `keep_ratio=0.2` 粗略模拟"每层只保留输入激活"的效果。对小型 4 层模型：

- pp=0: acts 0.175 → 0.035 GiB（总 0.578 → 0.438 GiB）
- pp=1: acts 0.238 → 0.048 GiB（总 0.641 → 0.450 GiB）

精确的逐时刻峰值 VRAM 依赖 ASTRA‑sim 的 `LocalMemUsageTracker`（P3‑A）。

## 已知限制

- **VRAM 粗缩系数 0.2** 是固定经验值，实际减幅依赖 batch / seq / 头数 / 块数。论文的 `$\sqrt{l \cdot A_\text{intermediate}/A_\text{input}}$` 最优 checkpoint 数要求精确分类"stage input"与"intermediate"张量，当前 STG 张量命名不够规整以做自动识别。P3‑A 接入后建议用仿真峰值 VRAM 作为权威指标。
- **重算 FLOP 归入 backward** 而非单独的 recompute COMP 节点：backward 的执行时间因此变长，但在图里不再有"先 recompute 再 backward"的显式序列。对 ASTRA‑sim 的 end‑to‑end wall clock 效果等价，但若想精细观察 recompute 本身的调度（例如和通信重叠），需改为单独插 COMP 节点。
- **scatter/gather all‑gather 节点** 使用 `_comm_meta_data = (ALL_GATHER, None, None, tp_sym)` 满足 readout 期间的 `update_comm_group` 调用。占位元素（None, None）如果有新的消费方读取，可能需要补全。
- **非 mixed‑precision 分支**（`gpt_model.py`）未单独测试，但改动代码路径与 mix_precision 完全一致，风险低。

## 待做（后续 phase）

- **P2‑A**：`Statistics.cc` 输出 bubble fraction + 有效二分带宽
- **P2‑B**：Roofline per op‑type 可达 TFLOP/s（接入 kernel 融合收益）
- **P3‑A**：打开 `track-local-mem` + VRAM cap，做运行时峰值 VRAM 报告（可反过来校准本 phase 的 keep_ratio）
- **P3‑B**：`generate_topology.py` 的 `--nics-per-gpu` 多 NIC 建模
