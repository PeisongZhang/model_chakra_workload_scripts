# 影响实验正确性的未完成项（排期计划）

> **定位**：`docs/PDT/` 下 P0–P3 八个子项已全部标 ✅，但在跑 `megatron_gpt_experiment/`（见 `astra-sim/megatron_gpt_experiment/analysis_report.md`）时发现了若干**真正会让仿真结果偏离物理语义或论文对比口径**的问题。本文件把这些问题按优先级排期。
>
> **目标读者**：后续 Claude Code 会话。每一项写成**自包含**——只读此文件的本节就能按步骤完成修复。
>
> **约定**：修完一项请把该小节的 `状态: 未完成` 改为 `状态: 已完成 (commit <sha>)`，并把论据补到节末的"验证记录"里。

---

## 优先级总览

| # | 问题 | 优先级 | 修复工作量 | 实验受影响范围 |
|---|------|:------:|:----------:|----------------|
| 1 | `1f1b-interleaved` 只做了图层交错，调度 fallback 到 mb 粒度 1F1B | **P0** | 1.5–2 人日 | 任何用 `PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL>1` 的实验都拿到错误结果（实测 −34.5%） |
| 2 | `Statistics.cc:extract_comm_bytes` 用 `network_bandwidth.has_value()` 分类 p2p/coll，实际全部命中 p2p | **P0** | 0.5 人日 | P2-A 报出的 `Effective BW (p2p=… coll=…)` 恒把 coll 算 0 —— 所有基于该字段的结论都错 |
| 3 | `_print_gpu_vram` 把 `activation_recompute` VRAM 缩成固定 `keep_ratio=0.2` | **P1** | 1 人日 | VRAM 报告与 P3-A VRAM cap 的 OK/OVERFLOW 判定在 AR=on 下失真 |
| 4 | Roofline 的 `achievable_fraction` 是全局系数，未做 per-op-type | **P1** | 1.5 人日 | 无法复现论文 §5.8 的 ~11–19% 峰值缺口；与论文表 1 比需手调全局 fraction，粗且无法分算子 |

---

## 1. `1f1b-interleaved` 半实现：图层交错了，调度没交错

**状态**：未完成
**优先级**：P0

### 影响

`PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL>=2` 组合目前**必然劣化性能**（见 `astra-sim/megatron_gpt_experiment/analysis_report.md` §4.2）：

| 配置 | 39.1B wall | TFLOP/s/GPU | Δ vs 138 |
|------|-----------:|------------:|---------:|
| b=2 / 1f1b / v=1（baseline） | 14.19 s | 140.5 | **+1.8%** |
| b=2 / 1f1b-interleaved / v=2 | **21.25 s** | **90.4** | **−34.5%** |

论文 §5.3.2 报 ~+10%。用户如果凭直觉打开 interleaved 会得到一个**方向相反**的结论。

### 代码定位

- `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/pipeline_schedule.py:245-294` — `_build_1f1b_interleaved_sequence(num_mb, p, v, rank)` 已经实现了正确的 `(mb, chunk)` 二元调度序列，但它**期望** `chunk` 信息从外部传入。
- `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/pipeline_schedule.py:297-312` — `_apply_1f1b_interleaved_to_rank` 当前直接 fallback 到 `_apply_1f1b_to_rank`，并挂 TODO 注释。
- `dnn_workload/symbolic_tensor_graph/main.py:26-38` — `_build_chunk_cumulative_bounds` / `_block_idx_to_device` 是唯一知道 chunk→device 映射的地方；这个信息算完就丢。
- `dnn_workload/symbolic_tensor_graph/main.py:_create_pipeline_tensor_map_mix_precision` 和 `_create_pipeline_tensor_map` 是生成 block→device 映射的入口；chunk 信息只存在局部变量 `cumulative` 里。

### 根因

`pipeline_schedule.py:_apply_1f1b_interleaved_to_rank` 需要区分一个 backward COMP 节点属于哪个 `chunk_on_device`（同一 device 上的第几个 virtual stage），但这个信息没从 `_create_pipeline_tensor_map_*` 传下来。后处理能拿到的只有 tensor id 里的 `transformer.{N}.`——它知道 block 号 N，却不知道 N → chunk_on_device 的映射规则。

结果：半实现"图拓扑交错 + 调度 mb 粒度串行"。非连续 chunk 带来额外 cross-pp p2p，但没有用 interleaved 填 bubble，净效应是退化。

### 修复方案

分两步：

**(a) 把 chunk 元数据透出来**

在 `BundledConvertChakra.apply` 或 `GraphDistributer` 里，为每个 `HybridGraph` 附一个 `block_to_chunk_on_device: dict[block_idx, chunk_idx_local]`。最简单的做法：

1. 在 `main.py:_create_pipeline_tensor_map_mix_precision` 计算完 `cumulative`（chunk 上限累积）和 `chunk_idx = next(i for i, up in enumerate(cumulative) if block_idx < up)` 后，额外产出一个 `block_to_chunk_local`：
   ```python
   block_to_chunk_local = {}
   for block_idx in range(num_stacks):
       chunk_idx = next(i for i, up in enumerate(cumulative) if block_idx < up)
       device = chunk_idx % range_
       chunk_on_device = chunk_idx // range_  # 0..v-1
       block_to_chunk_local[block_idx] = chunk_on_device
   return _tensor_map, block_to_chunk_local
   ```
2. 把 `block_to_chunk_local` 一路传到 `BundledConvertChakra.apply(..., block_to_chunk_local=...)`，挂到每个 `HybridGraph`（比如加一个 `hybrid_graph.block_to_chunk_local` 属性，`_mix_precision` 路径和非 mix 路径共用）。
3. 非 mix 路径的 `_create_pipeline_tensor_map` 也同步出 `block_to_chunk_local`。

**(b) 让 `_apply_1f1b_interleaved_to_rank` 真正按 `(mb, chunk)` 产 ctrl_deps**

1. 读取 `hybrid_graph.block_to_chunk_local`。
2. 对每个 backward/forward COMP 节点：从节点名里正则出 `transformer.{N}.`，查出 `chunk_on_device = block_to_chunk_local[N]`；对 embedding/loss 节点，按 pipeline_schedule.py 现有的命名规则归类到 chunk 0（device 0 侧）或 chunk v-1（device p-1 侧）。
3. 对 micro-batch 索引使用现有的 `mb{i}.` 前缀解析。
4. 把 `_build_1f1b_interleaved_sequence(num_mb, p, v, rank)` 产的 `[(PHASE_F, mb, chunk), (PHASE_B, mb, chunk), …]` 序列逐项匹配到 `(rank, mb, chunk_on_device, PHASE)` 的节点集合，按相邻 pair 注入 `ctrl_deps`。
5. 注意：同一个 `(rank, mb, chunk, PHASE)` 下有多个节点（多个 transformer block 的 COMP 节点）；只在相邻组的**首节点**加 ctrl_dep，保留组内原有 data dep。

### 复现/观测

```bash
# 跑半实现（会看到 wall 变长）
cd /home/ps/sow/part2/dnn_workload/megatron_gpt_39b
PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL=2 ACTIVATION_RECOMPUTE=1 bash megatron_gpt_39b.sh
# 对比 baseline（同配置 v=1）
PP_SCHEDULE=1f1b PP_VIRTUAL=1 ACTIVATION_RECOMPUTE=1 bash megatron_gpt_39b.sh
```

### 验收标准

1. **新回归测试** `test_cases/test_pipeline_interleaved.py` 扩展一段：对 `(p=4, v=2, m=8)` 构造的 bundled graph，断言每个 rank 上相邻 chunk 的 backward 之间有 ctrl_dep、相邻 chunk 的 forward 之间也有。
2. **端到端**：`megatron_gpt_39b` `1f1b-interleaved + v=2` 配置下 `run_analytical.log` 的 wall 时间**不高于** `1f1b + v=1` baseline 的 1.02×（允许 2% 噪声），目标 5–15% **加速**（论文 §5.3.2）。
3. `analysis_report.md` §4.2 表里加一行"修复后 v=2 结果"，Δ 落到 ±5% 之内并更新 §9.1#4 的状态。

---

## 2. `Statistics.cc::extract_comm_bytes` 的 p2p/coll 分类启发式失效

**状态**：未完成
**优先级**：P0

### 影响

`run_analytical.log` 里 P2-A 新增的这行：

```
sys[N], Comm bytes: <total> (p2p=<X> coll=<Y>), Effective BW: ... GB/s (p2p=..., coll=...)
```

**`coll=` 恒为 0，`p2p=` 等于 total**。任何基于这两个拆分字段的结论都错（例如"这次实验 p2p 占通信的比例"）。`total_comm_bytes_` 本身是对的，全局 `Effective BW` 也对。

### 代码定位

- `astra-sim/astra-sim/workload/Statistics.cc:305-327` — `extract_comm_bytes()`，当前逻辑：
  ```cpp
  if (stat.network_bandwidth.has_value()) {
      this->total_p2p_bytes_ += b;
  } else {
      this->total_coll_bytes_ += b;
  }
  ```
- `astra-sim/astra-sim/workload/Workload.cc:509-513` — **coll 完成**回调里设 `op_stat.network_bandwidth`。
- `astra-sim/astra-sim/workload/Workload.cc:553-561` — **p2p 完成**回调里也设 `op_stat.network_bandwidth`。

### 根因

`network_bandwidth` 在两类通信完成回调里都会被赋值（只要 `execution_time > 0 && comm_size.has_value()`），所以 `has_value()` 对两者都为真，启发式全部落到 p2p 分支。

### 修复方案

改用 Chakra 节点类型做分类，这是权威信息且在 `OperatorStatistics` 之外可访问。

**方案 A（推荐，最小改动）**：在 `Statistics::record_start` / `record_end` 阶段已知节点，把节点的 `ChakraNodeType` 缓存进 `OperatorStatistics`，然后 `extract_comm_bytes` 按 type 分类：

1. `astra-sim/astra-sim/workload/Statistics.hh` `OperatorStatistics` 里加一个字段 `std::optional<ChakraNodeType> chakra_node_type`（或 `int` 别名，避免新 include）。
2. `astra-sim/astra-sim/workload/Statistics.cc` — `record_start(node, tick)` 里写入 `stat.chakra_node_type = node->type();`。检查文件开头是否已经 include `<extern/graph_frontend/chakra/schema/protobuf/et_def.pb.h>`（同文件第 61-62 行已用过 `ChakraNodeType::COMM_*`，说明已可用）。
3. `extract_comm_bytes` 改为：
   ```cpp
   for (const auto& [node_id, stat] : operator_statistics) {
       if (stat.type != OperatorStatistics::OperatorType::COMM) continue;
       if (!stat.comm_size.has_value()) continue;
       if (!stat.chakra_node_type.has_value()) continue;
       uint64_t b = stat.comm_size.value();
       this->total_comm_bytes_ += b;
       auto t = stat.chakra_node_type.value();
       if (t == ChakraNodeType::COMM_SEND_NODE || t == ChakraNodeType::COMM_RECV_NODE) {
           this->total_p2p_bytes_ += b;
       } else if (t == ChakraNodeType::COLL_COMM_NODE) {
           this->total_coll_bytes_ += b;
       }
   }
   ```

**方案 B（不推荐）**：判断 `op_stat.network_bandwidth` 在哪条回调路径写入 —— 信息不够，两条路径都会写，改不动根本分辨逻辑。

### 复现/观测

```bash
# 任何已经跑过的仿真 log 都能观察到
grep 'Comm bytes' /home/ps/sow/part2/astra-sim/megatron_gpt_experiment/gpt_39b_512/run_analytical.log | head
# 会看到 coll=0 对所有 rank 成立，但工作负载里明显有 all-reduce/all-gather
```

### 验收标准

1. 重跑 `megatron_gpt_39b/run_analytical.sh`，`grep 'Comm bytes' run_analytical.log` 后 `coll=` 非零且 `p2p + coll == total`。
2. 对一个 pure-DP 不含 TP 的配置（如 `DP=4 TP=1 PP=1`），`p2p=0` 且 `coll = total`（因为没有 PP，也就没有 P2P SEND/RECV）。
3. 更新 `phase2_3_completion_zh.md` 第 §已知限制 里提到"启发式"的那一条：删除或改为"已修"。

---

## 3. 激活重算 VRAM 缩减系数是硬编码的 0.2

**状态**：未完成
**优先级**：P1

### 影响

`_print_gpu_vram` 在 `--activation_recompute true` 时直接把 acts 乘 `keep_ratio=0.2`。这意味着：
- **显存报告不正确**：实际保留比例取决于 batch / seq / head / block 数。对 `l=48` 层模型用选择性重算，论文公式 `A_input + l/c · A_intermediate`（`c=1` 对应每层都重算）下的实际保留比可能是 0.05–0.3 的任意值，0.2 只是经验中点。
- **P3-A VRAM cap 检查（`vram-capacity-gb`）可能给出错误的 OK/OVERFLOW 判定**。在 `megatron_gpt_experiment/` 的 `run_analytical.log` 里可观察到。
- AR on/off 的 VRAM 对比被人为锁死成 5× 比例，导致 Pareto 分析失真。

### 代码定位

- `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/vram_counting.py:95-146` — `_print_gpu_vram(bundle_graph, symbol_map, ..., activation_recompute, activation_recompute_keep_ratio=0.2)`。
- `dnn_workload/symbolic_tensor_graph/main.py` `_postprocess_chakra_graph` 上游调用链。
- 权威值应来自 ASTRA-sim 的 `LocalMemUsageTracker`（P3-A）。

### 修复方案

两条路，至少做一条：

**路径 A（推荐）：把 VRAM 峰值以 ASTRA-sim 仿真结果为权威**

1. `vram_counting.py` 改成"只估无重算的 acts"，在 `activation_recompute=True` 时**不**报 acts，改为打印提示："激活重算已启用；acts 峰值请以 ASTRA-sim `[workload] sys[N] peak memory usage:` 为准"。
2. 文档 `phase1_completion_zh.md` §已知限制、`reference_zh.md` 增加交叉引用：VRAM 权威值在仿真 log 里。
3. 去掉 `activation_recompute_keep_ratio` 参数。

**路径 B（次选）：按张量真实分类**

1. 在 `vram_counting.py:_tensor_size_bytes` 区分 "stage input activation" 和 "intermediate activation"。"stage input" 定义为：每个 pipeline stage 的第一个 block 的输入张量 + micro-batch 归一输入；"intermediate" 定义为其余 block 内临时激活。
2. 开 AR 时按论文公式：`A_input + l/c · A_intermediate`（`c` 是 checkpoint 段数，默认 `c=1` 等价每层重算）。
3. 需要给张量打类型标签；目前 STG 张量 id 的命名（`transformer.N.attn.xxx`）能正则分拣 block 内/跨 block，但 stage 输入比较隐晦，需要 `GraphDistributer` 里拿跨 bucket shadow 的反向信息。

### 验收标准

路径 A：
- 修改后 `_print_gpu_vram` 在 AR=on 时不产出数值化的 acts，改为一行提示；改动在 `test_cases/` 新增 `test_vram_ar_note.py`，断言日志文本里有该提示且不含 `@0.2`。
- `phase1_completion_zh.md` 更新，`analysis_report.md` §4 / §5 重新校准 "VRAM" 相关描述。

路径 B：
- `_print_gpu_vram` 输出遵循 `A_input + l/c · A_intermediate`；对 `qwen_32b LAYER=32 PP=4 MICROBATCH=2 BATCH=128` 的示例配置，AR=on 时 acts 下降比例在 `c=1,l=8` 下应 ≈ `1/8 + stage_input_frac`，通常 0.12–0.25。
- 交叉核对：同一配置下 ASTRA-sim `peak memory usage` 与 `_print_gpu_vram` 估计误差 < 20%（见 `implementation_plan_zh.md` §P3-A 验证条款）。

---

## 4. Roofline `achievable_fraction` 只有全局系数，无 per-op-type

**状态**：未完成
**优先级**：P1

### 影响

`implementation_plan_zh.md` §P2-B 原本的设计是 `peak-perf-per-op-type: {GEMM: 312, ELEMWISE: 90, SOFTMAX: 60, REDUCE: 40}`，用于复现论文 §5.8 的 11–19% 峰值缺口并解释"为什么 softmax / element-wise 会拖慢"。当前实现只支持**全局**单系数 `peak-perf-achievable-fraction`（默认 1.0）。

后果：
- 要对着论文 fp TFLOP/s 跑 sweep，必须全局调一个 fraction，它同时压 GEMM 与 softmax；无法区分"fused vs. non-fused attention"这类算子级效应。
- `qwen_experiment/in_dc/astra_system.json` 示例里写的 `"peak-perf-achievable-fraction": 1.0` 等价没开；真正有意义的数字需要用户自己猜。
- 与 P1-A scatter/gather 开/关的对比存在"通信-计算耦合误差"——计算侧 peak 过估反过来让通信"显得占比更高"。

### 代码定位

- `astra-sim/astra-sim/system/Roofline.hh:13-27` — `Roofline` 类成员仅 `bandwidth`、`peak_perf`、`achievable_fraction`。
- `astra-sim/astra-sim/system/Roofline.cc:27-33` — `get_perf(double OI)` 只返回 `min(bw*OI, peak) * fraction`，无 op-type 参数。
- `astra-sim/astra-sim/workload/Workload.cc:276-279` — `elapsed_time = num_ops / roofline->get_perf(OI)` 是唯一调用点，没传节点类型。
- `astra-sim/astra-sim/system/Sys.cc:413-420` — 从 `astra_system.json` 读 `peak-perf` 并构造 `Roofline`；读 `peak-perf-achievable-fraction`。
- STG 侧：`dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/convert_chakra.py` 在写 COMP_NODE 时目前没有把 op 子类型写进 Chakra 节点（attention、GEMM、softmax 混在一个 `COMP_NODE` 里）。

### 修复方案

三步，跨两个仓库：

**(a) Chakra 侧给 COMP 节点标注 `op_type`**

选项 I（最保守）：用 Chakra 节点的 `name` 字段的命名约定 —— STG 已经给节点命名如 `transformer.0.attn.q_proj@forward`。在 `Workload.cc` 里用字符串前缀正则识别类别。缺点：脆、易漂。

选项 II（推荐）：在 Chakra ET 节点的 `attr` 里加一个 `op_category: int`（0=GEMM, 1=ELEMWISE, 2=SOFTMAX, 3=REDUCE, 4=OTHER）。STG 侧在 `convert_chakra.py` 的 `_convert_comp_node` 位置按张量 op_type 注入。Chakra 协议已支持任意 attr（`et_def.proto` 的 `repeated AttributeProto attr`），不需要改 proto。

**(b) Roofline 扩成表**

1. `Roofline.hh` 加 `std::unordered_map<int, double> peak_per_op_category;`（没值时 fallback 到 `peak_perf`），并增补 `double mem_bw_per_op_category` 同构（可选）。
2. `Roofline.cc` 新签名：`double get_perf(double OI, int op_category = -1)`。默认 -1 走旧路径保兼容。
3. `Sys.cc:initialize_sys` 解析 JSON 新字段 `"peak-perf-per-op-category": {"GEMM": 312, "ELEMWISE": 90, ...}`；符号常量映射到整数 category id 用一个小 helper。

**(c) Workload 侧传 op_category**

1. `Workload.cc:276-279` 附近读节点 attr 里的 `op_category`，调 `roofline->get_perf(OI, op_category)`。
2. 兼容：节点没有 attr 时 `op_category = -1`，退回全局 `peak_perf`。

**(d) 示例配置更新**

`qwen_experiment/in_dc/astra_system.json` 增加：
```json
"peak-perf-per-op-category": {
    "GEMM": 312,
    "ELEMWISE": 90,
    "SOFTMAX": 60,
    "REDUCE": 40
}
```
同时保留 `peak-perf-achievable-fraction` 作为**顶层兜底**（对未分类节点与 all categories 同时生效）。

### 复现/观测

现状观察：跑 `megatron_gpt_39b` AR=on 实验，Δ vs 论文 138 TFLOP/s 是 +1.8%——这个精度靠的是用全局 fraction=1.0 + `active-chunks-per-dimension=2` 配合碰上了，没有物理对应。如果换 76B 到 TP=4 或做 b=4 sweep，就会看到 Δ 明显漂移（见 `analysis_report.md` §4.2）。

### 验收标准

1. `qwen_experiment/in_dc/astra_system.json` 按上述示例配置，对 `qwen_32b LAYER=32 BATCH=128 PP=4 TP=8` 的单迭代仿真，不同 `peak-perf-per-op-category` 取值下 `sys[0] TFLOP/s/GPU` 应呈现符合预期的单调变化。
2. 新增单元测试 `astra-sim/tests/roofline_per_op.cc`（或 Python 端等效脚本）：构造一个 mini Chakra 图，断言不同 op_category 的 `elapsed_time` 与表里的 peak 严格对应。
3. STG 测试 `test_cases/test_op_category_labeling.py`：生成一个 attention + MLP 的小工作负载，验证 Chakra ET 里相应节点的 `op_category` attr 填对了。
4. 更新 `reference_zh.md` 把 `peak-perf-achievable-fraction` 标注为"全局 fallback"，新增 `peak-perf-per-op-category` 表。
5. 更新 `phase2_3_completion_zh.md` §已知限制里关于"全局系数"的条目；如果 AR=on 两行的 Δ 在开启 per-category 后变化 > 3%，说明此前的 +1.8% / +1.4% 精度有相当一部分靠碰巧，要在 `analysis_report.md` §4 补充此坦白。

---

## 跨项修复后的统一验证

三项修完后做一次端到端回归：

```bash
cd /home/ps/sow/part2/astra-sim
bash examples/run_scripts/analytical/congestion_aware/$(ls examples/run_scripts/analytical/congestion_aware | head -1)   # 冒烟
cd megatron_gpt_experiment
bash gpt_39b_512/run_analytical.sh
bash gpt_76b_1024/run_analytical.sh
python collect_and_compare.py
```

对比 `report.md` 与 `analysis_report.md` §4 的老表。AR=on 两行 Δ 重新校准；`Bubble time` / `Comm bytes (p2p/coll)` / `peak memory usage` 三行都应非零且物理合理。

完成后在 `README.md` 顶部的"一张图看懂完成情况"框里把 ✅ 的含义提升到"已验证经受住 39B/76B 全规模跑通"。