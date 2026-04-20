# Megatron‑LM (PTD‑P) 仿真能力补齐实现计划

## Context

`docs/PDT/gap_analysis_zh.md` 盘点了当前仿真栈（`dnn_workload/qwen_32b/qwen_32b.sh` + `astra-sim/qwen_experiment/in_dc/analytical.sh`）对 Megatron‑LM SC'21 论文训练方案的覆盖度。其中 **未覆盖但可实现** 的特性，本文给出落地路径。

> **重要修正**：之前 gap 分析把"层次化拓扑"列为缺失是看错了——`qwen_experiment/in_dc/topology.txt` 已经是分层结构（NVLink 4800 Gbps 节点内、NIC 200 Gbps 节点间、leaf‑spine 胖树），生成器 `generate_topology.py` 参数化良好。**本计划不包含这一项**。

## Scope

### 纳入本计划（可实现）
1. **P0‑A** 显式流水线调度策略（1F1B / GPipe）
2. **P0‑B** 交错式（interleaved）调度（virtual stages $v > 1$）
3. **P1‑A** scatter / gather 通信优化
4. **P1‑B** activation recomputation 端到端打通
5. **P2‑A** 仿真报告增强：bubble time + 有效二分带宽
6. **P2‑B** 每 op‑type 的 achievable TFLOP/s（覆盖算子融合收益）
7. **P3‑A** VRAM 容量约束与峰值上报（复用已有 `LocalMemUsageTracker`）
8. **P3‑B** 8× IB NIC 多链路建模（只需改拓扑生成器，格式已支持）

### 不纳入（暂不可行或 ROI 低）
- checkpoint I/O 建模（需要系统层新增 I/O 资源抽象，工程量大、论文只做定性描述）
- 精细到 FP16/FP32 的 master weight 分精度建模（通过 P2‑B 的 per‑op TFLOP/s 可近似）
- 完整自动 FlexFlow 式切分搜索

---

## 架构事实（来自探索结果）

**STG 侧**（`dnn_workload/symbolic_tensor_graph/`）：
- `main.py:72-117` — `_create_pipeline_tensor_map[_mix_precision]`，每 tensor 映射到 `{pp_dim: stage_id}`，`GraphDistributer` 当作不透明 bucket 键。
- `symbolic_tensor_graph/graph/graph_distributer.py:35-91` — `_temporal_dispatch_tensors` + `_fix_cross_bucket_data_dependancies` 生成跨 stage 的 shadow tensor。
- `symbolic_tensor_graph/chakra/convert_chakra.py:460-493` — `_insert_send_node / _insert_recv_node` 写入 P2P 节点，comm_size 取完整张量。
- `convert_chakra.py:661-686` — `BundledConvertChakra.apply` 里 `remote_readable_rank` / `shadow_readable_rank` tuple 可见 TP rank 身份。
- `convert_chakra.py:527-539` — 已存在一个 disabled 的 `add_ctrl_deps()` 基础设施可复用。
- `symbolic_tensor_graph/graph/grad_updater.py:364+` — `MicroBatchReplicatorPostProcess` 把 mb 索引嵌入节点名前缀 `mb{i}`，可作为调度判据。
- `main.py:166-170` — `--activation_recompute` 存在但 **未接入** 任何下游。

**ASTRA‑sim 侧**（`astra-sim/`）：
- `astra-sim/workload/Workload.cc:147-164` — `issue_dep_free_nodes()`，所有 dep‑free 节点 FIFO 出队，排序/优先级是计划插入点。
- `astra-sim/workload/HardwareResource.cc:100-114` — `max_in_flight_gpu_comp_ops` 等资源上限。
- `astra-sim/workload/Statistics.cc:144-215, 268-291` — `report()` / `post_processing()`，per‑rank 指标的唯一输出点。
- `astra-sim/system/Sys.cc:404-415` — 从 `astra_system.json` 读 `peak-perf` 并构造 `Roofline`。
- `astra-sim/system/Roofline.cc:13-25` — 仅 $perf = \min(bw\cdot OI, peak)$，无 op‑type 区分。
- `astra-sim/workload/Workload.cc:276-279` — `elapsed_time = num_ops / roofline->get_perf(OI)`；是切换 per‑op 表的单点。
- `astra-sim/workload/LocalMemUsageTracker.cc` — 完整功能类，`astra_system.json` 里 `"track-local-mem": true` 启用；`getPeakMemUsageFormatted()` 可直接报告。
- `extern/network_backend/analytical/congestion_aware/basic-topology/CustomTopology.cpp:107-189` — 拓扑解析；支持异构带宽、节点-交换机分离、多平行链路（NIC×8）。

---

## 实施计划（按优先级分期）

### Phase 0（P0‑A + P0‑B）——流水线调度策略

**目标**：能切换 GPipe / 1F1B / interleaved 1F1B，并在仿真指标里直接给出 bubble 占比。

**改动 1：STG 支持非连续 block→stage 映射**（interleaved）
- 文件：`dnn_workload/symbolic_tensor_graph/main.py`
- 加参数 `--pipeline_virtual_stages v`（默认 1）。
- 修改 `_create_pipeline_tensor_map_mix_precision` 与 `_create_pipeline_tensor_map`：当 `v > 1` 时把 `num_stacks` 切成 `v * pp` 个 chunk，chunk $j$ 归属 device $j \bmod pp$；保持原 API 返回 `{pp_dim: device_stage}` 不变。
- `main.py` 的 CLI 验证：`num_stacks % (v * pp) == 0`。
- `GraphDistributer` 无改动（已经按 bucket 处理）。
- 风险：`graph_distributer.py:54-91` 的跨 bucket shadow tensor 生成需要验证在非连续 chunk 下仍正确；写一个 `pp=4, v=2, layer=8` 的回归测试。

**改动 2：STG 后处理——注入 1F1B ctrl_deps**
- 文件：`dnn_workload/symbolic_tensor_graph/chakra/convert_chakra.py`（复用已存在但被禁用的 `add_ctrl_deps()` 骨架，lines 527-539）
- 加新 post‑process 模块 `pipeline_schedule.py`：
  - 输入：Chakra graph、`schedule ∈ {"gpipe","1f1b","1f1b-interleaved"}`、$p$、$v$、$m$（mb 数）
  - 按 mb 前缀 `mb{i}` 识别每 mb 的 F / B 节点边界（复用 `MicroBatchReplicatorPostProcess` 的命名约定）
  - 对每个 rank，按选定的调度插入 `ctrl_deps`：
    - **GPipe**：所有 F 先于所有 B（稳态就是论文图 3）
    - **1F1B**：warm‑up 阶段按 $p - \text{rank}$ 份 F；稳态 1F1B 交织；cool‑down 同步倒排
    - **1F1B‑interleaved**：在 chunk 粒度上做 1F1B（参考论文图 4 底部）
- 调用点：`main.py` `_postprocess_chakra_graph` 里 `MicroBatchReplicator` 之后。
- 新增 CLI 参数：`--pipeline_schedule {natural,gpipe,1f1b,1f1b-interleaved}`（默认 `natural`，保持旧行为兼容）。

**改动 3：astra‑sim 侧（可选）——在 `issue_dep_free_nodes` 加 priority**
- 只有在上述 ctrl_deps 无法完全约束时才需要。优先走 "ctrl_deps 烘焙进 ET" 路线；留 `Workload.cc` 作为备选。

**改动 4：`qwen_32b.sh` 暴露参数**
```bash
PP_SCHEDULE=${PP_SCHEDULE:-natural}
PP_VIRTUAL=${PP_VIRTUAL:-1}
# ... 传入 main.py
```

**验证**：
- 单元：新写 `test_cases/test_pipeline_schedule.py`，对 `(p=4, v=1, m=8)` 的 GPipe/1F1B/natural 分别检查 ET 中 node→node 依赖数，对照论文图 3/4 的理论期望。
- 端到端：生成 `(DP=1, TP=1, PP=4, v∈{1,2}, m=8)` 的 workload，跑 analytical 仿真，读 `run.log`，bubble 占比应接近 $(p-1)/(v\cdot m)$。

---

### Phase 1

#### P1‑A scatter / gather 通信优化

**目标**：相邻 pipeline stage 间跨节点 P2P 按 TP 度 $t$ 拆成 $1/t$ 发送 + NVLink all‑gather。

**改动**：`dnn_workload/symbolic_tensor_graph/chakra/convert_chakra.py:460-493`
- 在 `_insert_send_node / _insert_recv_node` 旁新增 `_insert_scatter_gather_boundary(tensor, remote_rank, shadow_rank, symbol_map_value)`：
  - 通过 `remote_readable_rank` / `shadow_readable_rank` tuple 拿到 `(tp_dim, tp_rank)`；判断是否跨 pp stage 且 TP 度 > 1
  - 若是：
    1. 每个 tp_rank 只发 `comm_size / t` 字节的 P2P（自己的那份）
    2. 接收端插入一个 TP 组内 `COLL_COMM_NODE(all_gather, size=comm_size)` 作为 RECV 之后的依赖
- 开关：`--scatter_gather_optimization true|false`（默认 true 对齐 Megatron 论文）。
- 对 `GraphDistributer` 中的 shadow tensor 生成无需改动，只改 Chakra 层的节点尺寸与插入 all‑gather。

**验证**：
- 基线 vs 开启 scatter/gather 的 analytical 仿真对比，IB 链路总字节数应下降约 $(t-1)/t$。
- 对 `(p=2, t=8)` 的 175B/91B 小规模配置，实测应接近论文图 18 的相对加速曲线。

#### P1‑B activation recomputation 打通

**目标**：`--activation_recompute true` 时，backward 前多插一次 forward 节点、对应显存下降。

**改动**：
1. `dnn_workload/symbolic_tensor_graph/main.py`：把 `args.activation_recompute` 传给 `GradUpdater.apply(..., activation_recompute=...)`。
2. `dnn_workload/symbolic_tensor_graph/graph/grad_updater.py`：
   - `GradUpdater.apply` 增加 `activation_recompute=False` 参数；若为真，对每个 transformer block 的 backward 节点族注入一个复制的 forward 子图作为依赖前置（只为该 stage 保留输入激活）。
   - FLOP / 显存统计需要据此更新；`vram_counting.py` 里减去每层中间激活的计数。
3. `dnn_workload/symbolic_tensor_graph/vram_counting.py`：新增 `activation_recompute` 分支，输出 `A_input + A_intermediate / c` 样式的显存估计。
4. `qwen_32b.sh` 暴露 `ACTIVATION_RECOMPUTE=${ACTIVATION_RECOMPUTE:-false}`。

**验证**：
- 用 `--print_gpu_vram` 对比 on/off 的激活显存，差异应符合第 3.5 节的 $c \cdot A_\text{input} + l/c \cdot A_\text{intermediate}$ 公式（默认 $c = 1$）。
- 仿真 throughput：大 batch 场景下开启应反超不开启（论文图 17 效应）。

---

### Phase 2

#### P2‑A 仿真报告增强（bubble time + 有效二分带宽）

**目标**：直接从仿真日志读到 bubble fraction 与 P2P/all‑reduce 的 effective bisection bandwidth。

**改动**：
1. `astra-sim/astra-sim/workload/Statistics.cc`：
   - `OperatorStatistics` 结构中加 `idle_ns` 字段。
   - `post_processing()`（行 268‑291）中汇总 `idle_ns / total_ns`。
   - `report()`（行 144‑215）中新增一行：`bubble_fraction = idle_ns / total_ns`。
2. `astra-sim/astra-sim/workload/Workload.cc:147-164` — 在 `issue_dep_free_nodes()` 发现本 rank 无任何 dep‑free 且无 in‑flight op 时，累计 `idle_ns`。
3. 新增 `CommBandwidthStats`（小类，附着到 `Sys.cc`）：记录每个 collective 完成时的 `bytes / duration`，按 collective type 聚合，结束时打印 `p2p_bisection_GBps` / `allreduce_bisection_GBps`。可以挂接在 `astra-sim/system/astraccl/` 层 collective 完成事件上。
4. `analytical.sh` 日志输出格式无需改动，新增字段会直接进 `run_analytical.log`。

**验证**：
- 跑一个已知 $(p, m, v)$ 的配置，打印的 bubble fraction 应接近公式 $(p-1)/(v\cdot m)$，误差 <2 个百分点。
- `p2p_bisection_GBps` 应接近 `topology.txt` 里 inter‑node 链路带宽 × 并发度。

#### P2‑B 每 op‑type 的 achievable TFLOP/s

**目标**：对不同 op 类型（GEMM、element‑wise、softmax、attention 等）给不同的可达峰值，近似 kernel 融合与 memory‑bound 的差异。

**改动**：
1. `astra-sim/astra-sim/system/Roofline.{cc,hh}`：
   - `get_perf(OI)` → `get_perf(OI, op_type=0)`
   - 成员加 `unordered_map<int, double> peak_per_op_type_`
2. `astra-sim/astra-sim/system/Sys.cc:404-415`：解析新 JSON 字段 `"peak-perf-per-op-type": {...}`（可选；不填时退回统一 `peak-perf`）。
3. `astra-sim/astra-sim/workload/Workload.cc:276-279`：调用 `roofline->get_perf(OI, node->get_op_type())`。
4. `astra-sim/qwen_experiment/in_dc/astra_system.json`：示例
   ```json
   "peak-perf": 312,
   "peak-perf-per-op-type": {
     "GEMM": 312,
     "ELEMWISE": 90,
     "SOFTMAX": 60,
     "REDUCE": 40
   }
   ```
5. STG 侧在 `convert_chakra.py` 写入 COMP_NODE 时标注 `op_type` 属性（已有部分标注，补齐即可）。

**验证**：
- 对 GPT‑3 175B、96 卡配置，开启 per‑op‑type 峰值后单卡 TFLOP/s 应比统一 peak 下降合理比例；与论文表 1 数值做趋势比对。

---

### Phase 3

#### P3‑A VRAM 上限 + 峰值上报

**目标**：按 80 GB A100 设上限，若任一 rank 峰值超限则 WARN；日志里直接打印 per‑rank peak VRAM。

**改动**：
1. `astra-sim/qwen_experiment/in_dc/astra_system.json` 增 `"track-local-mem": true`, `"vram-capacity-gb": 80`。
2. `astra-sim/astra-sim/workload/LocalMemUsageTracker.cc`：`buildMemoryTimeline()` 行 263‑315，比较峰值与 cap，违反则在 `report()` 中打印警告。
3. `Sys.cc` 的 destructor 结束前调用 `getPeakMemUsageFormatted()` 汇总输出。

**验证**：
- 已知过大的 $(t=1, p=1, B=\text{large})$ 应触发 OOM warn。
- 和 `--print_gpu_vram` 的 STG 侧静态估算交叉校验，误差 <20%。

#### P3‑B 8× IB NIC 多链路

**目标**：让 topology.txt 在每对相邻 leaf 之间铺 8 条平行 IB 链路（对应 DGX A100 的 8 张 HCA）。

**改动**：
- 只动 `astra-sim/qwen_experiment/in_dc/generate_topology.py`：
  - 加 `--nics-per-gpu 8`（默认 1 保持现状）
  - `LinkSpec` 的 GPU↔NIC 段循环铺 `nics_per_gpu` 条平行链路；对应下游 leaf 段链路也按比例加。
- 格式层已支持，无需改 ASTRA‑sim 后端。
- 配合 P1‑A 的 scatter/gather，才能观察到 8 NIC 被吃满的 892 GB/s 有效带宽。

**验证**：`p2p_bisection_GBps`（来自 P2‑A）应随 `--nics-per-gpu` 线性增长。

---

## 关键文件清单（修改点汇总）

| 文件 | 改动 | 阶段 |
|------|------|------|
| `dnn_workload/symbolic_tensor_graph/main.py` | 加 CLI：`--pipeline_virtual_stages`, `--pipeline_schedule`, `--scatter_gather_optimization`, `--activation_recompute` 接入 | P0/P1 |
| `dnn_workload/symbolic_tensor_graph/main.py:72-117` `_create_pipeline_tensor_map[_mix_precision]` | 支持 $v>1$ 非连续映射 | P0‑B |
| `dnn_workload/symbolic_tensor_graph/chakra/convert_chakra.py` 新增 `pipeline_schedule.py` post‑process | 注入 1F1B / GPipe ctrl_deps | P0‑A |
| `dnn_workload/symbolic_tensor_graph/chakra/convert_chakra.py:460-493` | P2P 送前 scatter、收后 all‑gather | P1‑A |
| `dnn_workload/symbolic_tensor_graph/graph/grad_updater.py` `GradUpdater.apply` | 接入 activation_recompute | P1‑B |
| `dnn_workload/symbolic_tensor_graph/vram_counting.py` | 激活重算显存分支 | P1‑B |
| `dnn_workload/qwen_32b/qwen_32b.sh` | 暴露新 env var | P0/P1 |
| `astra-sim/astra-sim/workload/Statistics.cc:144-215, 268-291` | bubble_fraction 字段 | P2‑A |
| `astra-sim/astra-sim/workload/Workload.cc:147-164, 276-279` | 统计 idle_ns；传 op_type 给 roofline | P2‑A/B |
| `astra-sim/astra-sim/system/Roofline.{cc,hh}` | per op‑type peak | P2‑B |
| `astra-sim/astra-sim/system/Sys.cc:404-415` | 解析 `peak-perf-per-op-type`；收尾 peak VRAM 汇总 | P2‑B / P3‑A |
| `astra-sim/astra-sim/workload/LocalMemUsageTracker.cc:263-502` | VRAM cap 检查 | P3‑A |
| `astra-sim/qwen_experiment/in_dc/astra_system.json` | 开 `track-local-mem`, `vram-capacity-gb`, `peak-perf-per-op-type` | P2‑B/P3‑A |
| `astra-sim/qwen_experiment/in_dc/generate_topology.py` | `--nics-per-gpu` | P3‑B |

## 可复用的已有实现

- `convert_chakra.py:527-539` 的 `add_ctrl_deps()` 残桩——直接复活作为 1F1B 注入入口。
- `grad_updater.py:364+` `MicroBatchReplicatorPostProcess` 的 `mb{i}` 命名约定——作为 F/B 边界识别器。
- `LocalMemUsageTracker`（`astra-sim/astra-sim/workload/LocalMemUsageTracker.cc`）已完整实现 — 只要在 `astra_system.json` 开 `track-local-mem: true` 就能用。
- `Roofline` 类 `astra-sim/astra-sim/system/Roofline.{cc,hh}`——只需扩方法签名，不用重写。
- `CustomTopology.cpp` 已支持异构链路 / 平行链路——P3‑B 的多 NIC 只需上层生成器出文件即可。
- `qwen_experiment/in_dc/generate_topology.py` 的 `LinkSpec` 数据类——直接扩参数就能长出 8 NIC。

## 端到端验证（贯穿所有阶段）

1. **基线固化**：冻结当前 `(DP=4, TP=8, PP=4, LAYER=4, MICROBATCH=2, BATCH=128)` 的 `run_analytical.log`，作为回归 baseline。
2. **Phase 0 完成后**：
   - 跑 $v \in \{1, 2, 4\}$、$m \in \{4, 8, 16\}$ 的 sweep，绘制 bubble fraction 曲线，应与 $(p-1)/(v\cdot m)$ 吻合。
   - GPipe/1F1B 峰值激活内存差异（来自 P1‑B 的 VRAM 报告交叉验证）。
3. **Phase 1 完成后**：
   - scatter/gather on/off 对比，IB 字节下降 $(t-1)/t$。
   - 激活重算 on/off 在大 batch 下的相对吞吐，对照论文图 17。
4. **Phase 2/3 完成后**：
   - 复跑表 1 的 10 种配置中至少 3 种（1.7B/18.4B/145.6B），对比仿真预测 TFLOP/s 与论文数值的相对偏差；目标 <20%。
   - 观察 `p2p_bisection_GBps` 随 `--nics-per-gpu` 线性扩展。

## 工期估算（粗略）

| 阶段 | 子项 | 估计工作量 |
|------|------|------------|
| P0‑A | 1F1B / GPipe ctrl_deps 注入 | 3–5 人日 |
| P0‑B | interleaved 非连续映射 + 回归 | 1–2 人日 |
| P1‑A | scatter/gather 重写 P2P 节点 | 2–3 人日 |
| P1‑B | activation_recompute 打通 + VRAM 校核 | 2–3 人日 |
| P2‑A | bubble + 带宽统计 | 2 人日 |
| P2‑B | per‑op‑type roofline | 1–2 人日 |
| P3‑A | VRAM cap / 报告 | 0.5 人日 |
| P3‑B | multi‑NIC 拓扑生成 | 0.5 人日 |
| **合计** | — | **12–19 人日** |
