# 不影响实验正确性的改进项（Optional）

> **定位**：这些改动不会让仿真结果错或让用户得到错误结论，但做了能提升工程效率、对标精度、可扩展性。可以在 `correctness_todo.md` 的 P0/P1 清零后按 ROI 选做。
>
> **目标读者**：后续 Claude Code 会话。每项写成自包含的小 spec，可单独挑选。
>
> **约定**：完成后把节首 `状态: 未完成` 改为 `状态: 已完成 (commit <sha>)`。

---

## A. collective 旋钮从 JSON 改为 env 可覆盖（降 sweep 成本）

**状态**：未完成
**ROI**：高（做 collective 调参时特别有感）

### 背景

`active-chunks-per-dimension` / `preferred-dataset-splits` 等 collective 层参数目前只能改 `astra_system.json`。跑 `megatron_gpt_experiment/` 时我们有 4 个 bundle（`gpt_39b_512`、`gpt_39b_512_noar`、`gpt_76b_1024`、`gpt_76b_1024_noar`），每次扫值要手动同步 4 个文件，极易漏改（`analysis_report.md` §6 的那组 sweep 手动同步过数轮）。

### 修复方案

1. 在 `astra-sim/astra-sim/system/Sys.cc:initialize_sys` 里读 JSON 值之后，加一段用 `getenv` 覆盖的逻辑，约定 env 名 `ASTRA_SYS_<KEY_UPPER>`（如 `ASTRA_SYS_ACTIVE_CHUNKS_PER_DIMENSION=2`）。
2. 覆盖的字段至少包括：`active-chunks-per-dimension`、`preferred-dataset-splits`、`peak-perf-achievable-fraction`、`vram-capacity-gb`、`peak-perf`、`local-mem-bw`。
3. 提供 Python 侧小辅助 `astra-sim/examples/run_scripts/sweep_helper.sh`（新建），示例：
   ```bash
   for CHUNKS in 1 2 4 8; do
     ASTRA_SYS_ACTIVE_CHUNKS_PER_DIMENSION=$CHUNKS \
         bash gpt_39b_512/run_analytical.sh
   done
   ```
4. 文档：`reference_zh.md` 新增一张 "env 覆盖对照表"。

### 验收

- 同一个 `astra_system.json` 文件，不改动文件内容的前提下通过 env 改变一次仿真的行为（观察 `run_analytical.log` 里的 wall 有实质变化）。
- 新增一个测试/冒烟脚本：两次仿真 `ASTRA_SYS_ACTIVE_CHUNKS_PER_DIMENSION=1` 与 `=4` 跑 39.1B noar，wall 差 > 30%（已知数据点：20.3 s vs 11.4 s）。

---

## B. `.et` 事件流式消费以降低仿真内存上限

**状态**：未完成
**ROI**：中

### 背景

`analysis_report.md` §9.2#4：仿真 39.1B 的 `MICROBATCH=1`（每 rank 48 个 micro-batch）时单进程 RSS ≈ 29 GB，挤爆 30 GB 机器，OOM。看起来 `ETFeeder` 把每个 rank 的 ET 全部展开到内存里。

### 修复方案

1. 现状定位：`astra-sim/extern/graph_frontend/chakra/` 下的 `ETFeeder` / `FeederV3` 实现（这一子模块是 fork 来的）。首先跑 `/usr/bin/time -v` 或 `/proc/<pid>/status` 定位 peak RSS 到底是 ETFeeder 还是 Dataset / 事件队列。
2. 若确是 ETFeeder：改为 mmap 或增量 lazy-load。protobuf 的 `ParseFromCodedStream` 天然支持流式，但上层有 "一次性收全 node 图遍历邻接" 的预处理，需要改成边消费边建 index。
3. 若是事件队列或 Dataset（`astra-sim/astra-sim/system/`）在占大头：目标是 event 结构体去字段化（`Dataset.hh` 里有 bookkeeping 字段可能冗余）。
4. 先定位再动手；不要盲改。

### 验收

- 同机器跑 39.1B `MICROBATCH=1 ITERATION=1 BATCH=128` 能完成（不再 OOM）。
- 单进程 peak RSS 从 29 GB 降到 ≤ 18 GB（留 12 GB 余量允许并行两个进程）。
- 对 LAYER=4 的小 workload，改动前后仿真 wall 差异 < 1%（确认没引入回归）。

---

## C. 激活重算 FLOP 独立为 COMP 节点（观察 recompute-comm 重叠）

**状态**：未完成
**ROI**：低，除非将来要在仿真里"开启 recompute 与 send/recv 同步执行"的调度

### 背景

`phase1_completion_zh.md` §已知限制 第 2 条：现状把重算的 forward FLOP **并入** 同一 block 的 backward COMP 节点的 `num_ops`（scale by `1 + F/B`）。物理 wall clock 等价，但图里看不到独立的"recompute"节点，因此：
- 无法在 ASTRA-sim 里调度 recompute 与它上游的 collective / p2p 并发；
- 无法分拆仿真日志里 `recompute 耗时` 和 `真正的 backward 耗时`。

### 修复方案

1. `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/activation_recompute.py` 改为：对每个 `(mb, block)` 组，先从组内 forward COMP 节点**克隆一份**（新 node id、新 name 后缀 `_recompute`），`num_ops = F_total`；然后把克隆节点挂到该组第一个 backward COMP 的 `data_deps` 前面。不再 scale backward 的 num_ops。
2. `convert_chakra.py` 上游不需要改。
3. 新增守卫：仅对有至少 1 个 backward COMP 的组做克隆；对 embedding / loss 节点跳过（它们没参与论文意义上的 "per-block recompute"）。

### 验收

- 单位测试 `test_cases/test_activation_recompute.py` 扩展：AR=on 时总 COMP 节点数 = AR=off 节点数 + (num_blocks × num_mb)。
- AR=on 仿真 wall 在典型配置下（qwen_32b LAYER=32）相对老实现误差 < 1%（因为物理上就是"多一次 forward"）。
- 新的 `.et` 里可 grep 到 `_recompute` 节点名，用于论证"重算可被调度"。

---

## D. VRAM cap 从仅 warn 升级到可选 abort

**状态**：未完成
**ROI**：低

### 背景

`Workload.cc` 里 `VRAM OVERFLOW` 只打 `logger->warn`，仿真继续跑。对生产级实验定义"这个配置能不能用"时，我们仍然拿到完整 wall，这其实不对应物理世界（真实 OOM 的训练会崩）。

### 修复方案

1. `astra_system.json` 新增 `"vram-on-overflow": "warn" | "abort"`（默认 `warn`）。
2. `Sys.cc` 解析成枚举；`Workload::report()` 或更合理的触发点（VRAM 超标的第一瞬间）里按枚举决定 `logger->warn` 还是 `throw std::runtime_error`。
3. 文档 `reference_zh.md` 标注。

### 验收

- `astra_system.json` 设 `"vram-capacity-gb": 10, "vram-on-overflow": "abort"`、跑 qwen_32b 默认配置，仿真应中断且退出码非 0。
- `warn` 模式行为不变，回归测试 pass。

---

## E / F. 3-level fat-tree + ECMP / collective 等价多路径分流

**状态**：未完成
**ROI**：中（这是把 `build_selene_topology.py` 的 3-level 埋点真正利用起来的唯一途径）

### 背景

`analysis_report.md` §7：3 级 fat-tree 在当前 analytical 后端下反而变慢（20.3 s → 29.6 s），因为后端用确定性最短路径路由，多路径被白放。论文的 Selene 拓扑（三级）只有在 ECMP / adaptive routing 下才能兑现 bisection 带宽优势。

### 修复方案（分两半，合作完成才有收益）

**E（后端层）：analytical congestion-aware 支持 ECMP**

1. `astra-sim/extern/network_backend/analytical/congestion_aware/basic-topology/CustomTopology.cpp` 里查找 `shortest_path` / 路由表构建。现状是单条路径；改为 "对同源同目的有 ≥ 2 条等代价路径时，round-robin / hash 到其中一条"。
2. flow 粒度 hash（五元组模拟）：用 `flow_id` 做 hash，保证同 flow 路径稳定（避免乱序）。
3. Collective 拆 chunk 时不同 chunk 天然有不同 tag，自然可被 hash 到不同路径。

**F（collective 层）：多路径感知的 ring**

1. `astra-sim/astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.cc`：当 topology 层报告 "该 src→dst 有 k 条等价路径" 时，ring 自动把 chunk 数 `active-chunks-per-dimension` 扩到 `k × 原值`。
2. 需要 topology API 暴露 "平行度"，对应 `AstraNetworkAPI::get_parallel_paths(src, dst)`（新方法）。

### 验收

- 同一个 3-level topology.txt：
  - baseline（当前行为）：wall = 29.6 s
  - 启用 ECMP 后：wall ≤ 22 s
  - 启用 ECMP + 多路径 ring 后：wall ≤ 16 s（回到 2-level 的性能，并解锁 NIC 聚合带宽的潜力）
- `analysis_report.md` §7 表格追加 "ECMP 修复后" 列。

---

## G. 非混合精度路径 `out_emb/loss` 归属的行为兼容提示

**状态**：未完成（文档/注释任务，不改实现）
**ROI**：低

### 背景

`phase0_completion_zh.md` §已知限制 第 3 条：`_create_pipeline_tensor_map`（非 mix_precision 路径）的 `out_emb` / `loss` 归属从 `(num_stacks-1) % range_` 改为 `_block_idx_to_device(num_stacks-1, ...)`。

**这个改动是语义 fix（loss 本来就该落在最后一个 stage），不是回归**，但对**已有的 non-mix + 非整除 `num_stacks % pp != 0`** 的 workload 而言，loss 设备号会变。现有 Megatron / Qwen / Llama 驱动都用 `mixed_precision=true`，不受影响。

### 修复方案

纯文档/注释任务：

1. 在 `dnn_workload/symbolic_tensor_graph/main.py` `_create_pipeline_tensor_map` 函数上加一段注释：
   ```
   Note: this non-mixed-precision path's out_emb/loss assignment was fixed to
   use `_block_idx_to_device(last)` instead of the legacy `(num_stacks-1) % range_`.
   This matches physical semantics (loss lives on the last pipeline stage), but
   reproductions of older non-mixed-precision runs with num_stacks % pp != 0
   may see loss on a different device. See docs/PDT/phase0_completion_zh.md.
   ```
2. `reference_zh.md` 的 "行为兼容性变化" 小节追加这条。

### 验收

- 注释与文档到位；不需要功能测试。

---

## H. STG 微优化：节点 id/名称扫描的复杂度

**状态**：未完成
**ROI**：低，除非未来做超大规模（num_stacks ≥ 128 或 v ≥ 8）

### 背景

`dnn_workload/symbolic_tensor_graph/main.py:_create_pipeline_tensor_map*` 用 `re.search(r"transformer\.(\d+)", tid)` 扫每个 tensor id；`pipeline_schedule.py` 也反复正则节点名识别 F/B + mb。对 LAYER=32, v=2, DP=32 tp=8 pp=4 的 megatron-76b workload，一次生成耗时 > 5 min 里相当一部分就在 regex scan。

### 修复方案

1. STG 在 `Tensor` 创建时就把 block_idx / mb_idx / phase 存成字段，下游不再从 name 正则反解。
2. 没有 block 的 tensor（embedding/loss）走 enum 标签。
3. `convert_chakra.py` 下游消费方改查字段。

### 验收

- 生成 `megatron_gpt_76b` 的端到端时间降低 ≥ 30%。
- 所有回归测试 pass。

---

## 后续可能新增的条目

跑完 P0-A/B 的真·interleaved、per-op-type roofline、全规模 sweep 后，`analysis_report.md` 可能会暴露新的对标偏差。那些发现会补在本文件或 `correctness_todo.md` 里（新发现的"错结果"类问题归后者）。
