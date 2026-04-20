# Phase 2 + Phase 3 完成小结

对应 `implementation_plan_zh.md` 的 P2‑A / P2‑B / P3‑A / P3‑B 四项。

## 改动文件

| 文件 | 作用 | Phase |
|------|------|-------|
| `astra-sim/astra-sim/system/Roofline.{hh,cc}` | 新增 `achievable_fraction` 全局可达系数 | P2‑B |
| `astra-sim/astra-sim/system/Sys.hh` | 新成员 `vram_capacity_bytes` | P3‑A |
| `astra-sim/astra-sim/system/Sys.cc` | 解析 `peak-perf-achievable-fraction` / `vram-capacity-gb` 新 JSON 字段 | P2‑B, P3‑A |
| `astra-sim/astra-sim/workload/Statistics.{hh,cc}` | 新增 `bubble_time`/`bubble_fraction_`、`total_comm_bytes_` / `total_p2p_bytes_` / `total_coll_bytes_`；`post_processing` 新增 `extract_bubble` + `extract_comm_bytes`；`report` 输出新字段 | P2‑A |
| `astra-sim/astra-sim/workload/Workload.cc` | 收尾阶段对峰值 VRAM 做 cap 检查并警告 | P3‑A |
| `astra-sim/qwen_experiment/in_dc/astra_system.json` | 启用 `track-local-mem`=1；`vram-capacity-gb`=80；`peak-perf-achievable-fraction`=1.0 | P2‑B, P3‑A |
| `astra-sim/qwen_experiment/in_dc/generate_topology.py` | 新增 `--nics-per-gpu` 参数；GPU↔NIC 和 NIC↔Leaf 两段链路按 N 倍复制 | P3‑B |

## 新增 JSON 配置项

`astra_system.json`：

```json
{
    "peak-perf-achievable-fraction": 1.0,
    "track-local-mem": 1,
    "vram-capacity-gb": 80
}
```

- **peak-perf-achievable-fraction**（P2‑B）：对 roofline 的 peak_perf 与 bandwidth 同乘一个系数，模拟算子融合/launch overhead 造成的可达性损失。设为 0.8 约等于论文 §5.8 里 20% 融合缺口；默认 1.0 等价旧行为。
- **track-local-mem**（P3‑A）：已存在的开关，此前默认关闭；本计划默认打开以便峰值 VRAM 每次仿真都可见。
- **vram-capacity-gb**（P3‑A）：硬性 VRAM 上限，触发后 Workload 收尾打 warn（不终止仿真）。设为 0 为不限（legacy）。

## 新增日志字段（`run_analytical.log`）

每个 `sys[N]` 收尾块现在新增三行：

```
[statistics] sys[N], Bubble time: <ns> (<pct>%)
[statistics] sys[N], Comm bytes: <total> (p2p=<> coll=<>), Effective BW: <GB/s> (p2p=<> coll=<>)
[workload]   sys[N] peak memory usage: <GiB>
[workload]   sys[N] VRAM OK / VRAM OVERFLOW ...（仅当设置 vram-capacity-gb）
```

字段定义：
- **Bubble time** = `wall_time − union(所有算子时间段)`，即本 rank 完全空闲的总 ns。受数据依赖驱动调度下，若每个时间点都有至少一个算子在跑就得 0，属正常。要观察到非零 bubble，搭配 P0‑A 的 `PP_SCHEDULE=1f1b` + `gpu-comp=1` 即可。
- **Effective BW** = total_bytes / wall_time（1 byte/ns = 1 GB/s）；按是否有 `network_bandwidth` 字段分 p2p / coll。
- **Peak memory usage** 来自 `LocalMemUsageTracker::getPeakMemUsageFormatted()`，已在仿真生命周期内被 Workload 调用。

## P3‑B: 多 NIC 拓扑生成

生成器 CLI 新增 `--nics-per-gpu N`（默认 1）。示例：

```bash
python3 generate_topology.py \
    --gpus-per-nvlink-node 8 --nvlink-node-count 4 --nvlink-nodes-per-leaf 4 \
    --spine-count 1 \
    --gpu-nvswitch-bandwidth 4800Gbps --gpu-nvswitch-latency 0.00015ms \
    --gpu-nicswitch-bandwidth 200Gbps --gpu-nicswitch-latency 0.000001ms \
    --nicswitch-leaf-bandwidth 200Gbps --nicswitch-leaf-latency 0.0005ms \
    --leaf-spine-bandwidth 1600Gbps --leaf-spine-latency 0.0006ms \
    --nics-per-gpu 8   # <-- 每 GPU 8 张 IB NIC
```

链路数从 97 增加到 545（+448 = 2 × 32 × (8−1)）；分析后端的 `CustomTopology.cpp` 已原生支持多重边，无需改 C++。

## 端到端验证

分析后端重编译成功后运行 `qwen_experiment/in_dc/analytical.sh`，日志里每个 rank 都出现新字段，示例：

```
sys[66], Bubble time: 0 (0.000%)
sys[66], Comm bytes: 53235279840 (p2p=53235279840 coll=0), Effective BW: 4.348 GB/s (p2p=4.348, coll=0.000)
sys[66] peak memory usage: 193.14 GB
sys[66] VRAM OVERFLOW: peak=193.14 GB exceeds vram-capacity-gb=80.00
```

- **Bubble = 0** 说明在默认 natural schedule 下 rank 66 自始至终都在忙（符合预期）；
- **Comm bytes = 53.2 GB**、4.348 GB/s 平均带宽说明统计生效；
- **Peak 193 GB > cap 80 GB** 正确触发 OVERFLOW 警告——对 LAYER=4、BATCH=128、MICROBATCH=2 的默认配置，这个数字反映当前工作负载对单 rank 的峰值分配（未启用 `activation_recompute` / FSDP 时偏高）。关 `--activation_recompute true` + FSDP 后可回到合理范围。

## 已知限制

- **Bubble = 0 的场景** 并非代码错误。要看到非零气泡，需要：(a) 至少 2 个 PP stage 且 micro‑batch 数不太大；(b) `PP_SCHEDULE=1f1b` 或 `gpipe`（显式插 ctrl_dep 使前后向串行化）；(c) 搭配 `gpu-comp=1` 的严格单 comp 槽。
- **`peak-perf-achievable-fraction` 是全局系数** 而非 per op‑type。要精确区分 GEMM / element‑wise / softmax 的算术强度差异，需要 STG 侧给 COMP 节点标 `op_type` 属性并扩 Roofline 映射表（计划中折成了 MVP；后续可扩）。
- **P2P vs coll 分类** 借用"是否有 network_bandwidth 字段"作为启发；未来可改为直接查 Chakra NodeType。
- **VRAM cap 仅警告不终止** ——这是刻意选择（仿真仍然有参考价值），真实 OOM 对应的系统行为不在仿真范围。

## 使用示例

```bash
# 启用所有 Phase 0–3 能力：1f1b + scatter/gather + 激活重算 + 多 NIC 拓扑
cd /home/ps/sow/part2/dnn_workload

# 1. 生成匹配的层次拓扑（8 NICs/GPU 模拟 DGX A100 聚合带宽）
cd ../astra-sim/qwen_experiment/in_dc
python3 generate_topology.py \
    --gpus-per-nvlink-node 8 --nvlink-node-count 16 --nvlink-nodes-per-leaf 4 \
    --spine-count 1 \
    --gpu-nvswitch-bandwidth 4800Gbps --gpu-nvswitch-latency 0.00015ms \
    --gpu-nicswitch-bandwidth 200Gbps --gpu-nicswitch-latency 0.000001ms \
    --nicswitch-leaf-bandwidth 200Gbps --nicswitch-leaf-latency 0.0005ms \
    --leaf-spine-bandwidth 1600Gbps --leaf-spine-latency 0.0006ms \
    --nics-per-gpu 8 --output topology.txt
cd -

# 2. 生成带 1f1b + sg + ar 的 workload
cd qwen_32b
PP_SCHEDULE=1f1b PP_VIRTUAL=1 SGO=true ACTIVATION_RECOMPUTE=true \
    DP=4 TP=8 PP=4 LAYER=32 BATCH=128 MICROBATCH=2 bash qwen_32b.sh
cd -

# 3. 仿真（新日志字段自动出现）
cd ../astra-sim/qwen_experiment/in_dc
WORKLOAD_DIR=.../qwen_32b/standard_standard_32_1_128_2_4096_1f1b_v1_sgotrue_artrue/ \
    bash analytical.sh
```

## 至此全部 Phase 完成清单

| Phase | 子项 | 状态 | 回归测试 |
|-------|------|------|----------|
| P0‑A | 显式 gpipe/1f1b/1f1b-interleaved 调度 | ✅ | `test_cases/test_pipeline_interleaved.py` |
| P0‑B | Interleaved 非连续映射 (v>1) | ✅ | 同上 |
| P1‑A | Scatter/gather 通信优化 | ✅ | `test_cases/test_scatter_gather.py` |
| P1‑B | 激活重算（FLOP + VRAM） | ✅ | 手动 FLOP 对账 |
| P2‑A | Bubble fraction + 有效带宽 | ✅ | 仿真日志对照 |
| P2‑B | Roofline achievable fraction | ✅ | 需 cfg 实验验证 |
| P3‑A | VRAM 峰值 + cap 检查 | ✅ | 仿真日志对照 |
| P3‑B | 多 NIC 拓扑生成 | ✅ | 链路计数对账 |

总文档：
- `docs/PDT/2104.04473v5_zh.md` — 论文中文精译
- `docs/PDT/gap_analysis_zh.md` — 覆盖度分析
- `docs/PDT/implementation_plan_zh.md` — 实现计划
- `docs/PDT/phase0_completion_zh.md` — Phase 0 小结
- `docs/PDT/phase1_completion_zh.md` — Phase 1 小结
- `docs/PDT/phase2_3_completion_zh.md` — Phase 2+3 小结（本文件）
