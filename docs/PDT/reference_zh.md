# 配置参考

Phase 0–3 新增的所有 CLI 参数、环境变量与 JSON 字段汇总表。旧配置全部保持向后兼容——不显式设置这些新项，行为等同 Phase 0 前的 baseline。

## 1. STG CLI 参数（`symbolic_tensor_graph/main.py`）

| 参数 | 类型 | 默认 | Phase | 说明 |
|------|------|------|-------|------|
| `--pipeline_virtual_stages` | int | 1 | P0‑B | 每个设备的虚拟 pipeline stage 数 $v$（model chunks/device）。$v>1$ 时启用 Megatron 交错式，要求 `num_stacks % (v*pp) == 0` |
| `--pipeline_schedule` | choice | `natural` | P0‑A | 流水调度策略：`natural` / `gpipe` / `1f1b` / `1f1b-interleaved` |
| `--scatter_gather_optimization` | bool | false | P1‑A | 启用 Megatron §4.1 scatter/gather。每个 TP rank 只发 1/t 跨 PP 片段 + NVLink all-gather |
| `--activation_recompute` | bool | false | P1‑B | 启用 §3.5 激活重算。对 transformer block 的 backward COMP 抬升一次 forward 的 FLOP；`--print_gpu_vram` 报告中 AR=on 时仅输出 acts 上界并指向 ASTRA-sim `peak memory usage`（权威峰值，见 correctness_todo.md §3） |

已存在但本次重点使用的参数（完整清单见 `main.py --help`）：
- `--dp / --tp / --pp / --sp`（四种并行度）
- `--batch`（全局 batch）/ `--micro_batch`（每 rank micro-batch）
- `--num_stacks`（transformer 块数）
- `--mixed_precision`（通常 true）
- `--model_type llama | gpt | moe`
- `--print_gpu_vram`（开启 VRAM 粗估）

## 2. `qwen_32b/qwen_32b.sh` 环境变量

| env | 默认 | 透传到 |
|-----|------|--------|
| `DP` / `TP` / `PP` / `SP` | 4 / 8 / 4 / 1 | 四个并行度 |
| `LAYER` | 4 | `--num_stacks` |
| `BATCH` | 128 | `--batch` |
| `MICROBATCH` | 2 | `--micro_batch` |
| `SEQUENCE` | 4096 | `--seq` |
| `ATTENTION` | standard | `--attention_backend` |
| `SGD` | standard | localSGD 开关（`--dp_local_sgd_interval`） |
| `ITERATION` | 1 | `--num_iterations` |
| **`PP_SCHEDULE`** | `natural` | `--pipeline_schedule` |
| **`PP_VIRTUAL`** | 1 | `--pipeline_virtual_stages` |
| **`SGO`** | false | `--scatter_gather_optimization` |
| **`ACTIVATION_RECOMPUTE`** | false | `--activation_recompute` |

粗体为 Phase 0–3 新增。输出目录名现在形如：
```
<ATTENTION>_<SGD>_<LAYER>_<ITERATION>_<BATCH>_<MICROBATCH>_<SEQUENCE>_<PP_SCHEDULE>_v<PP_VIRTUAL>_sgo<SGO>_ar<ACTIVATION_RECOMPUTE>
```

## 3. ASTRA‑sim `astra_system.json` 字段

| 字段 | 类型 | 默认 | Phase | 说明 |
|------|------|------|-------|------|
| `peak-perf` | number (TFLOP/s) | 0 | – | A100 FP16 设 312 |
| `roofline-enabled` | 0/1 | 0 | – | 必须为 1 才启用 Roofline |
| `peak-perf-achievable-fraction` | double | 1.0 | P2‑B | 峰值 peak 与 bw 同乘此系数（**全局 fallback**）。模拟算子融合/launch overhead 缺口；设 0.8 近似论文 §5.8 的 ~20% 差距 |
| `peak-perf-per-op-category` | object | – | correctness_todo §4 | 按算子类别覆写 peak。键：`GEMM / ELEMWISE / SOFTMAX / REDUCE / OTHER`（或 0..4 数字），值：TFLOP/s。未列出的类别回退 `peak-perf`。与 `peak-perf-achievable-fraction` 相乘（后者充当顶层 fallback）。STG 侧通过 Chakra COMP_NODE 的 `op_category` int32 attr 告知节点类别 |
| `track-local-mem` | 0/1 | 0 | P3‑A | 开启后 `sys[N] peak memory usage` 写入日志；本次实验默认开 |
| `vram-capacity-gb` | double | 0 | P3‑A | 硬性 VRAM 上限（GB）。0 = 无上限。超出后日志打印 `VRAM OVERFLOW` warn |
| `local-mem-trace-filename` | string | `local_mem_trace` | 已存在 | track-local-mem 的 trace 输出文件名 |

示例：
```json
{
    "peak-perf": 312,
    "roofline-enabled": 1,
    "peak-perf-achievable-fraction": 0.8,
    "track-local-mem": 1,
    "vram-capacity-gb": 80,
    "local-mem-bw": 1560,
    "all-reduce-implementation": ["ring"],
    "all-to-all-implementation": ["direct"],
    "hardware-resource-capacity": {"cpu": 1, "gpu-comp": 1, "gpu-comm": 1, "gpu-recv": 64}
}
```

## 4. 拓扑生成器（`qwen_experiment/in_dc/generate_topology.py`）

| 参数 | 类型 | 默认 | Phase | 说明 |
|------|------|------|-------|------|
| `--gpus-per-nvlink-node` | int | 必填 | – | 单 NVSwitch 下挂 GPU 数 |
| `--nvlink-node-count` | int | 必填 | – | NVLink 节点数 |
| `--nvlink-nodes-per-leaf` | int | 4 | – | leaf 交换机聚合的 NVLink 节点数 |
| `--spine-count` | int | 1 | – | spine 交换机数 |
| `--gpu-nvswitch-bandwidth / latency` | str | 必填 | – | NVLink 带宽/延迟（如 `4800Gbps`, `0.00015ms`） |
| `--gpu-nicswitch-bandwidth / latency` | str | 必填 | – | GPU↔NIC 带宽/延迟 |
| `--nicswitch-leaf-bandwidth / latency` | str | 必填 | – | NIC↔leaf 带宽/延迟 |
| `--leaf-spine-bandwidth / latency` | str | 必填 | – | leaf↔spine 带宽/延迟 |
| **`--nics-per-gpu`** | int | 1 | P3‑B | 每 GPU 的平行 IB 链路数；设 8 近似 DGX A100 聚合带宽 |
| `--output` | path | `topology.txt` | – | 输出路径 |

## 5. 新增日志字段（`run_analytical.log` 每个 `sys[N]` 收尾块）

```
[statistics] sys[N], Bubble time: <ns> (<pct>%)           ← P2-A
[statistics] sys[N], Comm bytes: <total> (p2p=<> coll=<>),
             Effective BW: <GB/s> (p2p=<> coll=<>)         ← P2-A
[workload]   sys[N] peak memory usage: <GiB>               ← P3-A
[workload]   sys[N] VRAM OK | VRAM OVERFLOW: peak=... > cap=...  ← P3-A
```

## 6. 快速对照：论文 vs 新开关

| 论文要素 | 启用方式 |
|----------|----------|
| §2.2.2 interleaved schedule (v>1) | `PP_VIRTUAL=2 PP_SCHEDULE=1f1b-interleaved` |
| §2.2.1 1F1B (PipeDream‑Flush) | `PP_SCHEDULE=1f1b` |
| §2.2.1 GPipe (all F then all B) | `PP_SCHEDULE=gpipe` |
| §3.5 激活重算 | `ACTIVATION_RECOMPUTE=true` |
| §4.1 scatter/gather | `SGO=true` |
| §5.8 算子融合 effect (粗略) | `peak-perf-achievable-fraction: 0.8`（JSON，全局 fallback） |
| §5.8 算子融合 effect (分算子) | `peak-perf-per-op-category: {GEMM:312, ELEMWISE:90, SOFTMAX:60, REDUCE:40}`（JSON） |
| §5.9 多 NIC 聚合带宽 | `generate_topology.py --nics-per-gpu 8` |
| §3.3 bubble 占比观测 | 读 `Bubble time:` 日志字段（需 `PP_SCHEDULE != natural`） |
| §5.9 有效二分带宽观测 | 读 `Effective BW:` 日志字段 |
| VRAM 峰值 vs cap | `track-local-mem: 1` + `vram-capacity-gb: 80`（JSON） |
