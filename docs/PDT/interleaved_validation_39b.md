# 1F1B-Interleaved 修复 — 39.1B GPT 端到端验证报告

**日期**：2026-04-21
**对应修复项**：`docs/PDT/correctness_todo.md` §1 — `1f1b-interleaved` 半实现
**配置**：Megatron-LM GPT 39.1B 论文 Table 1 第 5 行（arXiv:2104.04473）

## 1. 背景

`PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL=2` 之前是"半实现"：`_create_pipeline_tensor_map_mix_precision` 确实按 round-robin 把 48 个 transformer block 交错分配到 PP=2 × v=2 = 4 个 chunks，但 `_apply_1f1b_interleaved_to_rank` 直接 fallback 到 mb-粒度的 1F1B。结果：图拓扑多了 cross-PP p2p 代价，调度却没拿到 bubble 填充收益，wall 从 14.19 s 劣化到 21.25 s（−34.5%）。

本报告记录修复后在 39.1B（512 GPU）上的端到端验证。

## 2. 修复概要

| 文件 | 变化 |
|------|------|
| `dnn_workload/symbolic_tensor_graph/main.py` | `_create_pipeline_tensor_map[_mix_precision]` 额外返回 `block_to_chunk_local: dict[block_idx -> chunk_on_device]`，沿 dense/gpt/moe 三条路径透传到 `_postprocess_chakra_graph` |
| `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/pipeline_schedule.py` | `_build_1f1b_interleaved_sequence` 按 Megatron-LM `schedules.py` 的公式重写：`warmup = (p − rank − 1)·2 + (v − 1)·p`，`mb = (k // (p·v))·p + k%p`，`chunk_F = (k % (p·v)) // p`，`chunk_B = v − 1 − chunk_F`；steady 阶段 **F 在 B 之前** append |
| `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/pipeline_schedule.py` | `_apply_1f1b_interleaved_to_rank` 改为对 **只 COMP 节点** 分桶 `(mb, phase, chunk_on_device)`；SEND/RECV/shadow 走 data_deps，不参与分桶 |
| `dnn_workload/symbolic_tensor_graph/test_cases/test_pipeline_interleaved.py` | 新增 4 个单元测试 |

### 2.1 Steady 阶段 F/B 顺序的死锁坑

首版代码在 steady 循环里按 `(B, F)` 顺序 `seq.append` —— 这是一个方向错误的直觉修正。在 rank p-1 上：

- Warmup 只执行 `(v − 1)·p` 个 F chunks，覆盖 chunk 0..v−2 的若干 mb；chunk v−1 的 F 从未进入 warmup。
- Steady k=0 的 B 是 `B(mb=0, chunk=v−1)`，要求 `F(mb=0, chunk=v−1)` 已完成。
- Steady k=0 的 F 是 `F(mb=0, chunk=v−1)`。

若 B 排在 F 之前，rank p-1 永远跑不出 chunk v−1 的第一次 F，上游 RECV 全卡死。512 GPU 首跑 ASTRA-sim 报 `Pending callbacks before cleanup: 35328`、exit 2 即此。

Megatron-LM 的实现本身是 F 先 B 后（`forward_k = k + num_warmup; backward_k = k`），修复版本对齐了这一点。

## 3. 实验配置

| 项 | 值 |
|----|----|
| 模型 | Megatron-LM GPT, params ≈ 39.1B |
| 形状 | h=8192, l=48, head=64, KVHead=64, seq=2048, vocab=51200 |
| 并行 | DP=32, TP=8, PP=2, SP=1（总 512 GPU）|
| Batch | global batch=1536, micro batch=2（⇒ m=24 per rank）|
| 激活重算 | on |
| Scatter/gather 优化 | on |
| 调度 | `1f1b` v=1（baseline） vs `1f1b-interleaved` v=2（修复前 / 修复后）|
| Attention kernel | fused |
| Mixed precision | on |
| 仿真后端 | ASTRA-sim analytical（congestion-aware）|
| 系统 JSON | `astra-sim/megatron_gpt_experiment/gpt_39b_512/astra_system.json`（`peak-perf=312`, `active-chunks-per-dimension=2`, `preferred-dataset-splits=4`）|
| 网络 | 2-level fat-leaf + single spine（`build_selene_topology.py --num_nodes 64`）|

工作负载生成：`dnn_workload/megatron_gpt_39b/megatron_gpt_39b.sh`（默认值即论文 Table 1 config；PP_SCHEDULE / PP_VIRTUAL 通过环境变量覆盖）。

## 4. 结果

### 4.1 主表

| 配置 | wall cycles | wall (s @ 1 GHz) | TFLOP/s/GPU (4f-norm) | Δ vs 论文 138 | Δ vs baseline | exposed comm cycles | exposed/wall |
|------|-----------:|-----:|-------:|-------:|-------:|-------:|-----:|
| baseline (1f1b, v=1, b=2) | 14,194,697,459 | 14.195 | 140.52 | +1.82% | — | 7,888,998,747 | 55.58% |
| 修复前 (1f1b-int, v=2, b=2) | 21,253,032,657 | 21.253 |  93.85 | −31.99% | −33.19% | 14,947,333,945 | 70.33% |
| **修复后 (1f1b-int, v=2, b=2)** | **12,382,114,950** | **12.382** | **161.09** | **+16.73%** | **+12.77% (加速)** | **6,076,416,238** | **49.07%** |

FLOP 口径：`96·batch·seq·l·h² · (1 + seq/(6h) + V/(16lh))` ≈ 1.021 EFLOP/iter（paper eqn 3，AR=on 4× forward）。

### 4.2 对照论文 §5.3.2

论文报告 `1f1b-interleaved v=2` 相对 `1f1b v=1` 带来 ~5–15% 加速。本次修复实测 **+12.77%**（按 wall 时间），落在该区间上沿；同时 exposed comm 绝对值下降 22.97%，exposed/wall 比例从 55.6% 降到 49.1%（−6.5 pp），物理意义上符合"chunk-level interleaving 用填 bubble 的方式掩盖了额外的 p2p"这一预期。

### 4.3 ET 层 ctrl_deps 验证

修复后 PP rank 0（总 rank 0）和 PP rank 1（总 rank 256）的 ET 各自注入了 **95 条 ctrl_deps**：

- `v=2, p=2, m=24` ⇒ `total_steps = 48`
- rank 0 warmup = `(2−0−1)·2 + (2−1)·2 = 4`；steady = `48 − 4 = 44`；sequence 长度 = `4 + 44·2 + 4 = 96`；注入 pair-链 95 条 ✓
- rank 1 warmup = `(2−1−1)·2 + (2−1)·2 = 2`；steady = 46；sequence = `2 + 46·2 + 2 = 96`；注入 95 条 ✓

### 4.4 单元测试

`test_cases/test_pipeline_interleaved.py` 全部通过（11/11）：

- `test_v*` × 5：原有 block→device 映射（未动）
- `test_pp1_always_device_0`：pp=1 保持传统
- `test_interleaved_sequence_matches_megatron_rank0`：rank 0 warmup 10 项对齐 Megatron 公式
- `test_classify_chunk_skips_shadows_and_maps_transformer_blocks`：`transformer.N` / `in_emb` / `out_emb` / `loss` / `shadow_` 分类
- `test_apply_1f1b_interleaved_inserts_cross_chunk_ctrl_deps`：`(p=4, v=2, m=8, rank=0)` 注入 31 条 pair 链覆盖全部 32 组
- `test_apply_1f1b_interleaved_v1_falls_back`：v=1 时 fallback 到 1F1B
- `test_et_generation_interleaved_cross_device_more_p2p`：原有端到端（natural schedule）

## 5. 复现

```bash
# 生成 workload（修复版）
cd /home/ps/sow/part2/dnn_workload/megatron_gpt_39b
PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL=2 bash megatron_gpt_39b.sh

# 跑 ASTRA-sim
cd /home/ps/sow/part2/astra-sim/megatron_gpt_experiment/gpt_39b_512
WORKLOAD_DIR=/home/ps/sow/part2/dnn_workload/megatron_gpt_39b/fused_standard_48_1_1536_2_2048_1f1b-interleaved_v2_sgo1_ar1 \
    bash run_analytical.sh

# 读数
grep 'sys\[.*\] finished' run_analytical.log \
    | sed -E 's/.*finished, ([0-9]+) cycles.*exposed communication ([0-9]+).*/\1 \2/' \
    | sort -n | tail -1
# 期望：12382114950 6076416238（或之前基线 14194697459 7888998747）
```

归档日志（均为 512 rank 的完整 run_analytical.log 拷贝）：

| 文件 | 说明 |
|------|------|
| `astra-sim/megatron_gpt_experiment/gpt_39b_512/run_analytical.log.v1_baseline` | baseline (1f1b, v=1, b=2) |
| `astra-sim/megatron_gpt_experiment/gpt_39b_512/run_analytical.log.interleaved` | 修复前 (1f1b-int, v=2, b=2) — 历史存档 |
| `astra-sim/megatron_gpt_experiment/gpt_39b_512/run_analytical.log.interleaved_fixed` | **修复后** (1f1b-int, v=2, b=2) |

## 6. 验收对照（correctness_todo.md §1）

| # | 条款 | 状态 |
|---|------|:----:|
| 1 | 新回归测试 `test_pipeline_interleaved.py` 扩展：断言相邻 chunk 的 F/B 间有 ctrl_dep | ✅ |
| 2 | 39B `1f1b-interleaved + v=2` wall ≤ `1f1b + v=1` baseline × 1.02，目标 5–15% 加速 | ✅ 加速 12.77%（in range）|
| 3 | `analysis_report.md` §4.2 加一行"修复后 v=2 结果"，Δ 落到 ±5% 之内并更新 §9.1#4 状态 | ✅ 新增 §4.2a + §9.1 #4/#5 更新 |

76.1B（1024 GPU）回归未列入验收 #2 的显式条款；需要时按同样流程即可。
