# docs/PDT — Megatron‑LM (PTD‑P) 仿真能力补齐

本目录围绕 Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（SC'21, arXiv:2104.04473v5）组织：**论文 → 能力差距盘点 → 实现计划 → 三阶段落地报告 → 配置/实验参考**。

## 阅读顺序

| # | 文件 | 内容 | 适用读者 |
|---|------|------|---------|
| 1 | `2104.04473v5_zh.md` | 论文中文精译（按原章节组织；含图/表/核心公式） | 需先理解论文 |
| 2 | `gap_analysis_zh.md` | 当前仿真栈（STG + ASTRA‑sim analytical）对论文 PTD‑P 的覆盖度逐项对照 | 想知道基线差哪些 |
| 3 | `implementation_plan_zh.md` | 基于 gap 分析的实现计划（8 个 P0–P3 子项、修改文件清单、工作量估算） | 想知道怎么补 |
| 4 | `phase0_completion_zh.md` | P0‑A + P0‑B 落地报告（显式流水调度 + interleaved 映射） | 具体查这组改动 |
| 5 | `phase1_completion_zh.md` | P1‑A + P1‑B 落地报告（scatter/gather + 激活重算） | 同上 |
| 6 | `phase2_3_completion_zh.md` | P2 + P3 落地报告（bubble/带宽统计、roofline derate、VRAM cap、多 NIC 拓扑） | 同上 |
| 7 | `reference_zh.md` | **配置参考**：新增 CLI/env/JSON 字段汇总对照表 | 使用时速查 |
| 8 | `experiments_zh.md` | **复现配方**：针对论文 takeaway 的最小实验 | 做验证实验 |
| 9 | `correctness_todo.md` | **正确性排期**：跑 Megatron-39B/76B 后发现的 4 项真·bug / 失真，按 P0/P1 排期 | 修 bug 的人 |
| 10 | `optional_todo.md` | **Optional 改进**：工程效率、对标精度、扩展性增强 | 有余力时挑做 |

## 原始 PDF 与页面渲染

- `2104.04473v5.pdf` — 原论文
- `figs/page-01.png … page-13.png` — 每页渲染图，中文译文里直接引用

## 一张图看懂完成情况

```
 Phase 0 ──┬── P0-A 显式调度 (gpipe/1f1b/1f1b-interleaved) ✅
           └── P0-B interleaved 非连续映射 (v>1)              ✅
 Phase 1 ──┬── P1-A scatter/gather (1/t 字节 + AG)            ✅
           └── P1-B 激活重算 (backward +1× forward FLOP)      ✅
 Phase 2 ──┬── P2-A bubble + 有效二分带宽统计                 ✅
           └── P2-B Roofline achievable fraction              ✅
 Phase 3 ──┬── P3-A VRAM 峰值 + 容量告警                      ✅
           └── P3-B generate_topology --nics-per-gpu          ✅
```

## 触达的代码文件

**STG 侧**（`dnn_workload/symbolic_tensor_graph/`）：
- `main.py` — 4 个新 CLI 参数、两路 `_create_pipeline_tensor_map_*` 支持 v>1
- `symbolic_tensor_graph/graph/pipeline_schedule.py`（新） — 流水调度注入
- `symbolic_tensor_graph/graph/activation_recompute.py`（新） — 激活重算 FLOP 抬升
- `symbolic_tensor_graph/graph/convert_chakra.py` — scatter/gather 的 P2P 切分 + AllGather
- `symbolic_tensor_graph/graph/graph.py` — 新 `NodeType.Y_RECV_AG`
- `symbolic_tensor_graph/vram_counting.py` — 激活重算 VRAM 粗缩
- `test_cases/test_pipeline_interleaved.py`（新）
- `test_cases/test_scatter_gather.py`（新）

**ASTRA‑sim 侧**（`astra-sim/`）：
- `astra-sim/system/Roofline.{hh,cc}` — achievable_fraction
- `astra-sim/system/Sys.{hh,cc}` — 解析 `peak-perf-achievable-fraction` / `vram-capacity-gb`
- `astra-sim/workload/Statistics.{hh,cc}` — bubble + effective BW
- `astra-sim/workload/Workload.cc` — VRAM cap 检查

**实验配置**：
- `astra-sim/qwen_experiment/in_dc/astra_system.json` — 启用 track-local-mem 等
- `astra-sim/qwen_experiment/in_dc/generate_topology.py` — `--nics-per-gpu`
- `dnn_workload/qwen_32b/qwen_32b.sh` — 透传 `PP_SCHEDULE` / `PP_VIRTUAL` / `SGO` / `ACTIVATION_RECOMPUTE`

## 回归测试

```bash
cd dnn_workload/symbolic_tensor_graph
source /home/ps/sow/part2/astra-sim/.venv/bin/activate
python3 test_cases/test_pipeline_interleaved.py   # P0-A/B
python3 test_cases/test_scatter_gather.py         # P1-A
```

两份测试脚本都可以单跑、返回 `ALL PASS`。

## 注意

- 所有新增行为都是 opt‑in，默认 env 保留旧行为（natural 调度、无 scatter/gather、不重算、cap=0 等）。把所有开关全关就等价 Phase 0 前的 baseline。
- 上游合并分析可跑 `git diff master -- dnn_workload/symbolic_tensor_graph astra-sim/astra-sim/{system,workload} astra-sim/qwen_experiment/in_dc/{generate_topology.py,astra_system.json} dnn_workload/qwen_32b/qwen_32b.sh`。
