# docs/PDT — Megatron‑LM (PTD‑P) 仿真能力补齐

本目录围绕 Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（SC'21, arXiv:2104.04473v5）组织：**论文 → 配置/实验参考 → 正确性与优化排期**。能力补齐的 P0–P3 计划、差距分析、三阶段落地报告已于 commit 77f9268 完成并归档，详见 `git log`。

## 阅读顺序

| # | 文件 | 内容 | 适用读者 |
|---|------|------|---------|
| 1 | `2104.04473v5_zh.md` | 论文中文精译（按原章节组织；含图/表/核心公式） | 需先理解论文 |
| 2 | `reference_zh.md` | **配置参考**：新增 CLI/env/JSON 字段汇总对照表 | 使用时速查 |
| 3 | `experiments_zh.md` | **复现配方**：针对论文 takeaway 的最小实验 | 做验证实验 |
| 4 | `correctness_todo.md` | **正确性排期**：跑 Megatron-39B/76B 后发现的 4 项真·bug / 失真，按 P0/P1 排期 | 修 bug 的人 |
| 5 | `optional_todo.md` | **Optional 改进**：工程效率、对标精度、扩展性增强 | 有余力时挑做 |

## 原始 PDF 与页面渲染

- `2104.04473v5.pdf` — 原论文
- `figs/page-01.png … page-13.png` — 每页渲染图，中文译文里直接引用

## 能力落地状态

```
 Phase 0 ──┬── P0-A 显式流水调度 (gpipe/1f1b/1f1b-interleaved) ✅
           └── P0-B interleaved 非连续映射 (v>1)                ✅
 Phase 1 ──┬── P1-A scatter/gather (1/t 字节 + AG)              ✅
           └── P1-B 激活重算 (backward +1× forward FLOP)        ✅
 Phase 2 ──┬── P2-A bubble + 有效二分带宽统计                   ✅
           └── P2-B Roofline achievable fraction                ✅
 Phase 3 ──┬── P3-A VRAM 峰值 + 容量告警                        ✅
           └── P3-B generate_topology --nics-per-gpu            ✅
```

所有新增行为都是 opt‑in，默认 env 保留旧行为。把所有开关全关就等价 Phase 0 前的 baseline。具体 CLI / env / JSON 字段见 `reference_zh.md`。

## 回归测试

```bash
cd dnn_workload/symbolic_tensor_graph
source /home/ps/sow/part2/astra-sim/.venv/bin/activate
python3 test_cases/test_pipeline_interleaved.py   # P0-A/B
python3 test_cases/test_scatter_gather.py         # P1-A
```

两份测试脚本都可以单跑、返回 `ALL PASS`。
