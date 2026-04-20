# Phase 0 完成小结：流水线调度策略

对应 `docs/PDT/implementation_plan_zh.md` 的 P0‑A + P0‑B 两项。

## 改动文件

| 文件 | 作用 |
|------|------|
| `dnn_workload/symbolic_tensor_graph/main.py` | 新增 `_build_chunk_cumulative_bounds` / `_block_idx_to_device` 辅助函数；`_create_pipeline_tensor_map[_mix_precision]` 接收 `virtual_stages` 参数并做非连续 chunk 映射；新增 CLI `--pipeline_virtual_stages`, `--pipeline_schedule`；`_postprocess_chakra_graph` 增加 `pp` 参数并调用 `PipelineScheduleInjector` |
| `dnn_workload/symbolic_tensor_graph/symbolic_tensor_graph/graph/pipeline_schedule.py` | 新模块。实现 `PipelineScheduleInjector`，支持 natural / gpipe / 1f1b / 1f1b-interleaved 四种调度；用节点名 `mb{i}.` 前缀识别 micro‑batch 索引，用 `.d{letter}@` 后缀识别前向/反向；对每个 rank 注入 ctrl_deps |
| `dnn_workload/symbolic_tensor_graph/test_cases/test_pipeline_interleaved.py` | 回归测试：映射函数单元测试 + v=1/v=2 的 ET 生成对照 |
| `dnn_workload/qwen_32b/qwen_32b.sh` | 暴露 `PP_SCHEDULE` / `PP_VIRTUAL` 环境变量；把新参数透传到 `main.py`；把两者写入输出目录名 |

## 使用方式

```bash
# 原行为（完全兼容，natural + v=1）
DP=4 TP=8 PP=4 LAYER=32 bash dnn_workload/qwen_32b/qwen_32b.sh

# GPipe 显式调度
PP_SCHEDULE=gpipe DP=4 TP=8 PP=4 LAYER=32 bash .../qwen_32b.sh

# 1F1B (PipeDream-Flush 风格)
PP_SCHEDULE=1f1b DP=4 TP=8 PP=4 LAYER=32 bash .../qwen_32b.sh

# 交错式 1F1B (virtual_stages=2，LAYER 必须能被 PP*PP_VIRTUAL 整除)
PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL=2 \
    DP=4 TP=8 PP=4 LAYER=32 bash .../qwen_32b.sh
```

输出目录名现在包含 `..._<schedule>_v<virtual>`，便于对比实验。

## 已验证

1. **映射正确性**：`test_cases/test_pipeline_interleaved.py` 通过——
   - v=1, 8 blocks, pp=4 → `[0,0,1,1,2,2,3,3]`（与原行为一致）
   - v=2, 8 blocks, pp=4 → `[0,1,2,3,0,1,2,3]`（chunk 级 round‑robin）
   - v=2, 16 blocks, pp=4 → `[0,0,1,1,2,2,3,3, 0,0,1,1,2,2,3,3]`
   - v=4, 16 blocks, pp=4 → `[0,1,2,3]*4`
   - 余数分布（9 blocks, pp=4, v=1）验证

2. **交错通信量**：生成的 Chakra ET 中，v=2 相对 v=1 的 SEND/RECV 数严格增加（对照论文 "interleaved 通信量 × v"）。

   | rank | v=1 SEND/RECV | v=2 SEND/RECV |
   |------|---------------|---------------|
   | 0    | 4 / 4         | 12 / 12       |
   | 1    | 8 / 8         | 16 / 16       |
   | 2    | 8 / 8         | 16 / 16       |
   | 3    | 4 / 4         | 12 / 12       |

3. **ctrl_dep 公式对账**（`num_mb = 4`）：
   - `gpipe`: `(m-1)*2 + 1 = 7` 条/rank（3× F‑序列 + 1× F/B 屏障 + 3× B‑序列）
   - `1f1b`: `2m - 1 = 7` 条/rank（序列长 2m，边数 seq‑1）
   - `natural`: 0 条 ✓

4. **结构不变量**：natural / gpipe / 1f1b 三种 schedule 的 COMP / SEND / RECV 数量完全一致，只有 ctrl_deps 不同——说明调度仅改执行顺序、不改数据图。

## 已知限制

- **1f1b‑interleaved 当前是 1f1b 的 mb 粒度退化**。块级 chunk 粒度调度需要在 post‑process 里拿到 "block→chunk 索引" 的映射，当前未从 `_create_pipeline_tensor_map` 透传下来。已在 `pipeline_schedule.py:_apply_1f1b_interleaved_to_rank` 留 TODO。
- **Analytical 仿真对 schedule 的时间差异不敏感**：ASTRA‑sim 目前 `gpu-comp=1`，自然调度已近似于 GPipe 串行；1F1B 的真正收益（激活显存下降）需等 P3‑A 的 VRAM 建模接入后才可观测。
- **非混合精度路径**的 `out_emb`/`loss` 归属由 `(num_stacks‑1) % range_` 改为 `_block_idx_to_device(num_stacks‑1)`。非整除时（如 num_stacks=9, pp=4）映射从 device 0 变为 device 3；这更贴近语义（loss 在最后一个 stage），但不同于原 legacy 行为。`qwen_32b.sh` 用 mixed_precision=true，该路径不受影响。

## 待做（后续 phase）

- **P1‑A**：在 `convert_chakra.py:460-493` 将跨 stage P2P 节点按 TP 度切成 `1/t` + NVLink all‑gather。
- **P1‑B**：`--activation_recompute` 真正接入 `GradUpdater` 和 `vram_counting`。
- **P2‑A**：在 ASTRA‑sim `Statistics.cc` 加 `bubble_fraction` 与 effective bisection bandwidth。
- **P2‑B**：Roofline 支持 per‑op‑type 可达 TFLOP/s。
- **P3‑A**：打开 `track-local-mem`，加 VRAM cap 检查。
- **P3‑B**：`generate_topology.py` 加 `--nics-per-gpu`。
