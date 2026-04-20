# `main.py` 参数速查

`dnn_workload/symbolic_tensor_graph/main.py` 是 Symbolic Tensor Graph (STG) 生成器的入口，根据模型配置和并行策略产出每个 rank 的 Chakra 执行轨迹 (`workload.%d.et`) 和通信组描述 (`workload.json`)。

日常使用通常不直接调用它，而是通过 `dnn_workload/<model>/<model>.sh`（`llama3_8b.sh`、`qwen_32b.sh` 等）封装。

## 1. 并行策略

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dp` | 1 | 数据并行度 |
| `--tp` | 1 | 张量并行度 |
| `--sp` | 1 | 序列并行度 |
| `--pp` | 1 | 流水线并行度 |
| `--ep` | 1 | 专家并行度（仅 MoE） |
| `--weight_sharded` | false | ZeRO/FSDP 风格的权重分片，按 `dp` 切分 |
| `--tpsp` | true | 是否同时启用 TP + SP；`false` 退化成 TP only |
| `--scatter_gather_optimization` | false | Megatron-LM §4.1 scatter/gather：PP 边界每个 TP rank 只发 1/t 片段，接收端用 intra-TP all-gather 拼回（仅 `tp>1` 有效） |

总 NPU 数 `= dp * tp * pp * sp`。

## 2. 模型结构

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model_type` | dense | `dense` / `llama` / `gpt` / `moe` / `debug` |
| `--num_stacks` | 80 | Transformer block 层数 |
| `--dmodel` | 8192 | hidden size |
| `--dff` | 28672 | FFN 中间维度 |
| `--seq` | 1024 | 序列长度 |
| `--head` | 64 | 注意力头数 |
| `--kvhead` | 8 | KV 头数（GQA） |
| `--dvocal` | 32000 | 词表大小 |
| `--experts` | 8 | MoE 专家数 |
| `--kexperts` | 2 | MoE 每 token 激活专家数 |

## 3. 训练 / 执行

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch` | 64 | **全局** batch size（跨所有 DP rank 的 GBS） |
| `--micro_batch` | -1 | **每卡** 一次前/反向处理的样本数（Megatron 习惯）；一个 iteration 内的 micro-batch 数 = `batch / (micro_batch * dp)`。默认 `-1` 表示“每卡跑满 `batch/dp`，不做梯度累积” |
| `--num_iterations` | 1 | 生成的 workload 里包含几个连续 iteration |
| `--dp_local_sgd_interval` | 1 | 每 K 个 iteration 才保留 DP all-reduce；`1` 即同步 DP。详见 [local_sgd.md](local_sgd.md) |
| `--activation_recompute` | false | 反向按 +1× forward FLOP 抬升，VRAM 按保留率粗缩（P1-B） |
| `--mixed_precision` | false | 打开后 `--model_type llama` 使用 `llama_model.py` 并切到混精流水线映射 |
| `--flash_attention` | false | 旧开关，等价于 `--attention_backend flash`；为兼容保留 |
| `--attention_backend` | auto | `standard` / `fused` / `flash`；`auto` 意味着按 `--flash_attention` 决定。详见 [flash_attention.md](flash_attention.md) |
| `--pipeline_schedule` | natural | `natural` / `gpipe` / `1f1b` / `1f1b-interleaved`（P0-A） |
| `--pipeline_virtual_stages` | 1 | 每卡虚拟 stage 数 `v`，`v>1` 启用 interleaved 1F1B，并要求 `num_stacks % (v * pp) == 0`（P0-B） |
| `--print_gpu_vram` | false | 打印各 GPU 的显存占用（参数/激活/梯度） |

`--batch` 必须能被 `micro_batch * dp` 整除；`--num_iterations` 与 `--dp_local_sgd_interval` 均要求 ≥ 1。

`--pipeline_schedule`、`--pipeline_virtual_stages`、`--scatter_gather_optimization`、`--activation_recompute` 属于 Megatron-LM (PTD-P) 仿真能力补齐的一部分，完整背景见 [PDT/README.md](PDT/README.md)。

## 4. 输出

| 参数 | 默认 | 说明 |
|---|---|---|
| `--output_dir` | 必填 | 写入目录 |
| `--output_name` | 必填 | 文件名模板，含 `%d` 占位 rank，例如 `workload.%d.et` |
| `--chakra_schema_version` | v0.0.4 | 目前所有后端都走 v0.0.4 |

通信组 JSON 由 `output_name` 去掉 `.%d.et` 后缀推得（例如 `workload.json`）。

## 5. 调用链

1. 根据 `--model_type` 在 `models/stage1/{llama_model,gpt_model,moe_model}.py` 中组装符号张量图
2. `MicroBatchReplicator` 按 `Batch / MicroBatch` 复制出每个 micro-batch 子图（`MicroBatch` 在符号表里被设为 `micro_batch * dp`，使 `Batch/dp` 形状替换后正好等于 `micro_batch`）
3. `GradUpdater` 追加反向和权重更新，`ReplicateGraph` 处理 FSDP 形状替换
4. `GraphDistributer` 把张量映射到并行轴上（含 pipeline tensor map）
5. `BundledConvertChakra` 下沉到 Chakra 节点
6. 可选后处理：`ActivationRecomputePostProcess` / `PipelineScheduleInjector` / `LocalSGDIterationPostProcess`
7. `Chakra004Backend` readout，每个 rank 一个 `.et`

## 6. 相关文档

- 注意力后端的三种实现（standard / fused / flash）和 FLOPs 建模：[flash_attention.md](flash_attention.md)
- Local SGD workload 的生成和语义：[local_sgd.md](local_sgd.md)
- Transformer 基础结构与 FLOPs：[transformer.md](transformer.md)
- Megatron-LM PTD-P 仿真能力补齐（P0-P3 阶段报告与实验配方）：[PDT/README.md](PDT/README.md)
- 训练系统优化综述：[transformer_training_optimization_survey.md](transformer_training_optimization_survey.md)
