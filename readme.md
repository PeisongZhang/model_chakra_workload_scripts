main.py 是 Symbolic Tensor Graph (STG) 生成器的核心脚本，主要用于根据模型配置和并行策略生成 Chakra 执行轨迹（Execution Traces）。

  以下是 main.py 支持的所有输入参数及其功能分析：

  1. 并行策略参数 (Parallelism Degrees)
  这些参数决定了模型在不同维度上的切分程度：
   * --dp: 数据并行度 (Data Parallel)。默认值为 1。
   * --tp: 张量并行度 (Tensor Parallel)。默认值为 1。
   * --sp: 序列并行度 (Sequence Parallel)。默认值为 1。
   * --pp: 流水线并行度 (Pipeline Parallel)。默认值为 1。
   * --ep: 专家并行度 (Expert Parallel)。用于 MoE 模型，默认值为 1。
   * --weight_sharded: 权重分片 (ZeRO/FSDP)。布尔值（支持 true/false, 1/0），默认 False。如果开启，权重将根据 dp 程度进行分片（类似于 FSDP）。
   * --tpsp: TP 与 SP 结合。布尔值，默认 True。决定是否同时使用张量并行和序列并行。

  2. 模型架构参数 (Model Architecture)
  用于定义 Transformer 模型的具体结构：
   * --model_type: 模型类型。可选值包括 dense (默认), llama, gpt, moe, debug。
   * --num_stacks: 层数。即 Transformer Block 的数量，默认 80。
   * --dmodel: 模型维度。即隐藏层大小 (Hidden Size)，默认 8192。
   * --dff: FFN 维度。Feed-Forward 网络中间层的维度，默认 28672。
   * --seq: 序列长度。默认 1024。
   * --head: 注意力头数。默认 64。
   * --kvhead: KV 头数。用于 GQA (Grouped Query Attention)，默认 8。
   * --dvocal: 词表大小。默认 32000。
   * --experts: 专家总数。仅在 model_type=moe 时有效，默认 8。
   * --kexperts: 激活专家数。每个 token 激活的专家数量，默认 2。

  3. 训练与执行配置 (Execution Config)
   * --batch: 全局批次大小 (Global Batch Size)。默认 64。
   * --micro_batch: 微批次大小 (Micro-batch Size)。默认 -1（表示等于 batch）。用于流水线并行中的微批次切分。
   * --activation_recompute: 重计算。是否开启激活值重计算以节省显存，默认 False。
   * --mixed_precision: 混合精度。是否使用混合精度训练（如 FP16/BF16），默认 False。
   * --print_gpu_vram: 打印显存占用。布尔值，默认 False。开启后会打印每个 GPU 的显存占用详情（包括参数、激活值、梯度等）。

  4. 输出配置 (Output Settings)
   * --output_dir: 输出目录（必选）。存储生成的 .et 轨迹文件和相关的 .json 通信组文件。
   * --output_name: 输出文件名（必选）。通常包含 %d 占位符，用于区分不同 rank 的轨迹文件（例如 workload.%d.et）。
   * --chakra_schema_version: Chakra 版本。默认 "v0.0.4"。


## Batch

在 main.py 脚本（以及大模型训练的通用上下文）中，--batch 和 --micro_batch 之间的关系是全局与局部、整体与切分的关系。

  它们通过流水线并行 (Pipeline Parallelism, PP) 和 数据并行 (Data Parallelism, DP) 耦合在一起。其核心逻辑可以用以下公式概括：

  1. 核心数学关系
  在 main.py 的代码逻辑中，它们的关系遵循：

  $$ \text{Gradient Accumulation Steps (梯累积步数)} = \frac{\text{Global Batch Size (--batch)}}{\text{Micro Batch Size (--micro\_batch)} \times
  \text{DP Degree (--dp)}} $$

   * --batch (Global Batch Size)：指一次完整的参数更新（Gradient Update）所消耗的总样本数。
   * --micro_batch (Micro Batch Size)：指单次前向传播（Forward）和反向传播（Backward）在单个设备（或流水线）上处理的小块样本数。

  ---

  2. 在脚本中的具体作用

  通过分析 main.py 的源码，这两个参数共同决定了流水线并行（Pipeline Parallelism）的调度行为：

  A. 决定流水线切分份数 (Number of Chunks)
  在流水线并行（PP > 1）模式下，脚本会根据这两个参数计算出需要将一个 Global Batch 切分成多少个 Micro-batches：

   1 # 逻辑推导
   2 num_micro_batches = args.batch // (args.micro_batch * args.dp)
  如果 num_micro_batches > 1，则意味着开启了梯度累积（Gradient Accumulation）。

  B. 影响 1F1B 调度算法
  在生成 Chakra 执行轨迹（Execution Trace）时，main.py 会利用这两个值来模拟典型的流水线调度（如 1F1B 策略）：
   * Warm-up 阶段：流水线先注入若干个 Micro-batches（数量通常等于 PP 级数）。
   * Steady 阶段：进入 1 个 Forward 紧跟 1 个 Backward 的循环，直到所有 Micro-batches 处理完毕。
   * Cooldown 阶段：完成剩余的反向传播。

  C. 显存压力与计算效率的平衡
   * 增大 --micro_batch：可以提高 GPU 利用率（算子计算更饱和），但会显著增加激活值（Activation）占用的显存。
   * 减小 --micro_batch 但保持 --batch 不变：会增加梯度累积步数。这可以节省显存，但会引入更多的流水线气泡（Pipeline Bubbles），降低整体吞吐量。

  ---

  3. 约束条件
  在运行 run_demo3.sh 时，你需要确保：
   1. 整除性：--batch 必须能被 (--micro_batch * --dp) 整除。
   2. PP 协调：为了让流水线并行高效运行，通常要求 num_micro_batches（即 batch / (micro_batch * dp)）远大于流水线级数 PP，以减少气泡占比。

  示例：
  在 run_demo3.sh 中，如果设置了 --dp 2，且你希望全局 Batch 为 32，Micro Batch 为 4：
   * 每个 DP 组处理 16 个样本。
   * 这 16 个样本会被切分成 $16 / 4 = 4$ 个 Micro-batches 进入流水线。
   * 程序会模拟 4 次 Micro-batch 的前向和反向，最后进行一次全量梯度同步（All-Reduce）。