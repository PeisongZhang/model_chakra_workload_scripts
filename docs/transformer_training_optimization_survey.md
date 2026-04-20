# Transformer 类大模型训练优化技术综述

本文面向 **Transformer / LLM 训练**，重点总结几类最常见、也最有代表性的优化方向：`FlashAttention` 系列、`ZeRO/FSDP` 系列、`Sliding Window / Sparse Attention`、长序列分布式注意力、激活重计算、并行训练、低精度与低比特优化器等。

这里有一个先行判断：

- 有些技术 **不改变模型数学定义**，主要提升训练效率，例如 FlashAttention、ZeRO、FSDP、激活重计算、混合精度。
- 有些技术 **改变了注意力模式或训练同步方式**，本质上是模型结构或优化算法的变化，例如 Sliding Window Attention、BigBird、Longformer、Local SGD。

如果你的目标是做系统建模或仿真，这个区分非常重要。

## 1. 总览

| 类别 | 主要解决的瓶颈 | 代表技术 | 是否改变模型语义 |
|---|---|---|---|
| 注意力内核优化 | Attention 的 HBM IO、显存峰值、kernel 效率 | FlashAttention 1/2/3 | 否 |
| 参数/梯度/优化器状态分片 | 单卡显存装不下模型与 optimizer state | ZeRO-1/2/3, FSDP | 否 |
| 长上下文注意力稀疏化 | 序列长度增大导致 O(S^2) 成本过高 | Sliding Window, Longformer, BigBird | 是 |
| 长序列分布式精确注意力 | 单卡无法容纳超长序列上下文 | Ring Attention, Ulysses, Context Parallelism | 通常否 |
| 激活显存优化 | 反向传播需要保存太多 activation | Gradient Checkpointing, Recompute, Reversible Layer | 否 |
| 分布式并行训练 | 单设备算力/显存/带宽不足 | TP, PP, DP, EP, 3D/4D Parallelism | 否 |
| 通信优化 | All-Reduce 频繁、跨机带宽不足 | Local SGD, overlap, compression | 可能是 |
| 数值表示优化 | Tensor Core 利用率、显存与带宽 | FP16/BF16/FP8, 8-bit Optimizer | 通常否 |

## 2. FlashAttention 系列

### 2.1 要解决什么问题

标准 self-attention 的算力复杂度是 `O(S^2 * d)`，更关键的是它通常会显式或隐式物化 `S x S` 注意力矩阵，导致：

- 显存占用高
- HBM 读写量大
- kernel 之间的数据搬运开销大

在长序列训练里，Attention 往往先被 **显存带宽和 IO** 卡住，而不是先被纯 FLOPs 卡住。

### 2.2 核心思想

FlashAttention 的核心不是“近似注意力”，而是 **IO-aware exact attention**：

- 通过 tiling / blockwise 方式在片上 SRAM 中分块计算
- 利用 online softmax，避免完整物化 `S x S` 注意力矩阵
- 在反向传播中用 recomputation 替代大规模中间状态保存

因此它保持了和标准 attention 相同的数学结果，但显著降低了显存访问和峰值内存。

### 2.3 代表论文

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
   Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re. NeurIPS 2022.  
   贡献：提出 IO-aware exact attention，把 attention 的瓶颈从“物化大矩阵”转到“块内流式计算”。

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**  
   Tri Dao. 2023.  
   贡献：进一步优化 GPU work partitioning，提高并行度与 occupancy，在训练场景下更快。

3. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision**  
   Tri Dao et al. 2024.  
   贡献：针对 Hopper 等新硬件进一步优化，引入更强的异步流水与低精度支持。

### 2.4 评价

- 优点：通常是 **最值得优先打开** 的训练优化之一；不改模型数学定义，收益直接。
- 局限：它解决的是 attention kernel 效率，不解决 optimizer state、参数分片、跨机通信这些问题。
- 建模含义：如果做系统仿真，FlashAttention 应体现为 **相同语义、更低显存峰值、更低 IO、更好的 kernel 效率**，而不是简单把 `O(S^2)` 改成线性。

### 2.5 在本仓库的状态

已在 STG 中落地：`--attention_backend flash`（或旧参数 `--flash_attention true`）会切到 `group_query_attention_kernel_flash.csv`，保留 $4BHS^2d$ 前向 FLOPs 和 2.5× 的反向/前向比，同时避免 $S \times S$ tensor 物化。细节见 [flash_attention.md](flash_attention.md)。

## 3. ZeRO / FSDP 系列

### 3.1 要解决什么问题

大模型训练的显存不只花在参数本身，还花在：

- 参数 `weights`
- 梯度 `gradients`
- 优化器状态 `optimizer states`，例如 Adam 的一阶矩、二阶矩

对 Adam 来说，优化器状态往往比参数本体还大。模型一旦上到几十 B 或上百 B 参数，只做 Data Parallel 很快就会显存爆炸。

### 3.2 ZeRO 的核心思想

ZeRO 的基本逻辑是：既然 Data Parallel 的多张卡上原本存着大量重复状态，那么就把这些状态 **分片** 到不同设备，而不是每张卡都存一整份。

- **ZeRO-1**：分片 optimizer states
- **ZeRO-2**：在 ZeRO-1 基础上，再分片 gradients
- **ZeRO-3**：进一步分片 parameters，本质上接近 fully sharded

### 3.3 代表论文

1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**  
   Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. SC 2020.  
   贡献：提出 ZeRO-1/2/3，把 DP 训练中的重复状态系统化拆解并分片。

2. **ZeRO-Offload: Democratizing Billion-Scale Model Training**  
   Samyam Rajbhandari et al. USENIX ATC 2021.  
   贡献：把部分训练状态和计算 offload 到 CPU，降低 GPU 显存压力。

3. **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**  
   Samyam Rajbhandari et al. SC 2021.  
   贡献：进一步把状态扩展到 CPU/NVMe 分层存储，实现更极端的超大模型训练。

### 3.4 FSDP 与 ZeRO 的关系

`FSDP`（Fully Sharded Data Parallel）可以理解为 PyTorch 生态中非常重要的 fully-sharded 实现路线，和 ZeRO-3 在思想上高度接近：都在做参数、梯度、优化器状态的全分片。

一个常见工程认知是：

- DeepSpeed 世界里更常说 `ZeRO`
- PyTorch 原生世界里更常说 `FSDP`

### 3.5 评价

- 优点：这是 **超大参数模型训练的基础设施级技术**。
- 代价：显存省了，但通信显著增加；参数 gather/shard 的时机设计非常关键。
- 建模含义：如果做系统仿真，ZeRO/FSDP 不只是“显存变小”，还意味着更多的参数重组、all-gather、reduce-scatter、prefetch 与 overlap。

## 4. Sliding Window / Sparse Attention 系列

### 4.1 这一类技术的本质

这类方法和 FlashAttention 不同。FlashAttention 是 **精确 attention 的高效实现**；而 Sliding Window / Sparse Attention 是 **改变注意力连接图本身**。

它们的目标是把 attention 从 `O(S^2)` 改成更低复杂度，例如 `O(S * W)`，其中 `W` 是窗口大小。

### 4.2 代表论文

1. **Longformer: The Long-Document Transformer**  
   Iz Beltagy, Matthew E. Peters, Arman Cohan. 2020.  
   贡献：局部滑动窗口 attention + 少量全局 token，适合长文档任务。

2. **Big Bird: Transformers for Longer Sequences**  
   Manzil Zaheer et al. NeurIPS 2020.  
   贡献：把 local、global、random sparse attention 组合起来，并给出较强的理论性质分析。

3. **Reformer: The Efficient Transformer**  
   Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya. ICLR 2020.  
   贡献：用 LSH attention 和 reversible layer 降低长序列训练成本。

4. **Mistral 7B**  
   Albert Q. Jiang et al. 2023.  
   贡献：在现代 LLM 里把 `Sliding Window Attention (SWA)` 用成非常实用的工程方案，并结合 GQA。

### 4.3 评价

- 优点：长序列成本下降很明显，尤其当全局精确 attention 没有必要时。
- 局限：它不是纯系统优化，而是 **模型结构改变**；效果依赖任务和窗口设计。
- 建模含义：如果做训练系统论文，需要明确区分“精确 attention kernel 优化”和“attention 稀疏化/局部化”。

## 5. 长序列训练的分布式精确注意力

当序列非常长时，另一个方向不是改稀疏模式，而是尽量保持精确 attention，同时把序列维度切到多卡上。

### 5.1 代表论文

1. **DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**  
   2023.  
   贡献：沿序列维度和头维度重新组织并行与通信，使超长序列训练更可行。

2. **Ring Attention with Blockwise Transformers for Near-Infinite Context**  
   Hao Liu et al. 2023.  
   贡献：通过 ring 式 KV 传递和 blockwise exact attention，把超长上下文扩展到多设备。

### 5.2 评价

- 这一类方法和 Sliding Window 不同，它更接近“**分布式 exact attention**”。
- 主要矛盾从单卡显存，转成跨卡通信和 pipeline 组织。
- 对 Astra-sim / Chakra 这类系统建模工具很有价值，因为它们会显著改变通信模式。

## 6. 激活显存优化：Checkpointing / Recomputation

### 6.1 核心思想

训练时最大的显存之一来自 activation。最直接的方法是：

- 前向传播时不保存全部 activation
- 反向传播时需要时再重算一部分

这类方法通常会增加一些计算量，但能显著减少显存峰值。

### 6.2 代表论文

1. **Training Deep Nets with Sublinear Memory Cost**  
   Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin. 2016.  
   贡献：经典 activation checkpointing 思想来源。

2. **Reducing Activation Recomputation in Large Transformer Models**  
   2022.  
   贡献：面向大型 Transformer 的更细粒度重计算设计，与 Megatron-LM 体系很相关。

### 6.3 评价

- 优点：非常通用，几乎所有大模型训练框架都支持。
- 局限：以额外 FLOPs 换显存；和 FlashAttention 的反向重计算可能存在叠加效应。

## 7. 并行训练：TP / PP / DP / EP

### 7.1 代表论文

1. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**  
   Mohammad Shoeybi et al. 2019.  
   贡献：现代 Tensor Parallel 训练的经典起点。

2. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**  
   Yanping Huang et al. NeurIPS 2019.  
   贡献：把模型按层切成流水线 stage，用 micro-batch 驱动 pipeline。

3. **PipeDream: Generalized Pipeline Parallelism for DNN Training**  
   Aaron Harlap et al. SOSP 2019.  
   贡献：更激进的流水线调度与版本化权重管理。

4. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**  
   Deepak Narayanan et al. 2021/2022.  
   贡献：把 TP、PP、DP 系统化结合，形成现代 3D parallelism 训练框架。

### 7.2 评价

- 这是大模型训练的骨架层技术，没有它，FlashAttention 或 ZeRO 也很难真正落地到超大规模。
- 真实系统中通常是多种技术叠加：  
  `TP + PP + DP + ZeRO/FSDP + FlashAttention + Checkpointing`

### 7.3 在本仓库的状态

STG 已支持 TP/PP/DP/SP/EP 五轴并行以及 Megatron-LM PTD-P 的多个关键特性：

- `--pipeline_schedule {natural, gpipe, 1f1b, 1f1b-interleaved}`
- `--pipeline_virtual_stages v`（interleaved 1F1B）
- `--scatter_gather_optimization`（Megatron-LM §4.1 的 PP 边界 scatter/gather）
- `--activation_recompute`（FLOPs +1×fwd，显存按保留率粗缩）
- `--weight_sharded`（ZeRO/FSDP 风格权重分片）

配置和验证方法见 [PDT/README.md](PDT/README.md) 与 [README.md](README.md)。

## 8. 通信优化：Local SGD 与压缩

### 8.1 Local SGD

标准 DP 通常每 step 同步一次梯度。`Local SGD` 则允许每张卡先本地走若干步，再周期性同步，从而减少通信频率。

代表论文：

1. **Local SGD Converges Fast and Communicates Little**  
   Sebastian U. Stich. ICLR 2019.  
   贡献：给出 Local SGD 的经典收敛分析。

### 8.2 梯度/优化器压缩

还有一类方法是压缩通信或优化器状态。

代表论文：

1. **8-bit Optimizers via Block-wise Quantization**  
   Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. 2022.  
   贡献：把优化器状态压成低比特，明显降低显存与带宽消耗。

### 8.3 评价

- Local SGD 会改变优化轨迹，不是单纯系统层透明优化。
- 8-bit optimizer 一般更偏工程实用，但对大规模训练的显存收益很直接。

### 8.4 在本仓库的状态

STG 已支持 LocalSGD workload 生成：`--num_iterations N` 复制 iteration，`--dp_local_sgd_interval K` 只在满足 `(iter+1) % K == 0` 的 iteration 上保留 DP `ALL_REDUCE`。默认值都为 `1`，行为仍是同步 DP。详见 [local_sgd.md](local_sgd.md)。

## 9. 低精度训练：FP16 / BF16 / FP8

### 9.1 代表论文

1. **Mixed Precision Training**  
   Paulius Micikevicius et al. ICLR 2018.  
   贡献：系统化奠定 FP16 mixed precision 训练的工程路径，包括 loss scaling 等关键技巧。

2. **FP8 Formats for Deep Learning**  
   2022.  
   贡献：推动 FP8 训练格式进入主流硬件与系统讨论。

### 9.2 评价

- FP16/BF16 现在几乎已经是默认配置。
- FP8 还更依赖硬件、kernel、标定策略和训练稳定性经验。

## 10. 怎么理解这些技术之间的关系

一个实用的理解方式是：

### 10.1 不改模型数学定义，但提升训练系统效率

- FlashAttention
- ZeRO / FSDP
- Tensor Parallel / Pipeline Parallel / Sequence Parallel
- Gradient Checkpointing
- Mixed Precision

这类技术更适合归为 **system optimization**。

### 10.2 会改模型结构或优化轨迹

- Sliding Window Attention
- Longformer / BigBird / Reformer
- Local SGD

这类技术通常同时属于 **algorithm + system co-design**。

## 11. 如果只抓最关键的 paper，建议先读这几篇

如果你要快速建立对 Transformer 训练优化的整体认知，优先级可以这样排：

1. `FlashAttention`  
   看 attention kernel 为什么会被 IO 卡住。

2. `ZeRO`  
   看大模型训练为什么首先死在 optimizer state 和参数副本上。

3. `Megatron-LM` + `GPipe`  
   看现代 TP/PP/DP 是怎么形成训练骨架的。

4. `Longformer` 或 `BigBird`  
   看 sliding window / sparse attention 为什么不只是 kernel 优化，而是结构变化。

5. `Training Deep Nets with Sublinear Memory Cost`  
   看 activation checkpointing 的原始思想。

## 12. 参考文献清单

- Tri Dao et al. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. NeurIPS 2022.
- Tri Dao. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. 2023.
- Tri Dao et al. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision**. 2024.
- Samyam Rajbhandari et al. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. SC 2020.
- Samyam Rajbhandari et al. **ZeRO-Offload: Democratizing Billion-Scale Model Training**. USENIX ATC 2021.
- Samyam Rajbhandari et al. **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**. SC 2021.
- Iz Beltagy et al. **Longformer: The Long-Document Transformer**. 2020.
- Manzil Zaheer et al. **Big Bird: Transformers for Longer Sequences**. NeurIPS 2020.
- Nikita Kitaev et al. **Reformer: The Efficient Transformer**. ICLR 2020.
- Albert Q. Jiang et al. **Mistral 7B**. 2023.
- Hao Liu et al. **Ring Attention with Blockwise Transformers for Near-Infinite Context**. 2023.
- **DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**. 2023.
- Tianqi Chen et al. **Training Deep Nets with Sublinear Memory Cost**. 2016.
- **Reducing Activation Recomputation in Large Transformer Models**. 2022.
- Mohammad Shoeybi et al. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. 2019.
- Yanping Huang et al. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**. NeurIPS 2019.
- Aaron Harlap et al. **PipeDream: Generalized Pipeline Parallelism for DNN Training**. SOSP 2019.
- Deepak Narayanan et al. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**. 2021/2022.
- Sebastian U. Stich. **Local SGD Converges Fast and Communicates Little**. ICLR 2019.
- Tim Dettmers et al. **8-bit Optimizers via Block-wise Quantization**. 2022.
- Paulius Micikevicius et al. **Mixed Precision Training**. ICLR 2018.
- **FP8 Formats for Deep Learning**. 2022.

## 13. 对本仓库工作的直接启发

结合 `dnn_workload/` 目录里已有的材料，可以把这些技术粗略分成三类：

- **已经建模落地**：`FlashAttention`（[flash_attention.md](flash_attention.md)）、`Local SGD`（[local_sgd.md](local_sgd.md)）、`TP/PP/DP/SP/EP` 骨架 + Megatron PTD-P 关键特性（[PDT/README.md](PDT/README.md)）、`FSDP` 权重分片（`--weight_sharded`）、`Activation Recomputation`（`--activation_recompute`）
- **非常适合后续继续建模**：ZeRO-1/2（更细粒度的 optimizer state / gradient 分片状态机，目前只支持 ZeRO-3/FSDP 风格）、`Ring Attention`、`DeepSpeed-Ulysses` 这类长序列分布式 exact attention
- **更偏模型结构变化**：`Sliding Window / Longformer / BigBird` 这类稀疏/局部 attention，需要改 kernel CSV 的连接图

如果目标是继续扩 ASTRA-sim / Chakra 侧的训练系统建模，下一步的优先级建议是：

1. 长序列分布式 exact attention（`Ring Attention` / `Ulysses`）——会显著改变跨卡通信模式，对 Astra-sim 仿真最有价值
2. 更细粒度的 `ZeRO-1/2` 状态机，区分 optimizer state / gradient / parameter 三种分片触发的通信
3. 低精度（`FP8` / 8-bit optimizer）对张量尺寸与带宽的影响
4. 最后再考虑 `Sliding Window` 这类会改模型语义的稀疏 attention
