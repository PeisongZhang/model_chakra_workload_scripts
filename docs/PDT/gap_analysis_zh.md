# 当前仿真栈对 Megatron-LM (PTD-P) 训练方案的覆盖度分析

> **对象**：
> - `dnn_workload/qwen_32b/qwen_32b.sh` — STG 生成 Qwen‑32B 的 Chakra ET workload
> - `astra-sim/qwen_experiment/in_dc/analytical.sh` — ASTRA‑sim analytical (congestion‑aware) 后端仿真
>
> **目标论文**：Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (SC'21, arXiv:2104.04473v5)
>
> 本文逐项核对论文提到的训练方案要素在当前仿真栈中的支持状态，指出缺失内容，并给出建议补齐路线。
>
> **勘误（2026-04-19）**：本文下面把"层次化拓扑"（#17–#19）列为缺失是错误判断。实际上 `astra-sim/qwen_experiment/in_dc/topology.txt` 与 `generate_topology.py` **已经** 实现 NVLink 4800 Gbps 节点内 + IB 200 Gbps 节点间的分层胖树。详见 `implementation_plan_zh.md` 开头的"重要修正"与 `phase2_3_completion_zh.md` 的 P3‑B 节。
>
> **实现状态（2026-04-19）**：本报告列出的 ❌/⚠️ 项在后续 Phase 0–3 中已全部补齐（见 `phase0_completion_zh.md` / `phase1_completion_zh.md` / `phase2_3_completion_zh.md`）；本文保留作为最初盘点记录。

---

## 1. TL;DR

当前仿真栈 **能覆盖 PTD‑P 的大部分静态切分维度**（张量并行 + 流水并行 + 数据并行 + micro‑batch + LocalSGD 可选），可以生成一个在 $(DP, TP, PP) = (4, 8, 4)$ 下 128 卡的 Qwen‑32B 工作负载，并在 analytical congestion-aware 后端跑出一次迭代/多次迭代的时间线和通信开销。

但对论文的几项 **关键创新与工程优化** 还缺：
1. **显式的 1F1B / GPipe / 交错式（interleaved）流水调度**——当前完全靠 Chakra 图的数据依赖自然派发，等同一个不带显式调度策略的 pipeline，既没有 warm‑up / 稳态 / cool‑down 三阶段的 1F1B，也无法测 interleaved schedule（$v > 1$）的 bubble 缩小效果。
2. **scatter / gather 通信优化**——相邻 PP stage 之间发送的张量仍是完整 $b\cdot s\cdot h$，没有按张量并行度切成 $1/t$ 再 all‑gather。
3. **激活重算（activation recomputation）对 FLOP / 内存的建模**——STG 侧有 `--activation_recompute` 参数但并未体现在当前脚本链路，ASTRA‑sim 也没有对比激活重算 on/off 的能力。
4. **算子融合 / kernel 级效率**——roofline 只用一个常量 `peak-perf=312 TFLOP/s` 当上限，无法复现论文 11%~19% 的融合收益。
5. **多 IB 网卡 + 三级胖树 + NVLink/NVSwitch 层次互连**——当前 `network.yml` 用的是一份自定义 `topology.txt`（277 节点 / 149 交换机 / 388 链路），是单层自定义结构，不能区分 node 内 NVSwitch vs node 间 IB，也没有 8×HCA 聚合的概念。
6. **checkpoint I/O、mixed‑precision 精度分档、bubble 解析验证** 等辅助能力。

结论：**定量上可以做相对比较（例如 `t` 增大时通信变多、pipeline 长一点气泡比例变大），但要精确复现论文中"52% 峰值利用率""bubble 随 $1/v$ 下降"之类结论还不够**。下文按要素详细说明。

---

## 2. 论文训练方案要素盘点

把 Megatron‑LM 论文中对训练吞吐有影响的要素按层次列出：

| 类别 | 要素 | 本文位置 |
|------|------|----------|
| 并行维度 | 张量并行 $t$ | §2.3 |
| | 流水并行 $p$ | §2.2 |
| | 数据并行 $d$，$p\cdot t\cdot d = n$ | §2.1 |
| | FSDP / ZeRO‑3（对比项） | §5.2 |
| 流水调度 | GPipe（全前向→全反向 + flush） | §2.2.1 |
| | PipeDream‑Flush / 1F1B | §2.2.1 |
| | **交错式 1F1B**（每设备 $v$ 个 chunk） | §2.2.2 |
| 通信优化 | **scatter / gather**（相邻 stage 间 P2P 只发 $1/t$） | §4.1 |
| | TP 层 all‑reduce（$f, g$ 算子） | §2.3 |
| 计算优化 | 算子融合（bias+GeLU、bias+drop+add、scale+mask+softmax） | §4.2 |
| | 张量布局改 `[s, b, a, h]` + strided batched GEMM | §4.2 |
| 显存 | 激活重算（activation recomputation） | §3.5 |
| | checkpoint 数量 $c$ 与显存关系 | §3.5 |
| 超参 | micro‑batch $b$（影响 bubble / 算术强度 / 显存） | §3.4 |
| | batch $B$（摊薄 bubble、减少 DP 通信频率） | §3.3.1 |
| 硬件 / 网络 | DGX A100 + NVLink + NVSwitch（节点内） | §5 |
| | 8× IB 200 Gbps HDR + 3 级胖树（节点间） | §5, §5.9 |
| | NVMe 并行文件系统（ckpt 读写） | §5, §5.10 |
| 结果指标 | 单卡 TFLOP/s & 峰值占比 | 表 1 |
| | pipeline bubble fraction | §2.2.1 |
| | 有效二分带宽（P2P、all‑reduce） | §5.9 |
| | 端到端训练时间（式 (4)） | §5.1 |

---

## 3. 当前仿真栈的组成与能力

### 3.1 Workload 生成端 — STG (`qwen_32b.sh` → `main.py`)

关键脚本片段（`qwen_32b.sh`）：

```bash
DP=${DP:-4}  TP=${TP:-8}  PP=${PP:-4}  SP=${SP:-1}   # NPUs = 4*8*4*1 = 128
SGD=standard                                           # 同步 DP（LocalSGD 可选）
LAYER=4  SEQUENCE=4096  BATCH=128  MICROBATCH=2        # 每 rank micro‑batch=2
python3 main.py ... --dp --tp --pp --sp --seq --batch --micro_batch \
                    --dvocal 152064 --dmodel 5210 --dff 27648 \
                    --head 40 --kvhead 8 --num_stacks LAYER \
                    --num_iterations 1 --dp_local_sgd_interval 1 \
                    --model_type llama --mixed_precision true \
                    --attention_backend standard --weight_sharded 0
```

STG 当前支持的能力（见 `main.py`）：

- **并行切分**：`--dp / --tp / --pp / --sp / --ep / --weight_sharded`，总 NPU 数 = $dp\cdot tp\cdot pp\cdot sp$（MoE 另计 ep）。
- **Pipeline stage 指派**：`_create_pipeline_tensor_map[_mix_precision]` 按 transformer block 索引 **连续分段** 到 $p$ 个 stage（余数前置），embedding→stage 0、out_emb/loss→最后 stage。**每个 device 恰好一个 stage**。
- **Micro‑batch 建模**：`MicroBatchReplicator` 克隆图 `batch // (micro_batch*dp)` 次，模拟 GA。默认使用正式克隆路径；`STAGE_MICROBATCH_OPTIMIZE=1` 为 shortcut（可能产生错误图，不推荐）。
- **LocalSGD / 多迭代**：`--num_iterations, --dp_local_sgd_interval`，`LocalSGDIterationPostProcess` 在非同步步删掉 DP all‑reduce。
- **TPSP**：`--tpsp` 选 TP+SP 或仅 TP。
- **Attention 后端**：`--attention_backend standard|fused|flash`，用于 FLOP 记账（flash 为 $O(s^2)$ 计，不物化 $s\times s$ 矩阵）。
- **Mixed precision 分支**：改变前向模型代码与 pipeline 指派（`llama_model.py` vs `gpt_model.py`）。
- **Readout**：`Chakra004Backend` 输出每 rank 一个 `workload.<rank>.et` + `workload.json` 通信组。

### 3.2 Simulator 端 — ASTRA‑sim analytical (`analytical.sh`)

- **后端**：`AstraSim_Analytical_Congestion_Aware`（预编译，`ANALYTICAL_SKIP_BUILD=1` 复用二进制）。
- **系统层调度**：`astra_system.json` 里：
  - collective：all‑reduce / all‑gather / reduce‑scatter 均为 **ring**，all‑to‑all 为 **direct**。
  - `roofline-enabled=1`，`peak-perf=312` TFLOP/s（A100 FP16）；`local-mem-bw=1560`。
  - `hardware-resource-capacity`：`gpu-comp=1, gpu-comm=1, gpu-recv=64`，即单 NPU 同时至多 1 个计算 + 1 个发送 + 64 个接收。
- **网络层**：`network.yml` 指向 `topology.txt`，`277 149 388` 表示 277 节点 / 149 交换机 / 388 链路（典型 in_dc 自建拓扑）。
- **事件并行（Plan C）**：`ASTRA_EVENT_PARALLEL_THREADS` / `ASTRA_EVENT_PARALLEL_MIN_EVENTS` 控制同时间戳事件批量派发，用于加速仿真，不影响结果。
- **通信组**：`workload.json` 已含 48 个并行组（DP×32 / TP×16 的排列）+ 大量单 rank 自用 group，由 STG 生成。

---

## 4. 覆盖度对照表

> 列 "状态" 含义：✅ 已覆盖；⚠️ 部分覆盖 / 近似；❌ 未覆盖

| # | 论文要素 | 状态 | 说明 |
|---|----------|------|------|
| 1 | 张量并行 $t=8$（节点内 all‑reduce 模型） | ✅ | STG 侧 `--tp=8`；TP all‑reduce 由 `BundledConvertChakra` 写入 ET，ASTRA‑sim 以 ring 执行 |
| 2 | 流水并行 $p$（层按 stage 连续切） | ✅ | `_create_pipeline_tensor_map` 正是这种分配 |
| 3 | 数据并行 $d$（每 batch 末 all‑reduce） | ✅ | `GraphDistributer` + `GradUpdater` 产生 DP all‑reduce |
| 4 | 序列并行 (SP) | ✅ | `--sp`，STG 侧已支持 |
| 5 | 同步权重更新 + pipeline flush | ⚠️ | **没有显式 flush 节点**；靠 DP all‑reduce 作为隐式屏障。若上游图存在跨 micro‑batch 的反向依赖收敛点，仿真上表现等效，但论文公式 $(p-1)/m$ 对应的 bubble **没有单独出具**。 |
| 6 | GPipe 全 F→全 B 调度 | ⚠️ | **没有显式调度策略**；Chakra ET 只表达依赖，ASTRA‑sim 的 workload 层按依赖就绪就执行。若 stage 上所有前向依赖先可用，实际轨迹接近 GPipe，但这是 **emergent 行为而非可配置** |
| 7 | PipeDream‑Flush (1F1B) | ❌ | STG 没有产生 1F1B 专用的依赖结构（如 warm‑up 阶段的不均匀前向数），ASTRA‑sim 也没有调度策略可选 |
| 8 | **交错式 1F1B** (v>1 chunks / device) | ❌ | `_create_pipeline_tensor_map` 是严格 "每 device 一段连续 block"；要实现 interleaved 需要非连续映射 {1,2,9,10} 这样的交错 chunk 分配，并且需要在 ET 里表达 chunk 粒度的 F/B 依赖 |
| 9 | **scatter / gather 优化** | ❌ | `BundledConvertChakra` 对相邻 stage 的 P2P 节点按完整张量 $b s h$ 写入；没有按 $1/t$ scatter + NVLink all‑gather 的拆分 |
| 10 | TP $f, g$ all‑reduce（两前两后） | ✅ | Megatron TP 的通信模式由模型图直接生成，ring 实现是合理近似 |
| 11 | 算子融合的吞吐收益 (11–19%) | ❌ | roofline 只用常量 `peak-perf`；融合 kernel 带来的真实算术强度提升不被模拟 |
| 12 | 激活重算 (+1 次前向) | ⚠️ | STG 有 `--activation_recompute` 参数，`qwen_32b.sh` 默认不打开；backward FLOP 是否按 "4× forward" 计取决于 GradUpdater 实现。ASTRA‑sim 无法看到显存压力对调度/卸载的反作用 |
| 13 | Micro‑batch $b$ 扫参 | ✅ | `MICROBATCH` 环境变量，可直接改 |
| 14 | Batch $B$ 扫参 | ✅ | `BATCH` 环境变量 |
| 15 | Mixed precision（FP16 compute + FP32 master） | ⚠️ | STG 有 `--mixed_precision`（切换模型代码和 pipeline map）；ASTRA‑sim 侧只用 A100 FP16 peak 一个值，不区分 FP16 / FP32 算力，也不建模 master‑weight 显存 |
| 16 | FSDP / weight sharded | ✅ | `--weight_sharded` + fsdp 维度替换；对应 ZeRO‑3 场景 |
| 17 | DGX A100 节点内 NVLink / NVSwitch | ❌ | 当前 `topology.txt` 是自定义单层结构，没有 node 内/node 间 带宽差；论文中正是 NVLink (600 GB/s) vs IB (25 GB/s) 的层次差驱动 "TP 在 node 内" 这个 takeaway |
| 18 | 8× IB HCA 并行 P2P | ❌ | 没有 multi‑NIC 概念；每对节点只有一条逻辑链路 |
| 19 | 3 级胖树（850 switches） | ❌ | 277 节点 / 149 交换机的 in_dc 拓扑是别的规模；要做 Selene 级对照需要新的拓扑文件和规模 |
| 20 | ns‑3 级 PFC / ECN / 拥塞反馈 | N/A | 该脚本是 analytical；如需可切换到 `ns3.sh`，但论文里只用于定性讨论 |
| 21 | checkpoint I/O（13.8 TB、1 TB/s 读） | ❌ | 当前栈不模拟 I/O |
| 22 | Bubble fraction 直接报告 | ❌ | ASTRA‑sim 输出 `sys[N] finished` 的 exposed comm / compute utilization，但没有 "bubble 时间" 的显式度量 |
| 23 | 每层/每 stage 的详细 breakdown | ⚠️ | 需自行解析 `run_analytical.log` |
| 24 | 训练时间估算式 $\approx 8TP/(nX)$ | ✅ | 用实测 $X$ 直接代入即可 |

---

## 5. 关键缺失点详述

### 5.1 显式 1F1B / Interleaved 调度（论文核心贡献）

**现状**：STG 产出的 Chakra ET 仅描述数据依赖，不强制执行顺序。ASTRA‑sim 的 workload 层用 `HardwareResource`（见 `astra_system.json` 的 `hardware-resource-capacity`）控制同时并发的计算/通信数，按就绪顺序派发。

**这意味着**：
- 当 $gpu\text{-}comp = 1$ 时，一张卡上不会同时跑两个计算节点，但**对 micro‑batch 间的前后向顺序没有偏好**——实际仿真轨迹取决于图展开后的节点 ID 顺序（通常是先前向后反向，即 GPipe‑like）。
- 无法通过配置切换到 "1F1B 稳态只留 $p$ 份激活" 的调度，也就无法测到 1F1B 相对 GPipe 的 **激活显存下降** 与 **bubble 相同但 steady state 更稳** 的特性。
- **Interleaved（$v > 1$）根本无法建模**，因为 `_create_pipeline_tensor_map` 只支持把 block 范围连续分给 stage。

**补齐要点**：
1. STG 扩展：新增 `--pipeline_virtual_stages v`；`_create_pipeline_tensor_map` 改为把 block $i$ 映射到 `(i // block_per_chunk) % p`，按 chunk 串接；同时为每个 chunk 插显式 F/B 边界节点。
2. ASTRA‑sim 扩展：在 workload 层增加 "pipeline scheduler" 抽象层，可选 `gpipe | 1f1b | 1f1b-interleaved`，在 ready 队列里按策略优先出队；或在 STG 生成阶段直接把调度顺序烘焙进 ET 的依赖边。
3. 仿真结果里补出 "bubble time" 与 "in‑flight micro‑batch 数" 两条指标，方便直接对照公式 $(p-1)/m$ 与 $\frac{1}{v}(p-1)/m$。

### 5.2 Scatter/Gather 通信优化

**现状**：BundledConvertChakra 写出的 stage‑间通信节点是按完整张量大小 $b\cdot s\cdot h$ 计数的。论文中在 $t = 8$ 时，跨 IB 的点对点通信被减到 $1/8$，并配合节点内 NVLink 做 all‑gather 补齐。

**影响**：当前仿真 **高估** 了 pipeline 相邻 stage 间跨节点通信；对于 8×IB 的 DGX A100 更是差得多（论文 P2P 二分带宽实测 892 GB/s，分摊到 8 NIC = 每 NIC ~112 GB/s）。

**补齐要点**：
- STG：BundledConvertChakra 里按 "相邻 pipeline 边界 + TP 组信息"，把 P2P 节点替换为 `(scatter → inter-node P2P 1/t size → intra-node all-gather)` 三段式。
- ASTRA‑sim：需要能区分 intra‑node 与 inter‑node 链路（依赖下面 §5.5 的层次拓扑）。

### 5.3 激活重算的显存 + FLOP 双重建模

**现状**：
- `qwen_32b.sh` 不传 `--activation_recompute`；STG 的 GradUpdater 是否默认乘 4（3× forward + 1× extra forward）需要代码层确认。
- `main.py` 的 `--print_gpu_vram` 可以粗算参数 / 激活 / 梯度显存，但没有基于 ASTRA‑sim 执行轨迹的峰值显存追踪。
- ASTRA‑sim 本身有 `LocalMemUsageTracker`（见 `CLAUDE.md` 描述），但当前脚本没开 VRAM 约束。

**影响**：论文 §5.6 展示 "激活重算在 batch 大时反而比不重算快 2×"——当前栈无法观察这种 trade‑off。

**补齐要点**：
- 把 `--activation_recompute` 显式接入 `qwen_32b.sh`，并确认 GradUpdater 在开启时会对 backward 前插入一次额外 forward 节点。
- 在 ASTRA‑sim 输出里报告 `peak_vram / rank`，并允许设置 `vram-capacity-gb=80` 触发 OOM 报错。
- 加一个对照实验模板：`activation_recompute ∈ {true, false}` × `batch ∈ {B₁, B₂}`。

### 5.4 算子融合 / kernel 算术强度

**现状**：`roofline-enabled=1, peak-perf=312` 用一个固定上限折算 GEMM 时间；FFN、attention 内部融合带来的算术强度提升不可见。

**影响**：论文 §5.8 指出融合使吞吐提高 11%–19%。这是论文 52% 峰值 vs DeepSpeed 36% 峰值的主要来源之一。

**补齐要点**：
- 在 `astra_system.json` 增加 `achievable-frac` 或按算子类型配置的 "effective TFLOP/s" 映射表；或在 Chakra ET 写出时把融合 kernel 的等效 runtime 直接写入 `duration_micros`（需要 STG 层做拟合）。
- 或接入外部 profile（真实 GPU 测出的每 op runtime 表），用 ET replay 时查表。

### 5.5 层次化网络拓扑（NVLink / NVSwitch / IB / fat‑tree）

**现状**：`topology.txt` 第一行 `277 149 388`，是自定义扁平拓扑，链路带宽/延迟通过文件指定。没有 "节点内 NVSwitch 600 GB/s、节点间 IB 25 GB/s" 的分层。

**影响**：
- Takeaway #1（"TP ≤ node 内 GPU 数"）在平坦拓扑里失效——因为节点内外带宽一致。
- scatter/gather 优化依赖 "跨节点贵、节点内便宜" 的前提，如果不分层，收益无法体现。

**补齐要点**：
- 用 `generate_topology.py`（同目录已有）生成分级结构：每个 DGX 节点 8 GPU + 1 NVSwitch（600 GB/s）、然后 leaf→spine→core 的 IB 胖树（200 Gbps/link）。`topology.txt` 已支持 switch 列表，可扩展。
- 增加 "每 GPU 多 NIC" 建模（对应 8×HCA）；当前 analytical 可通过同源节点多链路模拟，但需要 Scatter/Gather 的上层拆分才有意义。

### 5.6 Mixed precision 的精度分档

论文全程 FP16 混合精度；当前栈用 peak=312 TFLOP/s (A100 FP16) 统一处理，没有区分：
- FP16 矩阵乘（peak 312）与 FP32 reduction（peak 156）；
- master weight 的 FP32 拷贝在 optimizer step 时的带宽压力。

可作为次优先级补齐项。

### 5.7 显式的 bubble / 通信 breakdown 报告

论文结论很多是基于 "bubble fraction"、"有效二分带宽" 这类派生指标。当前仿真只输出 `sys[N] finished` 各 rank 的 wall cycles / exposed comm 等——需要额外脚本解析，而且无法直接对应论文图 6 / 图 11 / 图 12 的 y 轴。

**补齐要点**：在 ASTRA‑sim 主循环结束时新增一次统计 pass，打印：
- `bubble_time / iteration_time`
- `point_to_point bisection GB/s`
- `all-reduce bisection TB/s`
- `per‑stage compute_util / comm_util`

---

## 6. 建议补齐路线（按优先级）

| 优先级 | 动作 | 目标 | 改动位置 |
|--------|------|------|----------|
| **P0** | 接入显式 pipeline scheduler（1F1B / GPipe 切换 + virtual stages $v$） | 能复现论文 §2.2 & 图 4、图 12 的 bubble 对比 | STG `graph_distributer.py` + `convert_chakra.py`；ASTRA‑sim workload 层（或在 ET 里烘焙顺序） |
| **P0** | 层次化拓扑（node 内 NVSwitch / node 间 IB 胖树） | 区分 TP 在 node 内 vs 跨 node 的代价 | `generate_topology.py` + `topology.txt` |
| **P1** | scatter / gather 通信优化 | 复现论文 §4.1 & 图 18 的 11% 加速 | STG `convert_chakra.py` 里改 stage‑间 P2P 生成；依赖层次拓扑 |
| **P1** | `--activation_recompute` 打通 + 显存报告 | 复现 §5.6 激活重算权衡 | STG `grad_updater.py`、`vram_counting.py`；`astra_system.json` 加 vram 容量 |
| **P2** | 算子融合/有效算力表 | 复现 §5.8（融合 +11%~19%） | ASTRA‑sim workload 层加 "per‑op achievable TFLOP/s" 表；或 STG 产 ET 时写入 duration |
| **P2** | bubble / 有效二分带宽自动汇总 | 直接画图对比 | ASTRA‑sim main 结束前增加 summary，或者仿真日志解析脚本 |
| **P3** | 多 IB NIC 建模 | 与 scatter/gather 联动，复现 892 GB/s P2P | analytical 后端层 |
| **P3** | checkpoint I/O、mixed‑precision 分精度算力 | 完整度 | 系统层新增资源类型 |

---

## 7. 当前脚本能直接给出的对照实验（不需改代码）

即使暂不补齐上述缺口，现有栈仍可做的有意义实验：

1. **扫 $(t, p)$ 组合**（takeaway #1）：固定 GPU 数和 batch，扫 $(t, p) \in \{(1,32), (2,16), (4,8), (8,4), (16,2)\}$，看 per‑GPU throughput；但由于 **无层次拓扑**，结论可能和论文相反，属于已知偏差。
2. **扫 micro‑batch $b$**：同 $(p, t)$ 下改 `MICROBATCH`，对照 Figure 7/8 的趋势。
3. **扫 batch $B$ 与流水深度** 对 bubble 占比的影响（需手动算 `(p-1)/m`，不是直接读出）。
4. **LocalSGD 间隔对通信减少的影响**：改 `SGD=localsgd, ITERATION=N, DP_LOCAL_SGD_INTERVAL=K`，观察 DP all‑reduce 的频次下降。
5. **跨 in_dc 拓扑规模**（已有 277/149/388）vs 更大规模，测量 ASTRA‑sim 自身的扩展性。

以上实验可以先跑出一份基线报告，待 P0 / P1 改造完成后再作 "前/后" 对比，用以量化补齐各项收益。
