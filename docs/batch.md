# demo3 中 `batch` / `micro_batch` 的修正语义分析

## 修正后的结论

- `batch` 在当前 demo3/STG 生成链路里，更接近**标准训练里的 GBS（global batch size）**，不是“每个 DP rank 的本地 step batch”
- `micro_batch` 也不是“每个 DP rank 的 local micro-batch size”，而更接近**整个 DP 组这一小批的大小**
- 模板张量 shape 大量写成 `Batch/dp`；而 `MicroBatchReplicator` 会在每个 `mb{i}` 子图里把 `Batch -> MicroBatch`，所以：
  - 每个 DP rank 的**本地 step batch** = `batch / dp`
  - 每个 DP rank 的**本地 micro-batch size** = `micro_batch / dp`
- 当前实现会把一个 step 切成 `batch / micro_batch` 个 micro-batch，因此：
  - `GAS = batch / micro_batch`
  - `GBS = batch`
- **2026-04-09 修复后**：梯度会先在本地跨所有 `mb*` micro-batch 合并，再只对合并后的 step 级梯度做一次 DP 同步。  
  因此这里的 `micro_batch` 仍然决定 PP 的切分粒度，但**不再决定 DP AllReduce 的发生次数**。

## 分析过程

### 1. 从运行脚本确认实际传参

`llama3_8b.sh` 传入：

- `--dp 4 --tp 1 --pp 4`
- `--batch 128`
- `--micro_batch 2`

所以问题的关键不是抽象定义，而是这套 STG/Chakra/Astra-Sim 栈里这两个参数在当前实现中的真实语义。

### 2. `main.py` 如何使用 `batch` / `micro_batch`

在 `micro2024/symbolic_tensor_graph/main.py` 的 llama 路径中：

1. `Batch` 和 `MicroBatch` 直接由 `args.batch` / `args.micro_batch` 填入符号表
2. 图变换顺序是：
   1. `MicroBatchReplicator.apply(...)`
   2. `GradUpdater.apply(...)`
   3. `GraphDistributer.apply(...)`
   4. `BundledConvertChakra.apply(...)`

这说明会先做 micro-batch 复制，再拼接梯度依赖，再做分布式切分和 Chakra 转换。

### 3. 模板 shape 证明 `Batch` 不是 per-rank local batch

module3 / tpsp 模板里的激活 shape 大量写成 `Batch/dp`，例如：

- `embedding.csv`
- `layer_norm.csv`
- `loss.csv`

这意味着模板层面默认的本地 batch 维就是 **`Batch/dp`**，而不是 `Batch` 本身。  
所以在这条生成链路里，更自然的解释是：

- `Batch`：step 级别、DP 切分前的 batch 记账单位
- `Batch/dp`：每个 DP rank 真正持有的本地 batch 维

### 4. `MicroBatchReplicator` 决定了 micro-batch 个数和每个子图的 batch 维

`MicroBatchReplicator.apply(...)` 直接按 `Batch / MicroBatch` 复制训练图，形成 `mb0`, `mb1`, ..., `mbN` 这些子图。

同时，它在每个复制出的 `mb{i}` 子图里把符号 `Batch` 替换成 `MicroBatch`。  
由于模板原本写的是 `Batch/dp`，替换以后单个 micro-batch 子图里的本地 batch 维就变成了：

- `MicroBatch/dp`

因此这里有两个非常重要的结论：

- `batch / micro_batch` 才是一个 step 内的 micro-batch 个数，也就是这里的 `GAS`
- `micro_batch` 本身不是每卡 local micro-batch，而是**DP 组这一小批的大小**

### 5. `GradUpdater` 和修复后的 `MicroBatchReplicator` 现在都按 step 工作

`GradUpdater.apply(...)` 会对每个可训练权重做一次 step 级更新。  
而修复后的 `MicroBatchReplicator.apply(...)` 会把**同步前的本地梯度**先跨所有 `mb*` 合并，再把 DP 同步推迟到最终权重更新前。

这意味着现在同时满足：

- optimizer update 是 step 级的
- DP 通信也只在 step / GBS 末尾发生一次

### 6. 通信匹配和新的 ET 证明：DP AllReduce 已经推迟到 step 末尾

`coll_comm_matcher.py` 会把 `PARTIALSUM -> DUPLICATED` 的转换匹配成 `ALL_REDUCE`。

重新生成 `llama/*.et` 后，可以看到：

- 仍然有 64 个 micro-batch：`mb0` 到 `mb63`
- 但 `llama.0.et` 只剩 9 个 collective：
  - `transformer.0._sharded_weight@1_X2_COMM` 到 `transformer.7._sharded_weight@1_X2_COMM`
  - `in_emb.w@1_X2_COMM`
- 扫描全部 16 个 `llama.*.et` 后，`mb*` 前缀 collective 数都是 **0**
- 中间 PP stage 的每个 rank 只保留 8 个 step 级 collective
- 首 stage 多 1 个 `in_emb.w@1_X2_COMM`
- 末 stage 多 1 个 `out_emb.w@1_X2_COMM`

在当前脚本参数 `batch=128, micro_batch=2` 下：

- `128 / 2 = 64`

与 ET 中看到的 64 个 `mb*` 子图完全一致，说明 **PP/GAS 切分没有变**。

因此可以确认：**当前做法已经变成“每个 GBS 内先本地累积，再只做一次 step 级 DP AllReduce”。**

### 7. Astra-Sim 侧的额外 caveat

继续追到 Chakra/Astra-Sim，当前 `Workload.cc` 仍然用单一的 `comm_group` 发 collective，尚未真正按每个节点自己的 `pg_name` 动态切换通信组。

这意味着：

- 生成图里出现的 DP collective 会真正按当前 communicator 参与同步
- 不能只靠改 ET 节点上的 `pg_name` 来模拟真正的 LocalSGD

## 最终语义整理

在当前这套实现里，最稳妥的理解是：

| 参数 | 当前实现中的真实含义 |
| --- | --- |
| `dp` | DP degree |
| `pp` | PP stage 数 |
| `batch` | step 级 batch，语义上更接近标准训练里的 `GBS` |
| `batch / dp` | 每个 DP rank 的本地 step batch |
| `micro_batch` | 每个 micro-batch 在 DP 切分前的大小，可理解为“DP 组的 micro-batch size” |
| `micro_batch / dp` | 标准语义下每个 DP rank 的 local micro-batch size |
| `batch / micro_batch` | 当前实现里的 `GAS` / micro-batch 个数 |

## 和标准 `GBS / DP / local micro batch / GAS / PP` 的关系

标准训练里通常有：

- `GBS = DP * local_micro_batch * GAS`

而在当前实现中：

- `local_micro_batch = micro_batch / dp`
- `GAS = batch / micro_batch`

代入可得：

- `GBS = dp * (micro_batch / dp) * (batch / micro_batch) = batch`

也就是说，**在这条生成链路里，`batch` 更接近标准训练里的 `GBS`。**

如果你已经有一组标准训练参数：

- `GBS`
- `DP`
- `local_micro_batch`
- `GAS`
- `PP`

那么映射到这个脚本，最接近的写法是：

- `--dp = DP`
- `--pp = PP`
- `--batch = GBS`
- `--micro_batch = DP * local_micro_batch`

并且：

- `batch / micro_batch = GAS`

## 为什么它现在和常见大模型训练直觉更接近

在很多标准训练实现里：

- `micro_batch` 主要是 PP 的切分单位
- 多个 micro-batch 会先本地累积梯度
- 只在 step 末尾做一次 DP AllReduce

修复后的这套实现已经和这条直觉一致。这里 `micro_batch` 主要影响：

1. PP 流水线切分粒度
2. 一个 step 内会发射多少个 micro-batch（也就是 `GAS`）

因此现在可以按更标准的 3D 并行直觉来理解这个脚本：  
**PP 仍然按 `GAS` 个 micro-batch 发射，但 DP 同步只发生在 step 末尾。**

## 用当前脚本参数举例

当前脚本设置：

- `dp = 4`
- `pp = 4`
- `batch = 128`
- `micro_batch = 2`

按当前实现，它的含义更接近：

- `GBS = 128`
- 每个 DP rank 的本地 step batch = `128 / 4 = 32`
- micro-batch 个数 = `128 / 2 = 64`
- 每个 DP rank 的 local micro-batch size = `2 / 4 = 0.5`
- 这 64 个 micro-batch 会进入 PP 流水线
- 但整个 GBS 只会触发一轮 step 级 DP AllReduce 波次

也就是说，这个配置从标准训练语义看其实并不自然。  
如果你的本意是“每卡 local micro-batch = 2”，那么在 `dp=4` 时脚本里应该写：

- `--micro_batch 8`

而不是 `2`。

## 如果要按常见经验设 `GAS ≈ PP * 4`

在这套实现里，仍然可以沿用这个经验，但应写成：

- `batch / micro_batch ≈ PP * 4`

等价地：

- `batch ≈ (PP * 4) * micro_batch`

如果改用标准训练里的 `local_micro_batch` 来表示，则因为：

- `micro_batch = dp * local_micro_batch`

所以也可以写成：

- `batch ≈ DP * PP * 4 * local_micro_batch`

注意：这只是在匹配“pipeline 里要有多少个 micro-batch”这条经验；而当前实现现在**确实**会像标准训练那样只在 step 末尾做一次 DP 同步。

## 如果目标是“每个 GBS 只进行一次 DP AllReduce”

现在已经是这种行为了：

1. PP 仍然会按 `batch / micro_batch` 发射多个 micro-batch
2. 这些 micro-batch 的本地梯度会先在生成图里合并
3. 只对合并后的 step 级梯度插一次 DP AllReduce

如果你把 `batch = micro_batch`，那只是把 `GAS` 进一步降成 1，让 PP 也只发一个 micro-batch；  
**它不再是实现“每个 GBS 只同步一次”所必需的 workaround。**

## 一句话总结

**这条 demo3 生成链路里，`batch` 更接近标准训练中的 `GBS`，`micro_batch` 更接近“DP 组的 micro-batch 大小”，每个 rank 真正的 local micro-batch 是 `micro_batch / dp`；PP 仍按 `GAS = batch / micro_batch` 个 micro-batch 发射，但 DP AllReduce 已经被推迟到 step / GBS 末尾，只发生一次。**
