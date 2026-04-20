# `llama3_8b.sh` 生成 DP workload 与 LocalSGD 的分析 / 实现

## 结论

现在已经可以直接生成 **LocalSGD DP** workload。  
新增两个入口参数：

- `--num_iterations N`：在同一份 trace 里生成 `N` 个连续 iteration
- `--dp_local_sgd_interval K`：只在满足 `(iteration + 1) % K == 0` 的 iteration 上保留 DP `ALL_REDUCE`

默认值仍是：

- `--num_iterations 1`
- `--dp_local_sgd_interval 1`

因此默认行为依然是**同步 DP**；只有把 `dp_local_sgd_interval` 设成大于 `1` 时，才会变成 LocalSGD 风格的“多次本地 iteration 才做一次 DP 同步”。

---

## 1) 当前脚本如何生成 DP workload

`llama3_8b.sh` 关键调用（见 `llama3_8b.sh`）：

- `python3 main.py`
- `--dp 4 --tp 1 --pp 4`
- `--seq 8192 --batch 128 --micro_batch 2`（per-rank micro-batch，Megatron 习惯）
- `--model_type llama --weight_sharded 0`

脚本实际输出到由 `ATTENTION`, `SGD`, `LAYER`, `ITERATION`, `BATCH`, `MICROBATCH`, `SEQUENCE` 拼成的目录，例如默认参数下为
`standard_standard_32_8_128_2_8192/`，内部文件名是 `workload.%d.et` 和 `workload.json`（见 `llama3_8b.sh:38`）。
开启 LocalSGD 时通常写成 `SGD=local`（或 `local_sgd`），对应目录前缀就会带上那个标识。

在 STG 主流程（`symbolic_tensor_graph/main.py`）中，执行顺序是：

1. 组装模型图（llama）
2. `MicroBatchReplicator.apply(...)`
3. `GradUpdater.apply(...)`
4. `GraphDistributer.apply(...)`
5. `BundledConvertChakra.apply(...)` 输出 ET

---

## 2) 为什么默认仍然是同步 DP（不是 LocalSGD）

1. `main.py` 虽然现在有了 `--num_iterations` 和 `--dp_local_sgd_interval`，但默认值分别是 `1` 和 `1`。  
2. 在默认值下，`convert_chakra.py` 仍会按张量并行语义插入 DP `ALL_REDUCE`。  
3. `coll_comm_matcher.py` 的规则里，`PARTIALSUM -> DUPLICATED` 仍然会映射到 `ALL_REDUCE`；LocalSGD 只是对**非同步 iteration** 做后处理裁剪。  
4. 因此不显式打开 LocalSGD 参数时，trace 仍然表示同步 DP。

因此：默认链路仍然会周期性地在图中保留 DP all-reduce；  
只有显式打开新参数时，才会在生成阶段切换成 LocalSGD workload。

### 补充：`micro_batch` 语义已对齐 Megatron 习惯

`main.py` 现在按 Megatron / GPipe 惯例处理 `--micro_batch`：它表示**每张卡一次前反向处理的样本数**，一个 optimizer step 内的 micro-batch 数为

  `num_micro_batches = batch / (dp * micro_batch)`

具体实现：`main.py` 在写入符号表时令 `MicroBatch = args.micro_batch * args.dp`，使各张量保留的 `Batch/dp` 形状在替换后等于 `args.micro_batch`。`MicroBatchReplicator.apply(...)` 仍按 `Batch / MicroBatch` 计数，因而总份数自动落到 `batch / (dp * micro_batch)`。

对本例 `dp=4, batch=128, micro_batch=2`：

- `128 / (4 * 2) = 16` 个 `mb*`；
- 每个 `mb*` 在每张卡上携带 2 个样本。

梯度通信插入逻辑会先合并同 step 内所有 `mb*` 的本地梯度，再把 `PARTIALSUM -> DUPLICATED` 的 step 级转换映射成 `ALL_REDUCE`。因此 ET 中的 DP collective 表示**每个 iteration 末尾一次** DP 同步，而不是“每个 `mb*` 都触发一次 DP 同步”。

默认 workload（`ITERATION=1, DP_LOCAL_SGD_INTERVAL=1`）仍是同步 DP；要表达 LocalSGD 需显式打开下面两个参数。

---

## 3) 现在怎么生成 LocalSGD DP workload

### 直接用 `main.py`

```bash
python3 main.py \
  --output_dir generated_local_sgd/ \
  --output_name workload.%d.et \
  --dp 4 --tp 1 --pp 4 \
  --batch 128 --micro_batch 2 \
  --num_iterations 4 \
  --dp_local_sgd_interval 2
```

上面的语义是：

- trace 中包含 4 个连续 iteration
- 第 2、4 个 iteration 保留 DP `ALL_REDUCE`
- 第 1、3 个 iteration 只执行本地更新，不做 DP 同步

### 直接用 `llama3_8b.sh`

```bash
ITERATION=4 SGD=local DP_LOCAL_SGD_INTERVAL=2 ./llama3_8b.sh
```

关键点（见 `llama3_8b.sh`）：

- `ITERATION` 透传到 `--num_iterations`
- `DP_LOCAL_SGD_INTERVAL` 透传到 `--dp_local_sgd_interval`
- `SGD` 只是一个标签：默认 `standard` 会强制 `DP_LOCAL_SGD_INTERVAL=1`（同步 DP）；
  非 `standard`（例如 `local`、`local_sgd`）时，`DP_LOCAL_SGD_INTERVAL` 默认等于 `ITERATION`，也就是“一个 trace 里只做一次 DP 同步”
- 输出目录名自动带上 `SGD` 标签，方便同目录里同时放同步 DP 和 LocalSGD 两份 trace

## 4) 当前实现方式

当前实现选择的是**BundledHybridGraph 后处理**：

1. 先按现有 STG 流程生成“单个 step / iteration”的 Chakra 图
2. 将该 step 复制成多个 iteration
3. 在非同步 iteration 上删除 `parallel_dim == dp` 的 `ALL_REDUCE`
4. 重写 `data_deps`，让权重更新直接依赖本地梯度路径
5. 插入零开销 barrier 节点，把各 iteration 串行化，保证 Astra-Sim 按顺序执行

这样不会改变现有默认同步 DP 行为，也不需要改 Astra-Sim 运行时去循环重复单个 ET。
