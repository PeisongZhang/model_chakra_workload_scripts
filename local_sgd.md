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
- `--seq 8192 --batch 128 --micro_batch 8`
- `--model_type llama --weight_sharded 0`

默认情况下，这会在 `llama/` 下生成每个 rank 的 Chakra ET（如 `llama.0.et`）和通信组文件 `llama.json`；  
若 `NUM_ITERATIONS > 1` 或 `DP_LOCAL_SGD_INTERVAL > 1`，脚本默认改为输出到 `llama_local_sgd/`。

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

### 补充：这里的 `micro_batch` 语义与常见 PP 训练定义不一致

这次进一步核查后，可以更明确地说：**当前 STG / Chakra workload 中的 `micro_batch`，并不是多维并行训练里常说的“PP 调度 micro-batch”语义。**

在常见的 DP+PP 训练定义里：

- `batch` 通常表示 **global batch size**；
- `micro_batch` 通常表示 **每个 pipeline slot / 每个 rank 一次前反向处理的样本数**；
- 因此一个 optimizer step 内的 PP micro-batch 数通常应为：

  `num_micro_batches = batch / (dp * micro_batch)`

对本例 `dp=4, batch=128, micro_batch=2`，若按这个标准定义，应得到：

- `128 / (4 * 2) = 16` 个 PP micro-batches。

但当前 STG 实现并不是这样算的。`main.py` 只是把 `Batch=args.batch`、`MicroBatch=args.micro_batch` 原样塞进符号表；随后 `MicroBatchReplicator.apply(...)` 直接用：

- `num_batches = Batch / MicroBatch`

来复制图，并生成 `mb0`, `mb1`, ... 这些前缀。也就是说，本例实际被展开成：

- `128 / 2 = 64` 个 `mb*`

而**没有除以 `dp`**。这与标准 PP micro-batch 计数方式不同。

进一步地，修复后的梯度通信插入逻辑会先把所有 `mb*` 的本地梯度合并，再把 `PARTIALSUM -> DUPLICATED` 的 step 级转换映射成 `ALL_REDUCE`。因此 ET 中看到的 collective，更准确地表示的是：

- 一个 iteration / step 末尾的 DP `ALL_REDUCE`；
- 而不是“每个 `mb*` 都触发一次 DP 同步”。

所以，从建模语义上更准确的说法是：

- 这里的 `mb*` 更接近 **被 STG 显式复制出来的 local training chunk / local step**；
- 不是标准 1F1B / GPipe 文献语境下、服务于 PP 调度和梯度累积的那种 micro-batch。

这也是为什么**默认 workload** 不适合直接拿来表示 LocalSGD：它默认假设**每个 iteration 结束都进行一次同步 DP 梯度聚合**。

---

## 3) 现在怎么生成 LocalSGD DP workload

### 直接用 `main.py`

```bash
python3 main.py \
  --output_dir generated_local_sgd/ \
  --output_name workload.%d.et \
  --dp 4 --tp 1 --pp 4 \
  --batch 128 --micro_batch 8 \
  --num_iterations 4 \
  --dp_local_sgd_interval 2
```

上面的语义是：

- trace 中包含 4 个连续 iteration
- 第 2、4 个 iteration 保留 DP `ALL_REDUCE`
- 第 1、3 个 iteration 只执行本地更新，不做 DP 同步

### 直接用 `llama3_8b.sh`

```bash
NUM_ITERATIONS=4 DP_LOCAL_SGD_INTERVAL=2 ./llama3_8b.sh
```

脚本会自动：

- 把新参数透传给 `main.py`
- 在开启 LocalSGD 时默认输出到 `llama_local_sgd/`
- 保留默认同步 DP 的原有 `llama/` 输出路径

## 4) 当前实现方式

当前实现选择的是**BundledHybridGraph 后处理**：

1. 先按现有 STG 流程生成“单个 step / iteration”的 Chakra 图
2. 将该 step 复制成多个 iteration
3. 在非同步 iteration 上删除 `parallel_dim == dp` 的 `ALL_REDUCE`
4. 重写 `data_deps`，让权重更新直接依赖本地梯度路径
5. 插入零开销 barrier 节点，把各 iteration 串行化，保证 Astra-Sim 按顺序执行

这样不会改变现有默认同步 DP 行为，也不需要改 Astra-Sim 运行时去循环重复单个 ET。
