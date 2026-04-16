# FlashAttention 在 STG 中的实现

## 1. 背景：当前 STG 的注意力建模

`llama3_8b.sh` 脚本调用 Symbolic Tensor Graph (STG) 生成器 (`main.py`)，为 Llama-3 8B 模型生成 Chakra 执行轨迹。STG 通过 CSV "sharding spreadsheet" 定义各算子的符号计算图，其中注意力机制由两部分组成：

- **surrounding 图** (`group_query_attention_surrounding.csv`)：定义 QKV 投影（Wqkv 矩阵乘法）、输出投影（Wo 矩阵乘法）及其梯度
- **kernel 图** (`group_query_attention_kernel_*.csv`)：定义注意力核心计算（Q·K^T → softmax → attn·V）

当前提供了三种 kernel 变体：

| 文件 | 建模方式 | FLOPs | 是否物化 S×S 矩阵 |
|------|---------|-------|-------------------|
| `_kernel.csv` | 显式 Q·K^T 和 attn·V 两步矩阵乘法 (Einsum) | O(S²·D) ✓ | **是** — 输出 shape 含 `Seq × Seq` |
| `_kernel_fused.csv` | 单一 CUSTOM 算子，FLOPs = B·S·D·H·3 | O(S·D) ✗ | **否** — 输出 shape 为 O(S·D) |
| `_kernel_flash.csv` (**新增**) | 单一 CUSTOM 算子，FLOPs = 4·B·H·S²·D | O(S²·D) ✓ | **否** — 输出 shape 为 O(S·D) |

**问题**：当前 `_kernel_fused.csv` 虽然避免了 S×S 矩阵物化，但其 FLOPs 模型是 O(S·D·H)，**错误地将注意力计算建模为线性复杂度**，丢失了注意力机制的 O(S²) 特性。

## 2. FlashAttention 原理

FlashAttention (Dao et al., 2022; FlashAttention-2, 2023) 是一种 **IO-aware** 的精确注意力算法，核心思想：

### 2.1 前向传播

标准注意力的计算流程：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

标准实现需要将完整的 S×S 注意力矩阵 $P = \text{softmax}(QK^T / \sqrt{d})$ 写入 HBM（高带宽显存），内存占用 O(B·H·S²)。

FlashAttention 采用**分块 (tiling)** 策略：
1. 将 Q, K, V 分成大小为 $B_r × B_c$ 的块
2. 在 SRAM（片上缓存）中逐块计算注意力分数
3. 使用 **online softmax** 技巧（维护 running max 和 running sum）逐步累积结果
4. 仅将最终输出 O 和 **logsumexp 统计量** (每行一个标量, shape = B×H×S) 写回 HBM

**结果**：
- FLOPs 不变：与标准注意力相同，为 $4 \cdot B \cdot H \cdot S^2 \cdot d$（Q·K^T 和 attn·V 各贡献一半）
- 内存：从 O(B·H·S²) 降至 O(B·H·S·d)，**不物化 S×S 注意力矩阵**
- IO 复杂度：从 O(B·H·S²·d + B·H·S²) 降至 $O(B \cdot H \cdot S^2 \cdot d^2 / M)$，其中 M 为 SRAM 大小

### 2.2 反向传播

FlashAttention 的反向传播**不存储前向传播的 S×S 注意力矩阵**，而是：
1. 从 Q, K, V 和 logsumexp 统计量**重新计算**注意力分数（recomputation）
2. 同样使用分块策略计算 dQ, dK, dV
3. 以额外的计算（~1× forward FLOPs 的 recomputation）换取大幅的内存节省

**反向 FLOPs**：标准反向 ~8·B·H·S²·d + 重计算 ~2·B·H·S²·d ≈ **10·B·H·S²·d**

反向/前向 FLOPs 比 ≈ **2.5×**，这是 FlashAttention 的特征。

### 2.3 对比总结

| 特性 | 标准注意力 | Fused Kernel (当前) | FlashAttention (新增) |
|------|-----------|-------------------|--------------------|
| 前向 FLOPs | 4·B·H·S²·d | B·S·D·H·3 (欠估) | 4·B·H·S²·d ✓ |
| 反向 FLOPs | 8·B·H·S²·d | B·S·D·H·6 ×3 (欠估) | 10·B·H·S²·d ✓ |
| 峰值内存 | O(B·H·S²) (注意力矩阵) | O(B·H·S·d) | O(B·H·S·d) ✓ |
| 物化 S×S | 是 | 否 | 否 ✓ |
| S² 复杂度建模 | ✓ (通过 Einsum) | ✗ (线性近似) | ✓ (CUSTOM op_attr) |

## 3. STG 中的实现方式

### 3.1 新增 CSV 文件

为每个 sharding 目录创建了 `group_query_attention_kernel_flash.csv`：

```
sharding_spreadsheets/module3/
├── tpsp/group_query_attention_kernel_flash.csv        # Llama (GQA, SLICE)
├── tpsp_fsdp/group_query_attention_kernel_flash.csv   # Llama + FSDP (GQA, Identical)
├── tp/group_query_attention_kernel_flash.csv           # TP-only (GQA, Identical)
├── tpsp_gpt/group_query_attention_kernel_flash.csv    # GPT (MHA, TP+SP)
├── tp_gpt/group_query_attention_kernel_flash.csv      # GPT (MHA, TP-only)
├── tpsp_moe/group_query_attention_kernel_flash.csv    # MoE (MHA, TP+SP)
└── tp_gpt_moe/group_query_attention_kernel_flash.csv  # MoE (MHA, TP-only)
```

### 3.2 Kernel CSV 结构说明

以 `tpsp/group_query_attention_kernel_flash.csv` (GQA 变体) 为例：

```
前向:
  q  ─────────┐
  k → k1 (I) ─┤── qkv (CUSTOM, FLOPs = 4*B/dp*H/tp*S/cp*S*D/Head)
  v → v1 (I) ─┘     输出 shape: (B/dp, S/cp, D/Head, H/tp)  ← 无 S×S

反向:
  dqkv ──┬── dq  (CUSTOM, FLOPs = 4*B/dp*H/tp*S/cp*S*D/Head)
         ├── dk1 (CUSTOM, FLOPs = 3*B/dp*H/tp*S/cp*S*D/Head) → dk (SLICE)
         └── dv1 (CUSTOM, FLOPs = 3*B/dp*H/tp*S/cp*S*D/Head) → dv (SLICE)
```

关键设计点：

1. **FLOPs 公式中包含 `Seq/cp * Seq`**：正确建模了 O(S²) 的注意力计算复杂度。其中 `Seq/cp` 是本地 query 序列长度（经过 context parallelism 切分），`Seq` 是完整的 KV 序列长度（经过 allgather 恢复）。

2. **反向 FLOPs 分配**：
   - dq: 4× (包含 recomputation 份额)
   - dk: 3×
   - dv: 3×
   - 总计: 10× B·H·S²·d，反向/前向 = 2.5×

3. **输出 tensor shape 不含 S×S**：所有中间/输出张量的 shape 都是 O(S·D)，不会在 Chakra 轨迹中产生 S×S 大小的 tensor_size，从而正确反映 FlashAttention 的内存优势。

### 3.3 代码修改

修改了以下文件以支持 `--flash_attention` 参数：

**`main.py`**：新增 CLI 参数 `--flash_attention`（布尔值，默认 `false`），并将其传递给模型构造函数。

**`models/stage1/llama_model.py`**：
- `group_query_attention(flash_attention=False)`: 根据参数选择 `_kernel_fused.csv` 或 `_kernel_flash.csv`
- `transformer_decoder_block(flash_attention=False)`: 透传参数
- `llama(flash_attention=False)`: 透传参数，cache 文件名包含 `_fa{0|1}` 后缀

**`models/stage1/gpt_model.py`**：同上，支持 `tpsp` 和 `flash_attention` 双参数。

## 4. 使用方法

### 4.1 选择 Attention kernel

`llama3_8b.sh` 现在统一使用一个环境变量 `ATTENTION_BACKEND` 选择注意力实现，不再需要单独的 `FLASH_ATTENTION` 开关：

```bash
ATTENTION_BACKEND=standard ./llama3_8b.sh
ATTENTION_BACKEND=fused ./llama3_8b.sh
ATTENTION_BACKEND=flash ./llama3_8b.sh
```

如果直接调用 `main.py`，仍然推荐使用 `--attention_backend {standard,fused,flash}`；`--flash_attention true` 仅保留为兼容旧接口的写法。

### 4.2 对仿真结果的影响

启用 FlashAttention 后，生成的 Chakra 执行轨迹会有以下变化：

1. **COMP 节点 num_ops**：注意力相关的计算节点 FLOPs 将正确反映 O(S²) 复杂度。对于 Llama-3 8B (S=8192, d=128) 的配置，单个注意力层的前向 FLOPs 约为 4×B×H×8192²×128 ≈ 一个数量级高于 fused kernel 的线性估计。

2. **tensor_size**：注意力中间结果不包含 S×S 维度，ASTRA-sim 中对应的数据传输量更小。

3. **反向计算量增加**：由于 FlashAttention 的 recomputation 策略，反向传播的计算量比前向传播多 2.5×（而非标准注意力的 2×），但内存占用显著降低。

## 5. 局限性与后续工作

1. **MoE 模型**：当前 FlashAttention 仅支持 `model_type=llama`、`dense` 和 `gpt`，MoE 模型的注意力层需要在 `moe_model.py` 中额外适配。

2. **Activation Recomputation 交互**：FlashAttention 本身已包含反向传播中的 recomputation，如果同时启用 `--activation_recompute`，可能需要注意 recomputation 的重叠。

3. **Block-sparse / Ring Attention**：当前实现为标准 FlashAttention，未建模 block-sparse attention 或 ring attention 等变体。如需支持超长序列的 ring attention，需要在 kernel CSV 中额外建模跨设备的 KV 传递通信。
