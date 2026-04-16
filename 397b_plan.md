# Qwen3 397B MoE — STG Workload Generation Plan

## 1. 目标

在现有 Symbolic Tensor Graph (STG) 框架上扩展对 Qwen3-397B MoE 模型训练 workload 的支持。

---

## 2. Qwen3 397B 架构概要

| 参数 | 值 |
|------|-----|
| 架构 | `Qwen3_5MoeForConditionalGeneration` (MoE + 多模态) |
| hidden_size | 4096 |
| num_hidden_layers | 60 |
| num_attention_heads | 32 |
| num_key_value_heads | 2 |
| head_dim | 256 (显式设置，**非** Dmodel/Head) |
| num_experts | 512 |
| num_experts_per_tok | 10 |
| moe_intermediate_size | 1024 (每个 routed expert 的 FFN dim) |
| shared_expert_intermediate_size | 1024 |
| 注意力类型 | 混合: 每 4 层中 3 层 `linear_attention` + 1 层 `full_attention` |
| linear_key_head_dim | 128 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 64 |
| linear_value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |
| vocab_size | 248320 |
| max_position_embeddings | 262144 |

---

## 3. 差距分析 (Gap Analysis)

| # | 缺失特性 | 影响 | 优先级 |
|---|---------|------|--------|
| 1 | **Linear Attention 模块** | 75% 的层使用 linear attention，未建模则 FLOP 和通信量全部错误 | P0 |
| 2 | **异构层 (Heterogeneous Layers)** | 无法交替使用 full / linear attention 的 decoder block | P0 |
| 3 | **Shared Expert** | 每层额外的共享专家 FFN 未计入计算和通信 | P0 |
| 4 | **大规模专家图构建优化** | 512 专家时 per-expert 循环产生数万节点，构建极慢 | P1 |
| 5 | 多模态 Vision Encoder | 视觉编码器（27 层 ViT）未建模 | P2 |
| 6 | MTP (Multi-Token Prediction) | 额外预测头未建模 | P2 |
| 7 | head_dim ≠ Dmodel/Head | 现有 CSV 假设 head_dim = Dmodel/Head，Qwen3 中 head_dim = 256 而 Dmodel/Head = 128 | P1 |

---

## 4. 实施计划

### 4.1 Change 1 — Linear Attention 模块 (P0)

**目标**: 创建 linear attention 的前向/反向计算图，准确建模其 O(S·d²) 复杂度。

**Qwen3 Linear Attention 计算流程**:
```
x → Wqk projection → qk → split → q, k
                                     ↓
                              Conv1d (causal, kernel=4)
                                     ↓
                              q_conv, k_conv ──→ Linear Attention Kernel → attn
x → Wv projection → v ──────────────────────────────────────────────────↗
attn → Wo projection → o
```

**新增文件**:
- `sharding_spreadsheets/module3/tpsp_moe/linear_attention_surrounding.csv`
  - 处理 QK 融合投影 (wqk)、V 投影 (wv)、Conv1d (CUSTOM op)、输出投影 (wo)
  - 使用中间 `x_ag` 节点避免重复 all-gather
  - 反向传播中 `dx = dx_qk + dx_v` 通过 Add 节点合并
- `sharding_spreadsheets/module3/tpsp_moe/linear_attention_kernel.csv`
  - CUSTOM op，FLOP = `4 * Batch/dp * LKHead/tp * Seq/cp * LHeadDim * LHeadDim`
  - 关键区别：无 S² 项（vs full attention 的 `Seq/cp * Seq`）

**新增符号**:
| 符号 | CLI 参数 | 默认值 | 含义 |
|------|---------|--------|------|
| `LKHead` | `--linear_num_key_heads` | 0 | Linear attention 的 key/query head 数 |
| `LVHead` | `--linear_num_value_heads` | 0 | Linear attention 的 value head 数 |
| `LHeadDim` | `--linear_head_dim` | 128 | Linear attention 的 head 维度 |
| `ConvKernel` | `--conv_kernel_dim` | 4 | 1D 因果卷积核大小 |

**Python 变更**:
- `models/stage1/llama_model.py`: 新增 `linear_group_query_attention()` 函数
  - 加载 `linear_attention_surrounding.csv` + `linear_attention_kernel.csv`
  - 通过 `ConnectGraph` 连接 surrounding 和 kernel

### 4.2 Change 2 — 异构层支持 (P0)

**目标**: 支持每层使用不同的 attention block（full attention 或 linear attention）。

**设计**:
- 新增 CLI 参数 `--layer_types`: 逗号分隔的层类型字符串
  - 例如 `"linear,linear,linear,full"` 表示重复模式
  - 如果长度 < num_stacks，则循环重复 (tile)
- 修改 `moe_model.py` 中的 `transformer()`:
  - 为 `full_attention` 层和 `linear_attention` 层创建不同的 decoder block 模板
  - `transformer_decoders_heterogeneous()` 根据 `layer_types` 列表选择模板

**实现方式**:
```python
def transformer_decoders_heterogeneous(num_layers, layer_types, 
                                        full_attn_template, linear_attn_template):
    decoders = []
    for i in range(num_layers):
        template = full_attn_template if layer_types[i] == "full" else linear_attn_template
        decoder = ReplicateGraph.apply(template, f"transformer.{i}.%s")
        decoders.append(decoder)
        # ... connect sequential layers ...
```

### 4.3 Change 3 — Shared Expert (P0)

**目标**: 在 MoE FFN 中加入共享专家分支。

**Qwen3 Shared Expert 结构**:
```
post_attn_norm.y ──→ MoE (512 routed experts, top-10) ──→ moe_output ──┐
                 └──→ Shared Expert (dense FFN, Dff=1024) ──→ shared_out ─→ Add → ffn_output
```

**设计**:
- 新增 CLI 参数 `--shared_expert_dff` (默认 0，表示不使用)
- 新增符号 `SharedDff`
- 复用现有 `tpsp_moe/llama_feed_forward_network.csv`，通过 `ReplicateGraph` 将 `Dff` 映射为 `SharedDff`
- Shared expert 使用标准 TP 并行（不使用 EP，因为处理所有 token）
- 在 `transformer_decoder_block()` 中:
  1. 加载 shared FFN 并命名为 `shared_ffn.%s`
  2. 将 `post_attn_norm.y` 同时连接到 MoE 和 shared FFN
  3. 通过 Add 节点合并输出: `ffn_combined = moe.y + shared_ffn.xdown`
  4. `ffn_combined` 连接到 `ffn_res.x1`

### 4.4 Change 4 — 大规模专家图构建优化 (P1)

**问题**: 当前 `feed_forward_network()` 为每个本地专家创建独立的图分支。  
- EP=64, Experts=512 → 8 branches/group (OK)  
- EP=8 → 64 branches (慢)  
- EP=1 → 512 branches (不可行)

**优化策略**: 阈值切换
- `experts_per_group ≤ 32`: 保持现有 per-expert 循环（语义精确）
- `experts_per_group > 32`: 使用 **batched expert** 模式

**Batched Expert 模式**:
- 创建单一 FFN 分支，token 数量 = 所有本地专家的 token 总和
- 符号替换: `Seq → Seq * KExperts / ep`（不除以 Experts，等效于 Experts/ep 个专家的 token 合并）
- 权重内存通过额外的 placeholder 节点补偿（Experts/ep - 1 份额外权重）
- 跳过 per-expert 的 expert_wrapper 和 reduce_chain
- 图节点从 ~20N 减少到 ~20

**局限性**: 
- Batched 模式是近似：丢失了 per-expert 的 token routing 细节
- 对于大 EP 值（典型的 397B 训练场景），per-expert 模式仍然可用

### 4.5 Change 5 — 新增 `qwen3_moe` 模型类型

**新增 CLI 参数汇总**:

```bash
python3 main.py \
    --model_type qwen3_moe \
    --linear_num_key_heads 16 \
    --linear_num_value_heads 64 \
    --linear_head_dim 128 \
    --conv_kernel_dim 4 \
    --shared_expert_dff 1024 \
    --layer_types "linear,linear,linear,full" \
    # ... 其他现有参数 ...
```

---

## 5. 文件变更清单

| 操作 | 文件 |
|------|------|
| **新增** | `sharding_spreadsheets/module3/tpsp_moe/linear_attention_surrounding.csv` |
| **新增** | `sharding_spreadsheets/module3/tpsp_moe/linear_attention_kernel.csv` |
| **修改** | `models/stage1/llama_model.py` — 新增 `linear_group_query_attention()` |
| **修改** | `models/stage1/moe_model.py` — shared expert, 异构层, batched experts |
| **修改** | `main.py` — 新增 CLI 参数, `qwen3_moe` 模型类型 |
| **新增** | `chakra-demo/demo3/qwen3_397b.sh` — 生成脚本 |

---

## 6. 推荐并行策略 (Qwen3 397B)

| 参数 | 建议值 | 理由 |
|------|--------|------|
| `--tp` | 8 | NVLink 节点内张量并行 |
| `--pp` | 10 或 12 | 60 层需足够流水线级数分摊显存 |
| `--ep` | 64 | 512 专家 / 64 = 8 专家/组，每组构建高效 |
| `--dp` | 4–16 | 取决于总 GPU 数 |
| `--weight_sharded` | true | 397B 参数必须 ZeRO-3 分片 |
| `--seq` | 4096 | 预训练典型序列长度 |
| `--batch` | 1024 | 全局 batch size |
| `--micro_batch` | 1 | MoE 显存紧张 |
| `--mixed_precision` | true | BF16 |
| `--attention_backend` | flash | 仅对 full_attention 层生效 |

总 GPU 数 = dp × tp × pp = 4 × 8 × 10 = 320 (EP 与 DP 维度重叠)

---

## 7. 验证计划

1. **单元测试**: 加载新 CSV 文件，验证 shape 解析无错误
2. **Small-scale**: 用少量层 (num_stacks=4) + 少量专家 (experts=8) 跑通完整流程
3. **Full-scale**: 用完整 397B 配置生成 workload，检查:
   - 生成文件数 = dp × tp × pp
   - FLOP 量级合理（linear attention 层 << full attention 层）
   - 通信量包含 EP 维度的 AllToAll
4. **对比**: 与纯 MoE 模型（无 linear attention / shared expert）对比，确认差异合理
