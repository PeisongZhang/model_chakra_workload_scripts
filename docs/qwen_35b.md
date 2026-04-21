# Qwen3.5-MoE-35B Workload 接入记录

本篇记录把 **Qwen3.5-MoE (35B)** 接入 STG → Chakra → ASTRA-sim 这条流水的改动、踩到的坑、以及冒烟验证结果。配套脚本 `qwen_35b/qwen_35b.sh`、修复 `symbolic_tensor_graph/main.py`。

## 1. 模型结构来源

配置取自 `qwen_35b/qwen_35b_config.json`（HuggingFace `Qwen3_5MoeForConditionalGeneration`）：

| 项 | 值 | 对应 STG 参数 |
|---|---|---|
| `num_hidden_layers` | 40 | `--num_stacks 40` |
| `hidden_size` | 2048 | `--dmodel 2048` |
| `num_attention_heads` | 16 | `--head 16` |
| `num_key_value_heads` | 2 | `--kvhead 2` |
| `moe_intermediate_size` | 512 | `--dff 512` |
| `num_experts` | 256 | `--experts 256` |
| `num_experts_per_tok` | 8 | `--kexperts 8` |
| `vocab_size` | 248320 | `--dvocal 248320` |
| `head_dim` | 256 | ⚠️ 见下 |

> **head_dim 差异**：真实 head_dim 是 256（Q/K/V 投到 `head*head_dim = 4096`），但 STG 的 GQA kernel 隐式按 `head_dim = Dmodel/Head = 128` 建模。保留 `Dmodel = hidden_size` 让残差流形状正确，代价是注意力 FLOP 被 ×½。通信量、memory 占用、pipeline 拓扑都忠实，拿来跑网络/系统层实验足够。
>
> Qwen3.5 的 layer mix（每 4 层一次 full_attention、其余是 linear/mamba-style attention）、shared expert、MTP 层，STG 当前一律当成标准 transformer block 处理。

## 2. 驱动脚本 `qwen_35b/qwen_35b.sh`

对标 `qwen_32b.sh`，差异：

- `--model_type moe`（32B 是 dense Llama）
- 新增 `EP` 环境变量 → `--ep`（专家并行度）
- 新增 `EXPERTS` / `KEXPERTS`（默认 256 / 8）
- Dmodel / Dff / Head / KVHead / Dvocal 按上表固化

**默认参数**：`DP=4 TP=1 PP=4 SP=1 EP=8`（128 NPUs）、`LAYER=40 BATCH=128 MICROBATCH=2 SEQ=4096 ATTENTION=standard`。

输出目录命名同其他驱动：
```
{ATTENTION}_{SGD}_{LAYER}_{ITER}_{BATCH}_{MB}_{SEQ}_{PP_SCHED}_v{PP_VIRT}_sgo{SGO}_ar{AR}_ep{EP}_e{EXPERTS}k{KEXPERTS}
```

## 3. 主干 bug：MoE 路径的 `Batch → MicroBatch` 双重替换

### 症状

第一次跑 MoE smoke 时 STG 在 `BundledConvertChakra` 阶段崩溃：

```
TypeError: Cannot convert expression to float
expr=4*Dmodel*Seq**2*Micro(MicroBatch)/(cp*dp*tp) + ...
```

`Micro(MicroBatch)` 是 sympy 把 `Micro` 当成 Function 应用到 `MicroBatch` 的结果 —— 这显然不是正常 shape 表达式能长出来的。

### 根因

`ReplicateGraph._update_symbols` 对 `Slice / BroadcastReduce / Customized` 三种 op 的 `op_attr`（字符串）做 **字面替换**：

```python
tensor.op_attr = tensor.op_attr.replace(str(from_), f"({str(to_)})")
```

`MicroBatchReplicator.apply` 内部已经做过一次 `Batch → (MicroBatch)`，把原来含 `"Batch"` 的 op_attr 改写成含 `"MicroBatch"` 的版本。

`main.py` 的 MoE 分支紧接着又无条件地跑：

```python
transformer_moe = ReplicateGraph.apply(
    transformer_moe, inplace=True,
    old_symbol_map_new_symbol={"Batch": "MicroBatch"},
)
```

第二次替换在 `"(MicroBatch)"` 里又命中子串 `"Batch"`，得到 `"(Micro(MicroBatch))"`。`sympy.parse_expr` 把 `Micro(...)` 解析成函数应用，于是 `num_ops` 再也 `float()` 不了。

Dense / GPT 分支把第二次 ReplicateGraph 放在 `STAGE_MICROBATCH_OPTIMIZE=1` 的 `else` 分支里（走 shortcut 时 MicroBatchReplicator 不跑，只能由这次替换兜底）。MoE 分支漏了这个分支保护。

### 修复

把 MoE 的第二次 ReplicateGraph 移进 `else` 分支，对齐 dense/GPT：

```python
if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
    transformer_moe = MicroBatchReplicator.apply(transformer_moe, symbol_map_value)
else:
    print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs ...")
    transformer_moe = ReplicateGraph.apply(
        transformer_moe, inplace=True,
        old_symbol_map_new_symbol={"Batch": "MicroBatch"},
    )
```

改动在 `symbolic_tensor_graph/main.py`（MoE 分支）。

### 附带结论

`ReplicateGraph` 的字符串 `op_attr` 替换本身是脆弱的（前缀/后缀冲突随时会咬人）。更稳妥的做法是把 `op_attr` 也升级成 sympy 表达式，复用和 shape 同一套 `.replace(symbol, symbol)` 逻辑。暂时不动，先把 MoE 路径跑通。

## 4. 冒烟验证

最小参数（确认 STG + ASTRA-sim 端到端能过）：

```
DP=2 TP=1 PP=2 SP=1 EP=2 EXPERTS=8 KEXPERTS=2 \
LAYER=4 BATCH=4 MICROBATCH=1 SEQUENCE=512 ATTENTION=standard \
bash qwen_35b/qwen_35b.sh
```

总 NPU = 2·1·2·1·2 = 8。

**STG 输出**：`standard_standard_4_1_4_1_512_natural_v1_sgofalse_arfalse_ep2_e8k2/` 下 8 个 `workload.N.et`（~130 KB/rank）+ `workload.json`（24 个 comm group）。

**ASTRA-sim 回放**（analytical congestion-aware，Ring_8npus）：
- 配置：`astra-sim/qwen_experiment/qwen35b_smoke/{astra_system.json, network.yml, no_memory_expansion.json}`
- 结果（sys[0–7] 完全对称）：
  - Wall cycles: **42,821,020**
  - GPU time: 13,257,594
  - Comm time: 37,438,709（exposed 29,563,426）
  - Bubble: 0 %
  - Comm bytes: 569 MB/rank（p2p 4 MB + coll 565 MB），effective BW 13.3 GB/s
  - Average compute util: **82.57%**，op intensity 641

没有报错、没有 deadlock、所有 rank 对齐完成。

## 5. 下一步（未做）

- 按默认参数（128 NPUs）跑一遍完整 workload，接到 `llama_experiment/report.md` 那种汇总。
- 若要验证 MoE 的 All-to-All 真的被调度出来，结合 `traffic_analysis/extract_traffic_matrix.py` 过一遍 ns-3 mix.tr。
- 真 head_dim=256 要正确建模，需要让 STG 的 GQA 支持 `head_dim != Dmodel/Head`（改 sharding CSV 里 `Dmodel/Head` 的隐式假设）。
