# STG 里 MoE 的通信：EP 的 All-to-All 有没有"替换掉" TP 的 All-Reduce？

**结论先行**：**没有替换，是叠加。** 在 `ep > 1` 的 MoE 里，STG 会**新增**两次 EP All-to-All（dispatch / combine），但每个 expert 内部的 TP 通信（包括 attention 侧的 all-reduce / reduce-scatter）**原样保留**。换句话说：

| 层 | Dense + TP/SP | MoE + TP/SP + EP |
|---|---|---|
| Attention QKV / out proj | TP reduce-scatter + all-gather | **不变** |
| FFN 入口 | — | **新增 EP All-to-All (dispatch)** |
| Expert 内部 FFN | TP reduce-scatter（row-parallel `wdown`） | **同样保留** |
| FFN 出口 | — | **新增 EP All-to-All (combine)** |
| DP all-reduce | 每 iter 1 次 | 每 iter 1 次（`dp_local_sgd_interval=1`） |

当 `TP=1` 时 TP 那一列的通信自然退化为 no-op，表面上看像是"EP 接管了" —— 其实只是 TP 无事可做。

下面从 sharding CSV 层面展开细节，方便以后定位。

## 1. EP All-to-All 是怎么出现的

关键张量在 `sharding_spreadsheets/module3/tpsp_moe/moe_frame.csv`：

| 张量 | shape | 说明 |
|---|---|---|
| `x1`（pre-dispatch） | `Batch/dp, (Seq/cp)/tp, Dmodel/(1*ep)` | `ep` 切在 **Dmodel** |
| `xrouted = I(x1)` | `Batch/dp, (Seq/cp)*KExperts/(tp*ep), Dmodel/1` | `ep` 切换到 **Seq·KExperts** |
| `yrouted` | `Batch/dp, (Seq/cp)*KExperts/(tp*ep), Dmodel` | combine 前，仍切 Seq·KExperts |
| `y1 = I(yrouted)` | `Batch/dp, (Seq/cp)*KExperts/tp, Dmodel/(1*ep)` | 又切回 **Dmodel** |

`I`（Identical）节点本身是 0-FLOP 的，但它两端**分片维度不一致**。`GraphDistributer` 在 cross-bucket 阶段看到生产者/消费者 partition 对不上时会自动补一次 collective —— `Dmodel`-分片 ↔ `Seq`-分片 这对切换，落地就是经典 MoE 的：

- `x1 → xrouted`：**All-to-All dispatch**（各 rank 把 token 发到对应 expert 所在的 rank）
- `yrouted → y1`：**All-to-All combine**（expert 算完后把结果发回 token 原位）

冒烟跑 `qwen35b_smoke` 时日志里能看到每对 `ep` rank 间 `copying from ((pp, *), (ep, A), ...) to ((pp, *), (ep, B), ...)` 的记录，以及 `unefficient collective sliced found!` 告警 —— 后者是 STG 把 all-to-all 实现成 slice/concat 组合（语义等价，只是不高效，可以后续优化）。

反向传播对称地再走一次这俩 All-to-All：`dy / dxrouted` 等张量的 partition 切换会生成 `dx` 方向的 dispatch/combine。

## 2. TP 的通信没被替换

`tpsp_moe/llama_feed_forward_network.csv`（**每个 expert 自己的 FFN**，以 `ffn.%s` 前缀嵌进每个 expert 分支）：

```
wgate   : Dmodel,  Dff/tp
wdown   : Dff/tp,  Dmodel

xgate   = x00 @ wgate          → Batch/dp, Seq/cp, Dff/tp          # column-parallel
xdown1  = xgate @ wdown        → Batch/dp, Seq/cp, Dmodel          # row-parallel, hidden 按 tp 累加
xdown   = I(xdown1)            → Batch/dp, (Seq/cp)/tp, Dff        # 切回 SP
```

这条链和 dense Llama 的 FFN 结构**完全一致**：

- `xdown1` 是 row-parallel matmul 的输出，它的 `hidden` 维带着 `tp`（表示"各 tp rank 持部分和"）。
- 消费者 `xdown` 的 `Seq` 又被切回 `tp` 分片。
- 这对 `hidden=1/tp → hidden=1, Seq/cp → (Seq/cp)/tp` 的变化，对应 TP + SP 下经典的 **reduce-scatter**（TP-only 时就是 all-reduce）。

反向 `dx000 → dx0` 是 `hidden=1/tp → hidden=1, (Seq/cp)/tp → Seq/cp` 的反向切换，对应 TP **all-gather**。

### 注意力侧同理

`moe_model.transformer_decoder_block()` 里：

```python
from .llama_model import group_query_attention, transformer_decoders
```

所以 GQA 直接复用 `sharding_spreadsheets/module3/tpsp/` 那套 surrounding + kernel，QKV projection 出口的 TP all-reduce、output projection 入口的 TP reduce-scatter 完全照 dense 的来。Qwen3.5-MoE 的 attention 部分没有用到 EP。

## 3. 为什么 Qwen3.5-MoE 只有 gate+down 两个投影？

`tpsp_moe/llama_feed_forward_network.csv` 比 `tpsp/llama_feed_forward_network.csv` 少了 `wup` 和 `xupgate` 分支：Qwen3.5 的 MoE expert 用的是**单 gate + down**的 GLU 变体（等价于 `down(act(gate(x)))`，没有独立的 up projection）。这是模型本身的设计，不是通信结构差异 —— 但它确实让 per-expert FFN 的 TP 通信量相对 dense SwiGLU 少了一次 column-parallel 矩阵乘的开销。

## 4. 快速验证路径

想直接看 MoE 到底插了哪些通信：

```bash
# 生成最小 MoE workload
cd /home/ps/sow/part2/dnn_workload
DP=2 TP=2 PP=1 SP=1 EP=2 EXPERTS=4 KEXPERTS=2 \
LAYER=2 BATCH=4 MICROBATCH=1 SEQUENCE=512 \
bash qwen_35b/qwen_35b.sh

# ns-3 跑一遍，打开 ENABLE_TRACE，再用 traffic_analysis 看矩阵
# extract_traffic_matrix.py 会把 mix.tr 按 (src, dst, size) 聚合，
# EP all-to-all 会表现为 ep-group 内部 NxN 的"全对全"流量块。
```

TP 的 all-reduce/reduce-scatter 会落在同一 TP rank 子组的相邻流量里；EP 的 all-to-all 则是横跨 `ep` 维度、Dmodel ↔ Seq 重分片的块状流量。两类通信在 traffic matrix 上形状不同，应该能目视区分。
