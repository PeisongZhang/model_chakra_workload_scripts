# Transformer 基础结构、计算公式与每层计算量

这篇文档补一份 **Transformer 基础说明**，目标是回答三个问题：

1. Transformer 的整体结构是什么
2. 每一层在算什么，公式是什么
3. 每一层的大致参数量和 FLOPs 是多少

下文默认以 **decoder-only Transformer / LLM** 为主线，因为大语言模型训练里最常见；最后再补一句它和经典论文版 Transformer 的差别。

## 1. 先定义记号

为了把公式写清楚，先统一记号：

- $B$：batch size
- $S$：sequence length
- $d$：hidden size，也常写作 $ d_{model} $
- $h$：attention head数
- $d_h = d / h$：每个 head 的维度
- $d_{ff}$：FFN 中间层维度
- $V$：词表大小
- $L$：Transformer block 层数

本文默认 FLOPs 采用常见工程口径：**一次乘加算 2 FLOPs**。

## 2. Transformer 的整体结构

一个 decoder-only Transformer 可以写成：

$$
X^{(0)} = \text{TokenEmbedding}(T) + \text{PositionalEncoding}(T)
$$

然后堆叠 $L$ 个 block：

$$
X^{(\ell+1)} = \text{Block}^{(\ell)}(X^{(\ell)}), \quad \ell = 0,1,\dots,L-1
$$

最后输出 logits：

$$
\text{logits} = X^{(L)} W_{\text{lm}}^T
$$

其中单个 block 的经典 pre-norm 写法是：

$$
U = X + \text{MHA}(\text{LN}(X))
$$

$$
Y = U + \text{FFN}(\text{LN}(U))
$$

也就是：

1. $LayerNorm$
2. $Multi-Head Self-Attention$
3. 残差连接
4. $LayerNorm$
5. $Feed-Forward Network$
6. 残差连接

如果是语言模型，还会加 **causal mask**，保证位置 $i$ 只能看见 $<= i$ 的 token。

## 3. 每一层具体在算什么

## 3.1 Embedding 层

输入 token id  $t_i$ 先查词表：

$$
e_i = E[t_i], \quad E \in \mathbb{R}^{V \times d}
$$

如果使用绝对位置编码：

$$
x_i^{(0)} = e_i + p_i
$$

其中 $p_i$ 是第 $i$ 个位置的 position embedding。

如果使用 RoPE，通常不是在输入处相加位置向量，而是在 attention 里对 $ Q,K$ 做旋转编码。LLaMA、Qwen 这一类现代 LLM 多数更接近这种写法。

### Embedding 的参数量

- 词嵌入表：$V * d$
- 若使用绝对位置 embedding：再加 $S_max * d$

### Embedding 的计算量

严格说 embedding lookup 主要是**查表和访存**，不是大矩阵乘法。

- 前向计算量：通常不把它当作主要 FLOPs
- 但显存/带宽压力明显，尤其是大词表

## 3.2 LayerNorm / RMSNorm

经典 LayerNorm 对每个 token 的 hidden vector 做归一化：

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2
$$

$$
\text{LN}(x)_j = \gamma_j \frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_j
$$

现代 LLM 常见的是 RMSNorm：

$$
\text{RMSNorm}(x)_j = \gamma_j \frac{x_j}{\sqrt{\frac{1}{d}\sum_{k=1}^{d}x_k^2+\epsilon}}
$$

RMSNorm 没有减均值这一步，稍微更省。

### 参数量

- LayerNorm：$2d$（$\gamma$ 和 $\beta$）
- RMSNorm：$d$  ($\gamma$)

### 计算量

对每个 token 都是 $O(d)$，整个 batch 序列是：

$$
O(BSd)
$$

和 attention、FFN 相比，这部分通常不是主要 FLOPs 来源。

## 3.3 Multi-Head Self-Attention

这是 Transformer 的核心。

先把输入投影成 $Q,K,V$：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

其中：

- $ X \in R^{B \times S \times d} $
- $ W_Q,W_K,W_V \in R^{d \times d} $

然后 reshape 成多个头：

$$
Q,K,V \in \mathbb{R}^{B \times h \times S \times d_h}
$$

对每个 head 分别计算：

$$
A = \frac{QK^T}{\sqrt{d_h}}
$$

加上 mask 后做 softmax：

$$
P = \text{softmax}(A + M)
$$

再和 $V$ 相乘：

$$
O = PV
$$

多头拼接后再做一次输出投影：

$$
\text{MHA}(X) = \text{Concat}(O_1,\dots,O_h)W_O
$$

其中 $W_O \in R^{d \times d}$。

### 3.3.1 Attention 的参数量

标准 multi-head attention：

- $W_Q$: $d * d$
- $W_K$: $d * d$
- $W_V$: $d * d$
- $W_O$: $d * d$

总计：

$$
4d^2
$$

如果带 bias，再额外加 $4d$，但现代 LLM 往往去掉 bias。

### 3.3.2 Attention 的前向 FLOPs

先看四部分。

#### 1. Q/K/V 三个线性投影

每个投影把 $BS \times d$ 乘 $d \times d$：

$$
2BSd^2
$$

三个投影总计：

$$
6BSd^2
$$

#### 2. 计算注意力分数 $QK^T$

对每个 head：

$$
2S^2d_h
$$

对全部 batch 和全部 head：

$$
2BhS^2d_h = 2BS^2d
$$

#### 3. softmax

softmax 本身大约是：

$$
O(BhS^2)
$$

它有代价，但通常比矩阵乘法低一个量级，所以做系统级估算时常被当作次要项。

#### 4. 计算 $PV$

同理可得：

$$
2BS^2d
$$

#### 5. 输出投影 $W_O$

再做一次 $d x d$ 线性层：

$$
2BSd^2
$$

### 3.3.3 Attention 前向总 FLOPs

把主要项加起来：

$$
\text{FLOPs}_{\text{attn,fwd}} \approx 8BSd^2 + 4BS^2d
$$

这里很重要的一点是：

- $8BSd^2$：来自 QKV 和输出投影
- $4BS^2d$：来自 attention 核心 $QK^T$ 和 $PV$

所以 attention 并不只是 $O(S^2)$。更准确地说，它同时包含：

$$
O(BSd^2) + O(BS^2d)
$$

当 $S$ 很长时，$S^2$ 项占主导；当 $S$ 没那么长、$d$ 很大时，线性投影部分也很重。

### 3.3.4 Attention 的显存特征

标准实现里，最大的问题之一是会显式或隐式产生：

$$
B \times h \times S \times S
$$

大小的注意力矩阵。

所以它的主要中间激活不只包含 $Q,K,V,O$ 这些 $O(BSd)$ 张量，还包含：

$$
O(BhS^2)
$$

这也是长序列训练里 FlashAttention 很重要的原因。

## 3.4 FFN / MLP 层

经典 Transformer 的 FFN 写法是（下面统一按行向量写法）：

$$
\text{FFN}(x) = \phi(xW_1 + b_1) W_2 + b_2
$$

其中：

- $W_1 \in R^{d \times d_{ff}}$
- $W_2 \in R^{d_{ff} \times d}$
- $\phi$ 可以是 ReLU 或 GELU

现代 LLM 更常见的是 $SwiGLU / GeGLU$ 一类门控 MLP。以 SwiGLU 为例：

$$
u = xW_{up}, \qquad g = xW_{gate}
$$

$$
z = \text{SiLU}(g) \odot u
$$

$$
y = zW_{down}
$$

这里通常有三组矩阵：

- $W_{up} \in R^{d \times d_{ff}}$
- $W_{gate} \in R^{d \times d_{ff}}$
- $W_{down} \in R^{d_{ff} \times d}$

### 3.4.1 经典 FFN 的参数量

$$
d \cdot d_{ff} + d_{ff} \cdot d = 2dd_{ff}
$$

若带 bias，再额外加 $d_{ff} + d$。

### 3.4.2 经典 FFN 的前向 FLOPs

第一层线性：

$$
2BSdd_{ff}
$$

第二层线性：

$$
2BSd_{ff}d
$$

总计主项：

$$
\text{FLOPs}_{\text{ffn,fwd}} \approx 4BSdd_{ff}
$$

激活函数 GELU/ReLU 也是 $O(BSd_{ff})$，通常不是主项。

### 3.4.3 SwiGLU 的参数量与 FLOPs

参数量：

$$
dd_{ff} + dd_{ff} + d_{ff}d = 3dd_{ff}
$$

前向主 FLOPs：

$$
\text{FLOPs}_{\text{swiglu,fwd}} \approx 6BSdd_{ff}
$$

若固定同一个 $d_{ff}$，SwiGLU 比经典两层 FFN 多一组上投影，所以参数量和主 FLOPs 大约是后者的 $1.5 \times$；实际现代 LLM 往往会把门控 MLP 的 $d_{ff}$ 设得更小，因此整层成本未必更高。

## 3.5 残差连接

残差连接就是逐元素相加：

$$
y = x + f(x)
$$

参数量为 0，计算量大约：

$$
BSd
$$

通常可忽略不计，但在实现里它影响激活保存和通信边界。

## 4. 一个 Transformer Block 的总参数量和计算量

先给出两个最常用版本。

## 4.1 经典 Transformer block

由：

- 1 个 MHA
- 1 个两层 FFN
- 2 个 LayerNorm

组成。

### 参数量

$$
\text{Params}_{\text{block}} \approx 4d^2 + 2dd_{ff}
$$

再加两个 LayerNorm，大约：

$$
4d
$$

如果 $d_{ff} \gg d$，那主项通常是：

$$
4d^2 + 2dd_{ff}
$$

### 前向 FLOPs

$$
\text{FLOPs}_{\text{block,fwd}}
\approx (8BSd^2 + 4BS^2d) + 4BSdd_{ff}
$$

也就是：

$$
\text{FLOPs}_{\text{block,fwd}}
\approx 8BSd^2 + 4BS^2d + 4BSdd_{ff}
$$

## 4.2 现代 LLM block（RMSNorm + GQA/MHA + SwiGLU）

如果仍先按标准 MHA 近似，而 MLP 用 SwiGLU，则：

### 参数量

$$
\text{Params}_{\text{block}} \approx 4d^2 + 3dd_{ff}
$$

再加两层 RMSNorm：

$$
2d
$$

### 前向 FLOPs

$$
\text{FLOPs}_{\text{block,fwd}}
\approx 8BSd^2 + 4BS^2d + 6BSdd_{ff}
$$

若两者使用相同的 $d_{ff}$，这比经典 FFN 版本更重；但实际现代 LLM 常通过减小门控 MLP 的 $d_{ff}$ 来控制总成本。

## 5. 反向传播的计算量怎么估

训练里通常更关心的是 **forward + backward**。

粗略估算时，常用经验是：

- 线性层反向大约是前向的 $2x$
- 所以一个 matmul 的训练总 FLOPs 大约是前向的 $3x$

于是可以把一个 block 的训练 FLOPs 近似为：

$$
\text{FLOPs}_{\text{train}} \approx 3 \times \text{FLOPs}_{\text{fwd}}
$$

因此经典 block 的训练 FLOPs 可粗略写成：

$$
\text{FLOPs}_{\text{block,train}}
\approx 24BSd^2 + 12BS^2d + 12BSdd_{ff}
$$

如果 attention 使用 FlashAttention 这类重计算实现，attention 反向会比这个经验值再稍大一点，因为它会用额外计算换显存。

## 6. 什么时候是 Attention 更贵，什么时候是 FFN 更贵

看 block 前向主项：

$$
8BSd^2 + 4BS^2d + 4BSdd_{ff}
$$

可以得到几个直觉：

1. $QKV/ output projection$ 主要看 $d^2$
2. $attention score / AV$ 主要看 $S^2$
3. $FFN$ 主要看 $d \cdot d_{ff}$

如果用经典 Transformer 常见设定 $d_{ff} = 4d$，那么 FFN 主项是：

$$
16BSd^2
$$

这说明：

- **中短序列** 下，FFN 往往比 attention core 更重
- **长序列** 下，attention 的 $S^2$ 项会迅速超过 FFN

把 attention core 和投影项对比：

$$
4BS^2d \approx 8BSd^2 \Rightarrow S \approx 2d
$$

所以当 $S$ 接近或超过 $2d$ 时，attention 的二次项会非常显著。

## 7. 一个常见配置下的数量级例子

假设：

- $B = 1$
- $S = 4096$
- $d = 4096$
- $h = 32$
- $d_{ff} = 14336$（接近 LLaMA 一类配置）

则单层前向的主要项大约是：

### Attention 投影

$$
8BSd^2 = 8 \times 1 \times 4096 \times 4096^2 \approx 1.37 \times 10^{11}
$$

### Attention 核心

$$
4BS^2d = 4 \times 1 \times 4096^2 \times 4096 \approx 2.75 \times 10^{11}
$$

### SwiGLU MLP

$$
6BSdd_{ff}
= 6 \times 1 \times 4096 \times 4096 \times 14336
\approx 1.44 \times 10^{12}
$$

这个例子说明，在很多现代 LLM 配置里：

- MLP 非常重
- 长序列时 attention core 也很重
- 真正的总成本通常是两者一起大，而不是只看 $S^2$

## 8. 经典 Transformer 和现代 LLM 的几个差别

如果你后面要读 LLaMA、Qwen、Mistral 之类论文，常见差别主要有：

1. **Norm**
   - 经典 Transformer：LayerNorm
   - 现代 LLM：RMSNorm

2. **Position**
   - 经典 Transformer：绝对位置编码
   - 现代 LLM：RoPE 更常见

3. **FFN**
   - 经典 Transformer：ReLU/GELU 两层 FFN
   - 现代 LLM：SwiGLU/GeGLU 更常见

4. **Attention**
   - 经典论文常写标准 MHA
   - 现代模型常见 GQA / MQA，用更少的 KV heads 降低显存和带宽

5. **结构**
   - 原始 Transformer 同时有 encoder 和 decoder
   - 大语言模型训练通常更关注 decoder-only

## 9. 和训练优化文档的关系

如果把这篇文档和 [transformer_training_optimization_survey](transformer_training_optimization_survey.md) 对起来看，可以这样理解：

1. $FlashAttention$ 优化的是上面 $Attention$ 里的 $QK^T$ -> $softmax$ -> $PV$ 这部分实现
2. $ZeRO/FSDP$ 优化的是参数、梯度、优化器状态的存储和通信
3. $Checkpointing$ 优化的是 block 内各层 activation 的保存方式
4. $TP/PP/DP/SP$ 优化的是这些层如何分布到多卡执行

也就是说，训练优化技术大多不是在改变这些基础公式，而是在改变：

- 这些公式如何映射到 GPU kernel
- 中间结果是否保存
- 参数和激活如何切分
- 多卡之间如何通信

## 10. 一个最短总结

如果只保留最核心的结论，可以记这几句：

1. 一个 Transformer block 本质上就是：$Norm -> Attention -> Residual -> Norm -> FFN -> Residual$
2. Attention 的主计算由 $QKV projection + QK^T + softmax + PV + output projection$ 组成
3. Attention 的复杂度不是只有 $O(S^2)$，更准确是 $O(BSd^2) + O(BS^2d)$
4. FFN 的复杂度大约是 $O(BSdd_{ff})$，很多时候它和 attention 一样重，甚至更重
5. 长序列训练难，主要难在 attention 的 $S^2$ 激活和计算

如果你接下来要继续做 Chakra/Astra-sim 建模，下一步最值得单独展开的是：

- 标准 attention 在 trace 里应该拆成哪些算子
- FlashAttention 为什么不改变数学结果，但会改变内存和 kernel 组织
- GQA/MQA 为什么会显著影响 KV cache、通信量和显存
