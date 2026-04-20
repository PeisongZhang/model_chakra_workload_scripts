# 复现实验配方

针对 Megatron‑LM 论文中的几个主要 takeaway，给出在当前仿真栈上可直接运行的最小实验。每个实验都是一对 "baseline vs 开启" 的对照，结论读自仿真日志或 STG 的 VRAM 打印。

> 所有命令假设工作在仓库根目录 `/home/ps/sow/part2/`。先激活 venv：
> ```bash
> source astra-sim/.venv/bin/activate
> ```

---

## 实验 1：interleaved 调度的 bubble 缩小（论文 §2.2.2 / 图 4 / 图 12）

**断言**：气泡占比 ≈ $(p-1)/(v \cdot m)$，即 $v$ 翻倍 → 气泡减半。

### 生成

```bash
cd dnn_workload/qwen_32b
# baseline: v=1, 默认映射
PP_VIRTUAL=1 PP_SCHEDULE=1f1b LAYER=32 \
    DP=1 TP=1 PP=4 BATCH=8 MICROBATCH=1 SEQUENCE=1024 \
    bash qwen_32b.sh
# 交错: v=2
PP_VIRTUAL=2 PP_SCHEDULE=1f1b-interleaved LAYER=32 \
    DP=1 TP=1 PP=4 BATCH=8 MICROBATCH=1 SEQUENCE=1024 \
    bash qwen_32b.sh
```

### 运行仿真

两次分别把 `WORKLOAD_DIR` 指向对应输出目录：

```bash
cd ../../astra-sim/qwen_experiment/in_dc
WORKLOAD_DIR=.../qwen_32b/standard_standard_32_1_8_1_1024_1f1b_v1_sgofalse_arfalse/ \
    bash analytical.sh
WORKLOAD_DIR=.../qwen_32b/standard_standard_32_1_8_1_1024_1f1b-interleaved_v2_sgofalse_arfalse/ \
    bash analytical.sh
```

### 读结果

`grep "Bubble time" log/analytical_*.log | head`——对比两次的 `Bubble time ... (X%)`，v=2 的应接近 v=1 的 1/2。

---

## 实验 2：scatter/gather 的跨节点字节下降（论文 §4.1 / 图 18）

**断言**：开启 scatter/gather 后，跨 pipeline stage 的 P2P 字节数下降到 `1/t`。

### 生成

```bash
cd dnn_workload/qwen_32b
# off
SGO=false LAYER=16 DP=1 TP=8 PP=2 BATCH=8 MICROBATCH=1 SEQUENCE=1024 \
    bash qwen_32b.sh
# on
SGO=true LAYER=16 DP=1 TP=8 PP=2 BATCH=8 MICROBATCH=1 SEQUENCE=1024 \
    bash qwen_32b.sh
```

### 直接从 ET 对账（无需仿真）

```bash
source astra-sim/.venv/bin/activate
python3 dnn_workload/symbolic_tensor_graph/test_cases/test_scatter_gather.py
```

该测试内部就做了相同对照，验证 `bytes_off / bytes_on == tp`（这里是 8）。

### 仿真对比（可选）

跑两次 `analytical.sh`，`grep "Effective BW\|Comm bytes" log/...`——on 的 p2p 字节应是 off 的 1/8，有效 BW 不变或略降。

---

## 实验 3：激活重算的 FLOP+显存权衡（论文 §3.5 / 图 17）

**断言**：backward 的 transformer 块 FLOP 精确多出一次 forward 量；VRAM acts 下降 ~80%。

### 生成并打印 VRAM

```bash
cd dnn_workload/symbolic_tensor_graph
# off
python3 main.py --output_dir /tmp/ar_off --output_name workload.%d.et \
    --dp 1 --tp 1 --pp 2 --sp 1 --dvocal 32000 --dmodel 512 --dff 1024 \
    --head 8 --kvhead 8 --num_stacks 8 --seq 512 --batch 4 --micro_batch 1 \
    --model_type llama --mixed_precision true \
    --activation_recompute false --print_gpu_vram true 2>&1 | grep "GPU"

# on
python3 main.py --output_dir /tmp/ar_on --output_name workload.%d.et \
    --dp 1 --tp 1 --pp 2 --sp 1 --dvocal 32000 --dmodel 512 --dff 1024 \
    --head 8 --kvhead 8 --num_stacks 8 --seq 512 --batch 4 --micro_batch 1 \
    --model_type llama --mixed_precision true \
    --activation_recompute true --print_gpu_vram true 2>&1 | grep "GPU"
```

对比 `acts=` 字段；on 的行末尾会带 `[recomp: acts X->Y @0.2]`。

### FLOP 对账（读 ET）

参考 `phase1_completion_zh.md` 的 "激活重算" 表格。关键事实：每 transformer 块 `B_new = B_old + F_old`，block 外（embedding/loss）不变。

---

## 实验 4：算子融合的 throughput 差距（论文 §5.8）

**断言**：设 `peak-perf-achievable-fraction=0.8` 后，单 GPU 平均 `compute_utilization` 与 wall time 大致按 1/0.8 膨胀。

### 改配置

编辑 `astra-sim/qwen_experiment/in_dc/astra_system.json`：
```json
"peak-perf-achievable-fraction": 0.8
```

### 跑仿真

与既有 workload 一起跑 `analytical.sh`。对比 `grep "Average compute utilization\|Wall time"` 两份日志（baseline 1.0 vs derate 0.8）。

---

## 实验 5：VRAM 超限告警（论文 §3.5）

**断言**：当未启用 FSDP 与激活重算的大模型被塞进单 rank 80 GB A100 时，`VRAM OVERFLOW` 应被触发。

### 操作

```bash
# astra_system.json 已默认 track-local-mem=1, vram-capacity-gb=80
cd astra-sim/qwen_experiment/in_dc
bash analytical.sh
grep "peak memory\|VRAM OVERFLOW\|VRAM OK" log/analytical_*.log | head -20
```

对默认 qwen_32b 128‑rank 配置，已实测 `peak=193 GB > cap=80 GB` → OVERFLOW。

### 启用重算缓解

```bash
# 重新生成一份打开激活重算的 workload；再跑一次
cd dnn_workload/qwen_32b
ACTIVATION_RECOMPUTE=true DP=4 TP=8 PP=4 LAYER=32 BATCH=128 MICROBATCH=2 \
    bash qwen_32b.sh
# 把 analytical.sh 的 WORKLOAD_DIR 指向 ..._artrue/ 重跑
```

此时 peak 应降到 ~cap 以下（重算 keep_ratio=0.2 的效果）。

---

## 实验 6：多 NIC 聚合带宽（论文 §5.9）

**断言**：`--nics-per-gpu 8` 使 GPU↔NIC 段聚合带宽 ×8，跨节点 P2P 有效 BW 相应提升。

### 生成两份拓扑

```bash
cd astra-sim/qwen_experiment/in_dc
python3 generate_topology.py \
    --gpus-per-nvlink-node 8 --nvlink-node-count 16 --nvlink-nodes-per-leaf 4 \
    --spine-count 1 \
    --gpu-nvswitch-bandwidth 4800Gbps --gpu-nvswitch-latency 0.00015ms \
    --gpu-nicswitch-bandwidth 200Gbps --gpu-nicswitch-latency 0.000001ms \
    --nicswitch-leaf-bandwidth 200Gbps --nicswitch-leaf-latency 0.0005ms \
    --leaf-spine-bandwidth 1600Gbps --leaf-spine-latency 0.0006ms \
    --nics-per-gpu 1 --output topology_n1.txt
python3 generate_topology.py \
    ... \
    --nics-per-gpu 8 --output topology_n8.txt
```

链路数应从 97 → 545（差值 = 2 × 32 × 7）。

### 跑仿真

修改 `network.yml` 指向 `topology_n1.txt` / `topology_n8.txt` 各跑一次，对比 `Effective BW p2p`。搭配 `SGO=true` 才能看出完整收益（论文的 892 GB/s 二分带宽）。

---

## 实验 7：论文核心 Takeaway #1（张量并行在节点内，流水跨节点）

**断言**：固定 GPU 数和模型，`t > 节点内 GPU 数` 时性能掉悬崖。

### 扫参

```bash
cd dnn_workload/qwen_32b
for CONFIG in "TP=1 PP=32" "TP=2 PP=16" "TP=4 PP=8" "TP=8 PP=4" "TP=16 PP=2"; do
    eval $CONFIG LAYER=32 DP=1 SP=1 BATCH=8 MICROBATCH=1 \
        bash qwen_32b.sh
done
```

各自跑仿真，画 wall time vs TP 曲线。期望：TP=8（=node GPUs）点最快；TP=16 跌落因为跨出 NVLink。

---

## 运行回归测试

```bash
cd dnn_workload/symbolic_tensor_graph
python3 test_cases/test_pipeline_interleaved.py
python3 test_cases/test_scatter_gather.py
```

两份都应输出 `ALL PASS`。

---

## 把所有开关打开的"full PTD‑P"配置

```bash
cd dnn_workload/qwen_32b
PP_SCHEDULE=1f1b-interleaved PP_VIRTUAL=2 \
    SGO=true ACTIVATION_RECOMPUTE=true \
    DP=4 TP=8 PP=4 LAYER=32 BATCH=128 MICROBATCH=2 SEQUENCE=4096 \
    bash qwen_32b.sh
```

对应输出目录：`standard_standard_32_1_128_2_4096_1f1b-interleaved_v2_sgotrue_artrue/`
