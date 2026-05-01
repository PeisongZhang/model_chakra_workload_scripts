#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../symbolic_tensor_graph

ATTENTION=${ATTENTION:-standard}
case "${ATTENTION}" in
    standard|fused|flash)
        ;;
    *)
        echo "ATTENTION must be one of: standard, fused, flash" >&2
        exit 1
        ;;
esac

# Parallelism degrees. Total NPUs = DP * TP * PP * SP.
DP=${DP:-4}
TP=${TP:-8}
PP=${PP:-4}
SP=${SP:-1}
EP=${EP:-1}

SGD=${SGD:-standard}
LAYER=${LAYER:-4}
SEQUENCE=${SEQUENCE:-4096}
BATCH=${BATCH:-128}
# Per-rank micro-batch size (Megatron convention).
MICROBATCH=${MICROBATCH:-2}

# Pipeline-schedule controls (P0-A / P0-B).
#   PP_SCHEDULE: natural | gpipe | 1f1b | 1f1b-interleaved.
#     'natural' keeps the legacy dependency-driven execution.
#   PP_VIRTUAL : virtual stages per device (v). v=1 is contiguous mapping;
#     v>1 enables Megatron interleaved pipeline and requires
#     LAYER % (PP_VIRTUAL * PP) == 0.
PP_SCHEDULE=${PP_SCHEDULE:-1f1b}
PP_VIRTUAL=${PP_VIRTUAL:-1}
case "${PP_SCHEDULE}" in
    natural|gpipe|1f1b|1f1b-interleaved)
        ;;
    *)
        echo "PP_SCHEDULE must be one of: natural, gpipe, 1f1b, 1f1b-interleaved" >&2
        exit 1
        ;;
esac

# scatter/gather optimization (P1-A): 在跨 pipeline stage 时，每个 TP rank
# 只发送 1/t 片段（比完整张量少 (t-1)/t 字节），接收端用 intra-TP all-gather
# 补回完整张量。仅在 TP>1 时有效。
SGO=${SGO:-false}

# activation recomputation (P1-B): 开启后每个 transformer block 的 backward
# 会被抬升 1× forward 的 FLOP 量（论文 §3.5 的 "多跑一次 forward"），
# VRAM 报告里 acts 会按 activation_recompute_keep_ratio 粗略缩减。
ACTIVATION_RECOMPUTE=${ACTIVATION_RECOMPUTE:-false}

ITERATION=${ITERATION:-1}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${ITERATION}}
fi

OUTPUT_DIR=${SCRIPT_DIR}/att${ATTENTION}_sgd${SGD}_layer${LAYER}_iter${ITERATION}_batch${BATCH}_micro${MICROBATCH}_seq${SEQUENCE}_dp${DP}_tp${TP}_pp${PP}_sp${SP}_ep${EP}

# Run Symbolic Tensor Graph (STG) Generator for Qwen-32B.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp "${DP}" --tp "${TP}" --pp "${PP}" --sp "${SP}" \
               --seq "${SEQUENCE}" --batch "${BATCH}" \
                --dvocal 152064 --dmodel 5120 --dff 27648 \
               --head 40 --kvhead 8 --num_stacks "${LAYER}" \
               --micro_batch "${MICROBATCH}" \
                --num_iterations "${ITERATION}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type llama \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --pipeline_virtual_stages "${PP_VIRTUAL}" \
                --pipeline_schedule "${PP_SCHEDULE}" \
                --scatter_gather_optimization "${SGO}" \
                --activation_recompute "${ACTIVATION_RECOMPUTE}" \
                --weight_sharded 0
)
