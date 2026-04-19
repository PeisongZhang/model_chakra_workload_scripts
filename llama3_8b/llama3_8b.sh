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

SGD=${SGD:-standard}
LAYER=${LAYER:-32}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-8192}
BATCH_SIZE=${BATCH_SIZE:-128}

# Parallelism degrees. Total NPUs = DP * TP * PP * SP.
DP=${DP:-4}
TP=${TP:-1}
PP=${PP:-4}
SP=${SP:-1}

# Per-rank micro-batch size (Megatron convention).
# Number of micro-batches per iteration = BATCH_SIZE / (MICROBATCH * DP).
# Default 2 reproduces the previous behavior (BATCH_SIZE=128, DP=4 -> 16 mb*).
MICROBATCH=${MICROBATCH:-2}

# if SGD = standard, then DP_LOCAL_SGD_INTERVAL should be 1, otherwise it should be greater than 1.
# else INTERVAL = NUM_ITERATIONS
NUM_ITERATIONS=${NUM_ITERATIONS:-8}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${NUM_ITERATIONS}}
fi

OUTPUT_DIR=${SCRIPT_DIR}/${ATTENTION}_${SGD}_${LAYER}_${NUM_ITERATIONS}_${BATCH_SIZE}_${MICROBATCH}_${SEQUENCE_LENGTH}

# Run Symbolic Tensor Graph (STG) Generator for Llama-3 8B.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp "${DP}" --tp "${TP}" --pp "${PP}" --sp "${SP}" \
               --seq "${SEQUENCE_LENGTH}" --batch "${BATCH_SIZE}" \
                --dvocal 128256 --dmodel 4096 --dff 14336 \
               --head 32 --kvhead 8 --num_stacks "${LAYER}" \
               --micro_batch "${MICROBATCH}" \
                --num_iterations "${NUM_ITERATIONS}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type llama \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --weight_sharded 0
)
