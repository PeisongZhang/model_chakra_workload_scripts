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
TP=${TP:-1}
PP=${PP:-4}
SP=${SP:-1}

SGD=${SGD:-standard}
LAYER=${LAYER:-32}
SEQUENCE=${SEQUENCE:-8192}
BATCH=${BATCH:-128}
# Per-rank micro-batch size (Megatron convention).
MICROBATCH=${MICROBATCH:-2}

ITERATION=${ITERATION:-8}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${ITERATION}}
fi

OUTPUT_DIR=${SCRIPT_DIR}/${ATTENTION}_${SGD}_${LAYER}_${ITERATION}_${BATCH}_${MICROBATCH}_${SEQUENCE}

# Run Symbolic Tensor Graph (STG) Generator for Llama-3 8B.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp "${DP}" --tp "${TP}" --pp "${PP}" --sp "${SP}" \
               --seq "${SEQUENCE}" --batch "${BATCH}" \
                --dvocal 128256 --dmodel 4096 --dff 14336 \
               --head 32 --kvhead 8 --num_stacks "${LAYER}" \
               --micro_batch "${MICROBATCH}" \
                --num_iterations "${ITERATION}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type llama \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --weight_sharded 0
)
