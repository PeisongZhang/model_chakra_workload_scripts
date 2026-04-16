#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../../symbolic_tensor_graph

ATTENTION=${ATTENTION:-flash}
case "${ATTENTION}" in
    standard|fused|flash)
        ;;
    *)
        echo "ATTENTION must be one of: standard, fused, flash" >&2
        exit 1
        ;;
esac

SGD=${SGD:-standard}

# if SGD = standard, then DP_LOCAL_SGD_INTERVAL should be 1, otherwise it should be greater than 1.
# else INTERVAL = NUM_ITERATIONS
NUM_ITERATIONS=${NUM_ITERATIONS:-1}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${NUM_ITERATIONS}}
fi

# Basic Llama-3.1 405B training setup.
# Total NPUs = DP * TP * PP * SP
DP=${DP:-8}
TP=${TP:-8}
PP=${PP:-14}
SP=${SP:-1}

SEQ=${SEQ:-8192}
BATCH=${BATCH:-224}
MICRO_BATCH=${MICRO_BATCH:-8}

OUTPUT_DIR=${SCRIPT_DIR}/llama405b_${ATTENTION}_${SGD}

# Run Symbolic Tensor Graph (STG) Generator for Llama-3.1 405B.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp "${DP}" --tp "${TP}" --pp "${PP}" --sp "${SP}" \
               --seq "${SEQ}" --batch "${BATCH}" \
                --dvocal 128256 --dmodel 16384 --dff 53248 \
               --head 128 --kvhead 16 --num_stacks 126 \
               --micro_batch "${MICRO_BATCH}" \
                --num_iterations "${NUM_ITERATIONS}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type llama \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --weight_sharded 0
)
