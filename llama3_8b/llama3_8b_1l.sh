#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../../symbolic_tensor_graph

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

# if SGD = standard, then DP_LOCAL_SGD_INTERVAL should be 1, otherwise it should be greater than 1.
# else INTERVAL = NUM_ITERATIONS
NUM_ITERATIONS=${NUM_ITERATIONS:-1}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${NUM_ITERATIONS}}
fi

OUTPUT_DIR=${SCRIPT_DIR}/llama8b_1l_${ATTENTION}_${SGD}

# Run Symbolic Tensor Graph (STG) Generator for Llama-3 8B.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp 4 --tp 1 --pp 4 \
               --seq 1024 --batch 128 \
                --dvocal 128256 --dmodel 4096 --dff 14336 \
               --head 32 --kvhead 8 --num_stacks 32 \
               --micro_batch 8 \
                --num_iterations "${NUM_ITERATIONS}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type llama \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --weight_sharded 0
)
