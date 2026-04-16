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

OUTPUT_DIR=${SCRIPT_DIR}/qwen32b_${ATTENTION}_${SGD}

# Run Symbolic Tensor Graph (STG) Generator for Qwen-32B.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp 4 --tp 4 --pp 4 \
               --seq 4096 --batch 128 \
                --dvocal 152064 --dmodel 5210 --dff 27648 \
               --head 40 --kvhead 8 --num_stacks 64 \
               --micro_batch 8 \
                --num_iterations "${NUM_ITERATIONS}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type llama \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --weight_sharded 0
)

# (
# cd ${STG}
# python3 main.py --output_dir "${OUTPUT_DIR}" \
#                 --output_name workload.%d.et \
#                --dp 4 --tp 4 --pp 4 \
#                --seq 4096 --batch 128 \
#                 --dvocal 152064 --dmodel 5210 --dff 27648 \
#                --head 40 --kvhead 8 --num_stacks 16 \
#                --micro_batch 8 \
#                 --num_iterations "${NUM_ITERATIONS}" \
#                 --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
#                 --model_type llama \
#                 --mixed_precision true \
#                 --attention_backend "${ATTENTION}" \
#                 --weight_sharded 0
# )
