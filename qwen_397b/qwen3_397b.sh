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

NUM_ITERATIONS=${NUM_ITERATIONS:-1}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${NUM_ITERATIONS}}
fi

OUTPUT_DIR=${SCRIPT_DIR}/qwen3_397b_${ATTENTION}_${SGD}

# Qwen3-397B MoE configuration
# - 60 layers, 512 experts (top-10), hidden_size=4096
# - Mixed attention: 3 linear + 1 full repeating pattern
# - Shared expert with intermediate_size=1024
# - Linear attention: 16 key heads, 64 value heads, head_dim=128, conv_kernel=4

(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
               --dp 4 --tp 8 --pp 10 --ep 64 \
               --seq 4096 --batch 1024 \
                --dvocal 248320 --dmodel 4096 --dff 1024 \
               --head 32 --kvhead 2 --num_stacks 60 \
               --experts 512 --kexperts 10 \
               --micro_batch 1 \
                --num_iterations "${NUM_ITERATIONS}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type qwen3_moe \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --weight_sharded true \
                --linear_num_key_heads 16 \
                --linear_num_value_heads 64 \
                --linear_head_dim 128 \
                --conv_kernel_dim 4 \
                --shared_expert_dff 1024 \
                --layer_types "linear,linear,linear,full"
)
