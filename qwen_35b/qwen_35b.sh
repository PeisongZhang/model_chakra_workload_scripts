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

# Parallelism degrees. Total NPUs = DP * TP * PP * SP * EP.
# EP is the expert-parallel degree (num_experts_per_group = experts / ep).
DP=${DP:-4}
TP=${TP:-1}
PP=${PP:-4}
SP=${SP:-1}
EP=${EP:-8}

SGD=${SGD:-standard}
LAYER=${LAYER:-40}
SEQUENCE=${SEQUENCE:-4096}
BATCH=${BATCH:-128}
MICROBATCH=${MICROBATCH:-2}

# MoE: Qwen3.5-MoE has 256 experts, top-8 routing.
EXPERTS=${EXPERTS:-256}
KEXPERTS=${KEXPERTS:-8}

# Pipeline-schedule controls.
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

SGO=${SGO:-false}
ACTIVATION_RECOMPUTE=${ACTIVATION_RECOMPUTE:-false}

ITERATION=${ITERATION:-1}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${ITERATION}}
fi

OUTPUT_DIR=${SCRIPT_DIR}/att${ATTENTION}_sgd${SGD}_layer${LAYER}_iter${ITERATION}_batch${BATCH}_micro${MICROBATCH}_seq${SEQUENCE}_dp${DP}_tp${TP}_pp${PP}_sp${SP}_ep${EP}

# Qwen3.5-MoE config (from qwen_35b_config.json):
#   hidden_size (Dmodel) = 2048
#   moe_intermediate_size (per-expert Dff) = 512
#   num_attention_heads = 16, num_key_value_heads = 2
#   num_hidden_layers = 40
#   num_experts = 256, num_experts_per_tok = 8
#   vocab_size = 248320
# Note: the real head_dim is 256 (so Q/K/V project to Head*head_dim = 4096),
# but STG's GQA kernel implicitly uses head_dim = Dmodel / Head = 128.
# We keep Dmodel = hidden_size so downstream tensor shapes stay consistent
# with the residual stream; the attention FLOP count is off by the
# head_dim mismatch but comm patterns are faithful.
(
cd ${STG}
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
                --dp "${DP}" --tp "${TP}" --pp "${PP}" --sp "${SP}" --ep "${EP}" \
                --seq "${SEQUENCE}" --batch "${BATCH}" \
                --dvocal 248320 --dmodel 2048 --dff 512 \
                --head 16 --kvhead 2 --num_stacks "${LAYER}" \
                --experts "${EXPERTS}" --kexperts "${KEXPERTS}" \
                --micro_batch "${MICROBATCH}" \
                --num_iterations "${ITERATION}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type moe \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --pipeline_virtual_stages "${PP_VIRTUAL}" \
                --pipeline_schedule "${PP_SCHEDULE}" \
                --scatter_gather_optimization "${SGO}" \
                --activation_recompute "${ACTIVATION_RECOMPUTE}" \
                --weight_sharded 0
)
