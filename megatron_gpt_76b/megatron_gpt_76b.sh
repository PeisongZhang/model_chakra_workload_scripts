#!/bin/bash
set -e

# Megatron-LM GPT 76.1B (paper arXiv:2104.04473, Table 1 row 6).
# a=80, h=10240, l=60, t=8, p=4, GPUs=1024, Batch=1792, seq=2048, V=51200.
# Reported: 140 TFLOP/s/GPU (45% of A100 FP16 peak).

SCRIPT_DIR=$(dirname "$(realpath "$0")")
STG=${SCRIPT_DIR}/../symbolic_tensor_graph
PY=${PYTHON:-/home/ps/sow/part2/astra-sim/.venv/bin/python}

ATTENTION=${ATTENTION:-fused}
case "${ATTENTION}" in
    standard|fused|flash) ;;
    *) echo "ATTENTION must be one of: standard, fused, flash" >&2; exit 1 ;;
esac

# Parallelism (paper §5.1 Table 1): t=8, p=4, d=n/(t*p)=32 → n=1024.
DP=${DP:-32}
TP=${TP:-8}
PP=${PP:-4}
SP=${SP:-1}

SGD=${SGD:-standard}
LAYER=${LAYER:-60}
SEQUENCE=${SEQUENCE:-2048}
BATCH=${BATCH:-1792}
MICROBATCH=${MICROBATCH:-2}

ITERATION=${ITERATION:-1}
if [ "${SGD}" = "standard" ]; then
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
else
    DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-${ITERATION}}
fi

PP_SCHEDULE=${PP_SCHEDULE:-1f1b}
PP_VIRTUAL=${PP_VIRTUAL:-1}
SGO=${SGO:-1}
ACTIVATION_RECOMPUTE=${ACTIVATION_RECOMPUTE:-1}

OUTPUT_DIR=${SCRIPT_DIR}/${ATTENTION}_${SGD}_${LAYER}_${ITERATION}_${BATCH}_${MICROBATCH}_${SEQUENCE}_${PP_SCHEDULE}_v${PP_VIRTUAL}_sgo${SGO}_ar${ACTIVATION_RECOMPUTE}

SGO_ARGS=()
[ "${SGO}" = "1" ] && SGO_ARGS+=(--scatter_gather_optimization true)
AR_ARGS=()
[ "${ACTIVATION_RECOMPUTE}" = "1" ] && AR_ARGS+=(--activation_recompute true)

echo "[megatron_gpt_76b] DP=${DP} TP=${TP} PP=${PP} SP=${SP} (total NPUs=$((DP*TP*PP*SP)))"
echo "[megatron_gpt_76b] BATCH=${BATCH} MICROBATCH=${MICROBATCH} SEQ=${SEQUENCE} LAYER=${LAYER}"
echo "[megatron_gpt_76b] OUTPUT_DIR=${OUTPUT_DIR}"

(
cd "${STG}"
"${PY}" main.py --output_dir "${OUTPUT_DIR}" \
                --output_name workload.%d.et \
                --dp "${DP}" --tp "${TP}" --pp "${PP}" --sp "${SP}" \
                --seq "${SEQUENCE}" --batch "${BATCH}" \
                --dvocal 51200 --dmodel 10240 --dff 40960 \
                --head 80 --kvhead 80 --num_stacks "${LAYER}" \
                --micro_batch "${MICROBATCH}" \
                --num_iterations "${ITERATION}" \
                --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
                --model_type gpt \
                --mixed_precision true \
                --attention_backend "${ATTENTION}" \
                --pipeline_schedule "${PP_SCHEDULE}" \
                --pipeline_virtual_stages "${PP_VIRTUAL}" \
                "${SGO_ARGS[@]}" \
                "${AR_ARGS[@]}" \
                --weight_sharded 0
)

echo "[megatron_gpt_76b] done. Generated $(ls "${OUTPUT_DIR}"/workload.*.et 2>/dev/null | wc -l) .et files."
