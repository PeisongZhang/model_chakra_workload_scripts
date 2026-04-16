#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../../symbolic_tensor_graph

# Run Symbolic Tensor Graph (STG) Generator for Qwen-32B
# Configuration from qwen_32b_config.json and user requirements:
# seq=4096, batch=1024, tp=8, pp=4, dp=4
(
cd ${STG}
python3 main.py --output_dir ${SCRIPT_DIR}/toy_model/ \
               --output_name toymodel.%d.et \
               --dp 4 --tp 4 --pp 4 \
               --seq 2048 --batch 256 \
               --dvocal 65536 --dmodel 1024 --dff 4096 \
               --head 32 --kvhead 8 --num_stacks 4 \
               --micro_batch 4 \
               --model_type llama \
               --weight_sharded 0
)
