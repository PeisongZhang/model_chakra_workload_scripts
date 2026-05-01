#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(realpath $0)")

LOG_DIR=${SCRIPT_DIR}/logs_sgd_parallel
mkdir -p "${LOG_DIR}"

SGD=standard bash "${SCRIPT_DIR}/llama3_8b.sh" \
    >"${LOG_DIR}/standard.log" 2>&1 &
PID_STANDARD=$!

SGD=local bash "${SCRIPT_DIR}/llama3_8b.sh" \
    >"${LOG_DIR}/localsgd.log" 2>&1 &
PID_LOCALSGD=$!

echo "Launched standard SGD (pid=${PID_STANDARD}), localsgd (pid=${PID_LOCALSGD})"
echo "Logs: ${LOG_DIR}/{standard,localsgd}.log"

FAIL=0
wait ${PID_STANDARD} || { echo "standard SGD failed" >&2; FAIL=1; }
wait ${PID_LOCALSGD} || { echo "localsgd failed" >&2; FAIL=1; }

exit ${FAIL}
