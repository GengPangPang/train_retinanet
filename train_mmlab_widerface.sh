#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_DIR="${THIS_DIR}/../mmdetection"
CONFIG="${THIS_DIR}/retinanet_r50_fpn_1x_mmlab_widerface.py"
WORK_DIR="${WORK_DIR:-${THIS_DIR}/work_dirs/retinanet_r50_fpn_1x_mmlab_widerface}"

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"
PORT="${PORT:-29500}"
AMP="${AMP:-0}"
AUTO_SCALE_LR="${AUTO_SCALE_LR:-1}"
USE_PRETRAINED="${USE_PRETRAINED:-1}"
RESUME="${RESUME:-0}"

if [[ ! -d "${MMDET_DIR}" ]]; then
    echo "mmdetection not found: ${MMDET_DIR}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "config not found: ${CONFIG}" >&2
    exit 1
fi

export PYTHONPATH="${MMDET_DIR}:${PYTHONPATH:-}"
export MMDET_USE_PRETRAINED="${USE_PRETRAINED}"

mkdir -p "${WORK_DIR}"

ARGS=(
    --work-dir "${WORK_DIR}"
    --cfg-options
    "train_dataloader.batch_size=${BATCH_SIZE}"
    "train_dataloader.num_workers=${WORKERS}"
    "val_dataloader.num_workers=${WORKERS}"
    "test_dataloader.num_workers=${WORKERS}"
)

if [[ "${AMP}" == "1" ]]; then
    ARGS+=(--amp)
fi

if [[ "${AUTO_SCALE_LR}" == "1" ]]; then
    ARGS+=(--auto-scale-lr)
fi

if [[ "${RESUME}" == "1" ]]; then
    ARGS+=(--resume)
fi

if [[ "${GPUS}" -gt 1 ]]; then
    PORT="${PORT}" bash "${MMDET_DIR}/tools/dist_train.sh" "${CONFIG}" "${GPUS}" "${ARGS[@]}"
else
    python "${MMDET_DIR}/tools/train.py" "${CONFIG}" "${ARGS[@]}"
fi
