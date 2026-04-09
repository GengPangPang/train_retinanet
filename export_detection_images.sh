#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_DIR="${THIS_DIR}/../mmdetection"
PYTHON_BIN="${PYTHON_BIN:-/work/home/ackyrtsya4/users/GZX/Conda_env/FaceDet1/bin/python}"
CONFIG="${CONFIG:-${THIS_DIR}/retinanet_r50_fpn_1x_mmlab_widerface.py}"
TRAIN_WORK_DIR="${TRAIN_WORK_DIR:-${THIS_DIR}/work_dirs/retinanet_r50_fpn_1x_mmlab_widerface}"
TEST_WORK_DIR="${TEST_WORK_DIR:-${TRAIN_WORK_DIR}/vis_epoch12}"
SHOW_DIR_NAME="${SHOW_DIR_NAME:-det_images}"
WORKERS="${WORKERS:-4}"
VIS_SCORE_THR="${VIS_SCORE_THR:-0.3}"

if [[ -z "${CHECKPOINT:-}" ]]; then
    if [[ -f "${TRAIN_WORK_DIR}/last_checkpoint" ]]; then
        CHECKPOINT="$(< "${TRAIN_WORK_DIR}/last_checkpoint")"
    else
        CHECKPOINT="${TRAIN_WORK_DIR}/epoch_12.pth"
    fi
fi

if [[ ! -d "${MMDET_DIR}" ]]; then
    echo "mmdetection not found: ${MMDET_DIR}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

export PYTHONPATH="${MMDET_DIR}:${PYTHONPATH:-}"
export MMDET_USE_PRETRAINED=0

mkdir -p "${TEST_WORK_DIR}"

"${PYTHON_BIN}" "${MMDET_DIR}/tools/test.py" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --work-dir "${TEST_WORK_DIR}" \
    --show-dir "${SHOW_DIR_NAME}" \
    --cfg-options \
    "default_hooks.visualization.score_thr=${VIS_SCORE_THR}" \
    "test_dataloader.num_workers=${WORKERS}" \
    "val_dataloader.num_workers=${WORKERS}"

LATEST_RUN="$(find "${TEST_WORK_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -n "${LATEST_RUN}" ]]; then
    echo "visualization_dir=${LATEST_RUN}/${SHOW_DIR_NAME}"
fi
