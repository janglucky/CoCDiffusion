#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

PREDICTION_PATH="${PREDICTION_PATH:-/home/gd09385/work/CoCDiffusion/experience/deblur_test_c_sub/sample00}"
TARGET_PATH="${TARGET_PATH:-/home/gd09385/data/train_c_sub/target}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
CROP_BORDER="${CROP_BORDER:-0}"

cmd=(
  python eval_seesr.py
  --prediction_path "${PREDICTION_PATH}"
  --target_path "${TARGET_PATH}"
  --crop_border "${CROP_BORDER}"
)

if [[ -n "${OUTPUT_JSON}" ]]; then
  cmd+=(--output_json "${OUTPUT_JSON}")
fi

"${cmd[@]}" "$@"
