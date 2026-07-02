#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

INPUT_ROOT="${INPUT_ROOT:-/home/gd09385/data/test_c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/gd09385/data/test_c_sub}"
PATCH_SIZE="${PATCH_SIZE:-512}"
STRIDE="${STRIDE:-512}"
INCLUDE_DEPTH="${INCLUDE_DEPTH:-auto}"
PAIRING_MODE="${PAIRING_MODE:-auto}"
DEPTH_PAIRING_MODE="${DEPTH_PAIRING_MODE:-auto}"

python scripts/crop_paired_dataset.py \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --patch_size "${PATCH_SIZE}" \
  --stride "${STRIDE}" \
  --include_depth "${INCLUDE_DEPTH}" \
  --pairing_mode "${PAIRING_MODE}" \
  --depth_pairing_mode "${DEPTH_PAIRING_MODE}" \
  "$@"
