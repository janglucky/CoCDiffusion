#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/gd09385/models/stable-diffusion-2-base}"
IMAGE_PATH="${IMAGE_PATH:-/home/gd09385/data/test_c/target/1P0A0916.png}"
DEPTH_PATH="${DEPTH_PATH:-/home/gd09385/data/test_c/depth/1P0A0917.png}"
SOURCE_PATH="${SOURCE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gd09385/work/CoCDiffusion/experiment/coc_forward_visualization}"
MODE="${MODE:-both}"
TIMESTEPS="${TIMESTEPS:-0 100 250 500 750 999}"
MAX_SIZE="${MAX_SIZE:-768}"
SEED="${SEED:-123}"
MATRIX_ROWS="${MATRIX_ROWS:-5}"
MATRIX_AXIS="${MATRIX_AXIS:-global_blur}"
COC_FOCUS_DEPTH="${COC_FOCUS_DEPTH:-0.7}"
COC_FOCUS_WIDTH="${COC_FOCUS_WIDTH:-0.0}"
COC_FOCUS_DEPTH_MIN="${COC_FOCUS_DEPTH_MIN:-0.1}"
COC_FOCUS_DEPTH_MAX="${COC_FOCUS_DEPTH_MAX:-0.9}"
COC_FOCUS_WIDTH_MIN="${COC_FOCUS_WIDTH_MIN:-0.0}"
COC_FOCUS_WIDTH_MAX="${COC_FOCUS_WIDTH_MAX:-0.12}"
COC_GLOBAL_BLUR_MIN="${COC_GLOBAL_BLUR_MIN:-0.0}"
COC_GLOBAL_BLUR_MAX="${COC_GLOBAL_BLUR_MAX:-1.0}"
COC_MAX_RADIUS="${COC_MAX_RADIUS:-2.5}"
COC_GAMMA="${COC_GAMMA:-1.5}"
COC_SCHEDULE_POWER="${COC_SCHEDULE_POWER:-3.0}"
COC_GLOBAL_BLUR_AT_MAX="${COC_GLOBAL_BLUR_AT_MAX:-0.0}"
COC_DEPTH_BLUR_STRENGTH="${COC_DEPTH_BLUR_STRENGTH:-1.0}"
IMAGE_RADIUS_MULTIPLIER="${IMAGE_RADIUS_MULTIPLIER:-8.0}"

read -r -a TIMESTEP_ARRAY <<< "${TIMESTEPS}"

cmd=(
  python scripts/visualize_coc_forward.py
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}"
  --image_path "${IMAGE_PATH}"
  --depth_path "${DEPTH_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --mode "${MODE}"
  --timesteps "${TIMESTEP_ARRAY[@]}"
  --max_size "${MAX_SIZE}"
  --seed "${SEED}"
  --matrix_rows "${MATRIX_ROWS}"
  --matrix_axis "${MATRIX_AXIS}"
  --coc_focus_depth "${COC_FOCUS_DEPTH}"
  --coc_focus_width "${COC_FOCUS_WIDTH}"
  --coc_focus_depth_min "${COC_FOCUS_DEPTH_MIN}"
  --coc_focus_depth_max "${COC_FOCUS_DEPTH_MAX}"
  --coc_focus_width_min "${COC_FOCUS_WIDTH_MIN}"
  --coc_focus_width_max "${COC_FOCUS_WIDTH_MAX}"
  --coc_global_blur_min "${COC_GLOBAL_BLUR_MIN}"
  --coc_global_blur_max "${COC_GLOBAL_BLUR_MAX}"
  --coc_max_radius "${COC_MAX_RADIUS}"
  --coc_gamma "${COC_GAMMA}"
  --coc_schedule_power "${COC_SCHEDULE_POWER}"
  --coc_global_blur_at_max "${COC_GLOBAL_BLUR_AT_MAX}"
  --coc_depth_blur_strength "${COC_DEPTH_BLUR_STRENGTH}"
  --image_radius_multiplier "${IMAGE_RADIUS_MULTIPLIER}"
)

if [[ -n "${SOURCE_PATH}" ]]; then
  cmd+=(--source_path "${SOURCE_PATH}")
fi

"${cmd[@]}" "$@"
