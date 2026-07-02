#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/gd09385/models/stable-diffusion-2-base}"
SEESR_MODEL_PATH="${SEESR_MODEL_PATH:-/home/gd09385/work/CoCDiffusion/experiment/deblur_train_coc_blur/checkpoint-121000}"
IMAGE_PATH="${IMAGE_PATH:-/home/gd09385/data/test_c/source}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gd09385/work/CoCDiffusion/experiment/deblur_test_coc_blur-121000-rawinput-step20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
SAMPLE_TIMES="${SAMPLE_TIMES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CONDITIONING_SCALE="${CONDITIONING_SCALE:-1.0}"
ALIGN_METHOD="${ALIGN_METHOD:-adain}"
COC_FOCUS_DEPTH="${COC_FOCUS_DEPTH:-0.7}"
COC_MAX_RADIUS="${COC_MAX_RADIUS:-2.5}"
COC_GAMMA="${COC_GAMMA:-1.5}"
COC_SCHEDULE_POWER="${COC_SCHEDULE_POWER:-1.0}"
COC_INFERENCE_START="${COC_INFERENCE_START:-encoded_input}"
USE_DEPTH="${USE_DEPTH:-}"
START_BLUR_SIGMA="${START_BLUR_SIGMA:-8.0}"
START_BLUR_KERNEL_SIZE="${START_BLUR_KERNEL_SIZE:-}"
UPDATE_BLEND="${UPDATE_BLEND:-1.0}"

cmd=(
  python test_seesr.py
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}"
  --seesr_model_path "${SEESR_MODEL_PATH}"
  --image_path "${IMAGE_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --start_point lr
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --sample_times "${SAMPLE_TIMES}"
  --mixed_precision "${MIXED_PRECISION}"
  --conditioning_scale "${CONDITIONING_SCALE}"
  --align_method "${ALIGN_METHOD}"
  --diffusion_process coc_blur
  --coc_focus_depth "${COC_FOCUS_DEPTH}"
  --coc_max_radius "${COC_MAX_RADIUS}"
  --coc_gamma "${COC_GAMMA}"
  --coc_schedule_power "${COC_SCHEDULE_POWER}"
  --coc_inference_start "${COC_INFERENCE_START}"
  --start_blur_sigma "${START_BLUR_SIGMA}"
  --update_blend "${UPDATE_BLEND}"
  --timestep_conditioning off
)

if [[ -n "${USE_DEPTH}" ]]; then
  cmd+=(--use_depth)
fi

if [[ -n "${DEPTH_PATH:-}" ]]; then
  cmd+=(--depth_path "${DEPTH_PATH}")
fi

if [[ -n "${START_BLUR_KERNEL_SIZE}" ]]; then
  cmd+=(--start_blur_kernel_size "${START_BLUR_KERNEL_SIZE}")
fi

"${cmd[@]}" "$@"
