#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/gd09385/models/stable-diffusion-2-base}"
SEESR_MODEL_PATH="${SEESR_MODEL_PATH:-/home/gd09385/work/CoCDiffusion/experiment/deblur_train_coc_image_latent/checkpoint-15000}"
IMAGE_PATH="${IMAGE_PATH:-/home/gd09385/data/test_c/source}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gd09385/work/CoCDiffusion/experiment/deblur_test_coc_image_latent-lr-15000-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
SAMPLE_TIMES="${SAMPLE_TIMES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CONDITIONING_SCALE="${CONDITIONING_SCALE:-1.0}"
ALIGN_METHOD="${ALIGN_METHOD:-adain}"
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
COC_INFERENCE_START="${COC_INFERENCE_START:-encoded_input}"
COC_IMAGE_LATENT_REVERSE="${COC_IMAGE_LATENT_REVERSE:-scheduler}"
COC_IMAGE_LATENT_TIMESTEP_SPACING="${COC_IMAGE_LATENT_TIMESTEP_SPACING:-scheduler}"
COC_IMAGE_LATENT_NORMALIZE_START="${COC_IMAGE_LATENT_NORMALIZE_START:-0}"
COC_NOISE_NORMALIZATION="${COC_NOISE_NORMALIZATION:-sample}"
COC_NOISE_NORMALIZATION_EPS="${COC_NOISE_NORMALIZATION_EPS:-1e-6}"
DIFFUSION_PROCESS="${DIFFUSION_PROCESS:-coc_image_latent}"
USE_DEPTH="${USE_DEPTH:-}"
START_BLUR_SIGMA="${START_BLUR_SIGMA:-8.0}"
START_BLUR_KERNEL_SIZE="${START_BLUR_KERNEL_SIZE:-}"
START_STEPS="${START_STEPS:-201}"
UPDATE_BLEND="${UPDATE_BLEND:-1.0}"
TIMESTEP_CONDITIONING="${TIMESTEP_CONDITIONING:-auto}"

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
  --diffusion_process "${DIFFUSION_PROCESS}"
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
  --coc_inference_start "${COC_INFERENCE_START}"
  --coc_image_latent_reverse "${COC_IMAGE_LATENT_REVERSE}"
  --coc_image_latent_timestep_spacing "${COC_IMAGE_LATENT_TIMESTEP_SPACING}"
  --coc_noise_normalization "${COC_NOISE_NORMALIZATION}"
  --coc_noise_normalization_eps "${COC_NOISE_NORMALIZATION_EPS}"
  --start_blur_sigma "${START_BLUR_SIGMA}"
  --start_steps "${START_STEPS}"
  --update_blend "${UPDATE_BLEND}"
  --timestep_conditioning "${TIMESTEP_CONDITIONING}"
)

if [[ "${COC_IMAGE_LATENT_NORMALIZE_START}" == "1" ]]; then
  cmd+=(--coc_image_latent_normalize_start)
fi

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
