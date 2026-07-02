#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/gd09385/models/stable-diffusion-2-base}"
SEESR_MODEL_PATH="${SEESR_MODEL_PATH:-/home/gd09385/models/seesr}"
ROOT_FOLDERS="${ROOT_FOLDERS:-/home/gd09385/data/test_c_sub}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gd09385/work/CoCDiffusion/experiment/deblur_train_coc_image_latent}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
UNET_LEARNING_RATE="${UNET_LEARNING_RATE:-1e-6}"
UNET_TRAIN_PRESET="${UNET_TRAIN_PRESET:-controlnet_interaction_full}"
UNET_TRAINABLE_MODULES="${UNET_TRAINABLE_MODULES:-}"
COC_TRAIN_INPUT_MODE="${COC_TRAIN_INPUT_MODE:-conditioning}"
DIFFUSION_PROCESS="${DIFFUSION_PROCESS:-coc_image_latent}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
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
TIMESTEP_CONDITIONING="${TIMESTEP_CONDITIONING:-auto}"

cmd=(
  accelerate launch train_seesr.py
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_PATH}"
  --controlnet_model_name_or_path "${SEESR_MODEL_PATH}"
  --unet_model_name_or_path "${SEESR_MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --root_folders "${ROOT_FOLDERS}"
  --enable_xformers_memory_efficient_attention
  --mixed_precision "${MIXED_PRECISION}"
  --learning_rate "${LEARNING_RATE}"
  --unet_learning_rate "${UNET_LEARNING_RATE}"
  --unet_train_preset "${UNET_TRAIN_PRESET}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
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
  --timestep_conditioning "${TIMESTEP_CONDITIONING}"
)

if [[ "${DIFFUSION_PROCESS}" == "coc_blur" ]]; then
  cmd+=(--coc_train_input_mode "${COC_TRAIN_INPUT_MODE}")
fi

if [[ -n "${UNET_TRAINABLE_MODULES}" ]]; then
  read -r -a UNET_TRAINABLE_MODULE_ARRAY <<< "${UNET_TRAINABLE_MODULES}"
  cmd+=(--unet_trainable_modules "${UNET_TRAINABLE_MODULE_ARRAY[@]}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  cmd+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

"${cmd[@]}" "$@"
