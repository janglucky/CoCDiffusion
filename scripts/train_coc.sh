#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/gd09385/models/stable-diffusion-2-base}"
SEESR_MODEL_PATH="${SEESR_MODEL_PATH:-/home/gd09385/models/seesr}"
ROOT_FOLDERS="${ROOT_FOLDERS:-/home/gd09385/data/test_c_sub}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gd09385/work/CoCDiffusion/experiment/deblur_train_coc_blur}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
COC_FOCUS_DEPTH="${COC_FOCUS_DEPTH:-0.7}"
COC_MAX_RADIUS="${COC_MAX_RADIUS:-2.5}"
COC_GAMMA="${COC_GAMMA:-1.5}"
COC_SCHEDULE_POWER="${COC_SCHEDULE_POWER:-1.0}"

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
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --diffusion_process coc_blur
  --coc_focus_depth "${COC_FOCUS_DEPTH}"
  --coc_max_radius "${COC_MAX_RADIUS}"
  --coc_gamma "${COC_GAMMA}"
  --coc_schedule_power "${COC_SCHEDULE_POWER}"
  --timestep_conditioning off
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  cmd+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

"${cmd[@]}" "$@"
