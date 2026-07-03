#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/gd09385/anaconda3/bin/activate seesr
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/gd09385/models/stable-diffusion-2-base}"
SEESR_MODEL_PATH="${SEESR_MODEL_PATH:-/home/gd09385/work/CoCDiffusion/experiment/deblur_train_ddim/checkpoint-40000}"
IMAGE_PATH="${IMAGE_PATH:-/home/gd09385/data/test_c/source}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gd09385/work/CoCDiffusion/experiment/deblur_test_ddim-40000-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
SAMPLE_TIMES="${SAMPLE_TIMES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CONDITIONING_SCALE="${CONDITIONING_SCALE:-1.0}"
ALIGN_METHOD="${ALIGN_METHOD:-adain}"
START_STEPS="${START_STEPS:-999}"
TIMESTEP_CONDITIONING="${TIMESTEP_CONDITIONING:-off}"

python test_seesr.py \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
  --seesr_model_path "${SEESR_MODEL_PATH}" \
  --image_path "${IMAGE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --start_point lr \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --sample_times "${SAMPLE_TIMES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --conditioning_scale "${CONDITIONING_SCALE}" \
  --align_method "${ALIGN_METHOD}" \
  --diffusion_process gaussian \
  --start_steps "${START_STEPS}" \
  --timestep_conditioning "${TIMESTEP_CONDITIONING}" \
  "$@"
