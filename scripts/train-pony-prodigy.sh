#! /bin/bash

source ./scripts/common.sh

export HF_HOME=${REMOTE_ROOT:-/workspace}/hf
export SDXL_PATH=${REMOTE_ROOT:-/workspace}/sdxl
export TOKENIZERS_PARALLELISM=true

TRAINING_NAME=${1:-error}

if [ "${TRAINING_NAME}" == "error" ]; then
  echo "Usage: $0 <training-name>"
  exit 1
fi

# Hyperparams
PARAMS_LEARNING_RATE=1e-4
PARAMS_LEARNING_SCHEDULER=constant
PARAMS_MAX_STEPS=750
PARAMS_NETWORK_ALPHA=4
PARAMS_NETWORK_DIM=32
PARAMS_OPTIMIZER_ARGS=""
PARAMS_RESUME_FROM=null
PARAMS_SAVE_EVERY_N_STEPS=200
PARAMS_SAMPLE_EVERY_N_STEPS=100

DEFAULT_NUM_CYCLES=$(( ${PARAMS_MAX_STEPS} / 250 ))

# Load override params from config.env
source ${REMOTE_ROOT:-/workspace}/config/${TRAINING_NAME}/config.env

# Training params
DATASET_CONFIG=${REMOTE_ROOT:-/workspace}/config/${TRAINING_NAME}/dataset.json
OUTPUT_DIR=${REMOTE_OUTPUT:-/output}
OUTPUT_NAME=${TRAINING_SLUG}

# If optimizer args are set, prefix them with the option
OPTIMIZER_ARGS=""
if [ -n "${PARAMS_OPTIMIZER_ARGS:-}" ] && [ "${PARAMS_OPTIMIZER_ARGS:-}" != "null" ]; then
  OPTIMIZER_ARGS=''
fi

# If resume-from is set, resume training from the specified file
RESUME_WEIGHTS=""
if [ -n "${PARAMS_RESUME_FROM:-}" ] && [ "${PARAMS_RESUME_FROM:-}" != "null" ]; then
  RESUME_WEIGHTS="--network_weights ${REMOTE_ROOT:-/workspace}/dataset/${PARAMS_RESUME_FROM}"
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}/${OUTPUT_NAME}
mkdir -p ${OUTPUT_DIR}/sample/${OUTPUT_NAME}
mkdir -p ${OUTPUT_DIR}/logging/${OUTPUT_NAME}

# ./scripts/download-model.sh

# Make sure GPU ID is set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES not set"
  exit 1
fi

# Print env for debugging
env
echo "Continue in 10 seconds..."
sleep 10

source ${REMOTE_ROOT:-/workspace}/sd-scripts/venv/bin/activate
wandb login ${WANDB_API_KEY}

# --network_train_unet_only
# Train
time accelerate launch \
        --mixed_precision no \
        --num_cpu_threads_per_process 2 \
        ${REMOTE_ROOT:-/workspace}/sd-scripts/sdxl_train_network.py \
        --pretrained_model_name_or_path ${SDXL_PATH}/pony_v6_base.safetensors \
        --no_half_vae \
        --cache_latents_to_disk \
        --save_model_as safetensors \
        --sdpa \
        --persistent_data_loader_workers \
        --max_data_loader_n_workers 2 \
        --seed 42 \
        --gradient_checkpointing \
        --mixed_precision no \
        --save_precision float \
        --min_snr 5 \
        --network_module networks.lora \
        --network_alpha ${PARAMS_NETWORK_ALPHA} \
        --network_dim ${PARAMS_NETWORK_DIM} \
        --network_train_unet_only \
        --optimizer_type ${PARAMS_OPTIMIZER_TYPE} \
        --optimizer_args \
            "decouple=True" \
            "weight_decay=${PARAMS_WEIGHT_DECAY}" \
            "betas=0.9,0.999" \
            "use_bias_correction=False" \
            "safeguard_warmup=False" \
            "d_coef=${PARAMS_D_COEF:-2}" \
        --network_args "train_t5xxl=False" "split_qkv=True" \
        --cache_text_encoder_outputs \
        --cache_text_encoder_outputs_to_disk \
        --lr_scheduler ${PARAMS_LEARNING_SCHEDULER} \
        --lr_scheduler_num_cycles ${PARAMS_NUM_CYCLES:-${DEFAULT_NUM_CYCLES}} \
        --noise_offset ${PARAMS_NOISE_OFFSET} \
        --learning_rate 1.0 \
        --highvram \
        --max_train_steps=${PARAMS_MAX_STEPS} \
        --huber_schedule=snr \
        --min_snr_gamma=5 \
        --save_every_n_steps ${PARAMS_SAVE_EVERY_N_STEPS} \
        --dataset_config ${DATASET_CONFIG} \
        --output_dir ${OUTPUT_DIR} \
        --output_name ${OUTPUT_NAME} \
        --sample_every_n_steps ${PARAMS_SAMPLE_EVERY_N_STEPS} \
        --sample_prompts ${REMOTE_ROOT:-/workspace}/config/${TRAINING_NAME}/sample-prompts.txt \
        --log_with=all \
        --logging_dir ${OUTPUT_DIR}/logging/${OUTPUT_NAME} \
        --wandb_run_name=${TRAINING_NAME}-${PARAMS_OPTIMIZER_TYPE}-${PARAMS_LEARNING_SCHEDULER}-$(date +%s) \
        ${RESUME_WEIGHTS}

echo "Training complete"
