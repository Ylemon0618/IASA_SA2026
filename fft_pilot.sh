PRETRAINED_MODEL=""
TRAIN_DATA_DIR=""
OUTPUT_DIR=""

for arg in "$@"; do
  case $arg in
    --pretrained_model_name_or_path=*) PRETRAINED_MODEL="${arg#*=}" ;;
    --train_data_dir=*) TRAIN_DATA_DIR="${arg#*=}" ;;
    --output_dir=*) OUTPUT_DIR="${arg#*=}" ;;
  esac
done

if [ -z "$PRETRAINED_MODEL" ] || [ -z "$TRAIN_DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: fft_pilot.sh --pretrained_model_name_or_path=... --train_data_dir=... --output_dir=..."
  exit 1
fi

export HTTPX_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=warning
export DIFFUSERS_VERBOSITY=warning
export HF_HUB_VERBOSITY=warning

SCRIPT="diffusers/examples/text_to_image/train_text_to_image_sdxl.py"
if [ ! -f "$SCRIPT" ]; then
  echo "Script not found: $SCRIPT"
  echo "Run: git clone https://github.com/huggingface/diffusers.git"
  exit 1
fi

accelerate launch "$SCRIPT" \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --resolution=1024 \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --checkpointing_steps=250 \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --report_to="tensorboard"
