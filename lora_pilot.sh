accelerate launch /workspace/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --resolution=1024 \
  --output_dir="./sdxl-lora-pilot" \
  --mixed_precision="bf16" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --checkpointing_steps=250 \
  --seed=42 \
  --rank=32 \
  --gradient_checkpointing \
  --report_to="tensorboard" \
  "$@"