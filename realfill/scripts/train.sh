export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="../data/bear"
export OUTPUT_DIR="../pth_file/bear-model"

accelerate launch ../train_realfill.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --lora_rank=8 \
  --lora_dropout=0.1 \
  --lora_alpha=16 \