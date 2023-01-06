export MODEL_NAME="hakurei/waifu-diffusion"
export INSTANCE_DIR="instance-images/okoma"
export OUTPUT_DIR="model/okoma2"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint="checkpoint-100" \
  --instance_prompt="okoma" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=600 \
  --mixed_precision=fp16 \
  --checkpointing_steps=200