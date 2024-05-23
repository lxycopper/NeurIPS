export OUT_DIR="./experiments/door-distill-0.2"

accelerate launch \
    --config_file="./configs/single_gpu.yml" \
    train.py \
    --seed="2023" \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="./reversion_benchmark/door_3_8" \
    --placeholder_token="<R>" \
    --initializer_token="and" \
    --train_batch_size="2" \
    --gradient_accumulation_steps="4" \
    --max_train_steps="3000" \
    --learning_rate='2.5e-04' --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps="0" \
    --output_dir=$OUT_DIR \
    --checkpointing_steps="1000" \
    --save_steps="1000" \
    --importance_sampling \
    --denoise_loss_weight="1.0" \
    --distill_loss_weight="0.2" \
    --steer_loss_weight="0.01" \
    --num_positives="4" \
    --temperature="0.07"

python inference.py \
--model_id $OUT_DIR \
--template_name "simple_pull_door_templates" \
--placeholder_string "<R>" \
--num_samples 10 \
--guidance_scale 7.5 \
--device "cuda:0" \
