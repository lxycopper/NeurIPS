python inference.py \
--model_id ./experiments/door-309-distill-0.2 \
--template_name "sd_pull_door_templates_v2" \
--placeholder_string "<R>" \
--num_samples 10 \
--guidance_scale 7.5 \
--device "cuda:5" \



