set -x -e
export CPATH=$CONDA_PREFIX/include/python3.10:$CPATH
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=1,2,3,6
export NUM_GPUS=4
source .venv/bin/activate
uv pip install transformers==4.57.3 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
torchrun --nproc_per_node=$NUM_GPUS --standalone \
    gr00t/experiment/launch_finetune.py \
    --base_model_path /data0/luojingzhou/data/checkpoints/GR00T-N1.6-InternVL3.5-3B \
    --dataset_path /data0/luojingzhou/data/datasets/hcp_panda_real_data_easy_v2_lerobotv2 \
    --embodiment_tag HCP_PANDA \
    --num_gpus $NUM_GPUS \
    --output_dir /data0/luojingzhou/data/checkpoints/GR00T-N1.6-InternVL3.5-3B-checkpoints/hcp_panda/rovla \
    --save_steps 5000 \
    --save_total_limit 5 \
    --max_steps 60000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 128 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 8 \
    --use_pgd \
    --use_consistency_learning \
    --deepspeed_stage 2 \
    --gradient_checkpointing