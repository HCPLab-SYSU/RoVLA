export HOME=/vla/users/luojingzhou 
export LIBERO_CONFIG_PATH=/vla/users/luojingzhou/.liberoplus
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0,1
gr00t/eval/sim/LIBEROPLUS/liberoplus_uv/.venv/bin/python gr00t/eval/rollout_policy_multigpus.py \
    --n_episodes 1 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 23782 \
    --max_episode_steps=720 \
    --benchmark_name liberoplus/all \
    --n_action_steps 8 \
    --n_envs 1 \
    --use_spawn \
    --experinment_path /vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/liberoplus/rovla/checkpoint-60000