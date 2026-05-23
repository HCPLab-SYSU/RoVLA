export HOME=/vla/users/luojingzhou 
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=3,4,5,6
conda activate z_robotwin2
python gr00t/eval/rollout_policy_multigpus.py \
    --n_episodes 100 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 43786 \
    --max_episode_steps=1700 \
    --benchmark_name robotwin/all \
    --n_action_steps 8 \
    --n_envs 1 \
    --use_spawn \
    --experinment_path /vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/robotwin/robotwin_all_12w_pgd_step3_wonoiseadv_epsilon0d03_cslearning_tasklist/checkpoint-120000



