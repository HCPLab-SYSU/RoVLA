export HOME=/vla/users/luojingzhou 
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=3,4,5,6
source .venv/bin/activate
uv pip install transformers==4.57.3 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
./.venv/bin/python gr00t/eval/run_gr00t_server_multigpus.py \
    --model-path /vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/robotwin/robotwin_all_12w_pgd_step3_wonoiseadv_epsilon0d03_cslearning_tasklist/checkpoint-120000 \
    --embodiment-tag ROBOTWIN_ALOHA_AGILEX \
    --use-sim-policy-wrapper \
    --port 43786
