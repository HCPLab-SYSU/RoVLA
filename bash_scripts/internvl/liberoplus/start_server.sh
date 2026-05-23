export HOME=/vla/users/luojingzhou 
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0,1
source .venv/bin/activate
uv pip install transformers==4.57.3 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
./.venv/bin/python gr00t/eval/run_gr00t_server_multigpus.py \
    --model-path /vla/users/luojingzhou/data/checkpoints/GR00TN-1.6-InterVL3.5-3B/liberoplus/rovla/checkpoint-60000 \
    --embodiment-tag LIBEROPLUS_PANDA \
    --use-sim-policy-wrapper \
    --port 23782