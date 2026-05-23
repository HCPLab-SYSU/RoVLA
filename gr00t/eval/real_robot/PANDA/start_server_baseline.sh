cd /data0/luojingzhou/projects/Isaac-GR00T-1d6
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=6
source .venv/bin/activate

uv pip install transformers==4.57.3 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
./.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path /data0/luojingzhou/data/checkpoints/GR00T-N1.6-InternVL3.5-3B-checkpoints/hcp_panda_0315/hcp_panda_all_6w/checkpoint-60000 \
    --embodiment-tag HCP_PANDA \
    --port 25788 \
    --host 0.0.0.0
