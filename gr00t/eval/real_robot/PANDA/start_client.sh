cd /media/hcp/disk/projects/Isaac-GR00T-1d6
export HF_ENDPOINT=https://hf-mirror.com 
export CUDA_VISIBLE_DEVICES=0
source $HOME/miniforge3/bin/activate datacollect
# source .venv/bin/activate
# uv pip install transformers==4.57.3 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
# ./.venv/bin/python ./gr00t/eval/real_robot/PANDA/eval_panda.py \
#433数量验证
#=======pick banana=====
# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "pick up the banana." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "take the banana." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "pick up the curved fruit with a yello skin." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

#======pick apple======
# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "pick up the red apple." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "take the red apple." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "pick up the shiney, round fruit with a red skin." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

#=====put banana bowl==========
# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "put the banana in the bowl." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "place the banana into the bowl." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "place the yellow, curved fruit into the round container." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

#=====put red apple bowl==========
python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
    --policy_host 222.200.185.196 \
    --record_video \
    --policy_port 23788 \
    --lang_instruction "put the red apple in the bowl." \
    --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "place the red apple into the bowl." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "place the red, round fruit into the round container." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

#======put apple drawer========
# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "put the red apple in the drawer." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "place the red apple into the drawer." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \

# python ./gr00t/eval/real_robot/PANDA/eval_panda_franky.py \
#     --policy_host 222.200.185.196 \
#     --record_video \
#     --policy_port 23788 \
#     --lang_instruction "place the red, round fruit in the storage compartment." \
#     --experiment_path /media/hcp/disk/projects/panda_rollout/mtc-vla/checkpoint-60000 \
