import os
import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime

# ============ 配置区域 ============
TASKS = [
    "adjust_bottle", "handover_mic", "pick_dual_bottles", "place_fan", "scan_object",
    "hanging_mug", "place_a2b_left", "place_mouse_pad", "shake_bottle_horizontally",
    "beat_block_hammer", "place_a2b_right", "place_object_basket", "shake_bottle",
    "blocks_ranking_rgb", "lift_pot", "place_bread_basket", "place_object_scale",
    "stack_blocks_three", "blocks_ranking_size", "move_can_pot", "place_bread_skillet",
    "place_object_stand", "stack_blocks_two", "click_alarmclock", "move_pillbottle_pad",
    "place_burger_fries", "place_phone_stand", "stack_bowls_three", "click_bell",
    "move_playingcard_away", "place_can_basket", "place_shoe", "stack_bowls_two",
    "dump_bin_bigbin", "move_stapler_pad", "place_cans_plasticbox", "press_stapler",
    "stamp_seal", "open_laptop", "place_container_plate", "put_bottles_dustbin",
    "turn_switch", "grab_roller", "open_microwave", "place_dual_shoes", "put_object_cabinet",
    "handover_block", "pick_diverse_bottles", "place_empty_cup", "rotate_qrcode"
]

# 数据路径配置
INPUT_BASE_DIRS = [
    # "/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_randomized_500",
    "/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_clean",
    ]
OUTPUT_BASE_DIRS = [
    # "/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_randomized_500_lerobotV2",
    "/vla/users/luojingzhou/data/datasets/robotwin2_aloha_agilex_clean_lerobotV2",
    ]
INPUT_SUBDIRS = [
    # "demo_randomized", 
    "demo_clean",
    ]

CONVERT_SCRIPT = "/vla/users/luojingzhou/projects/Isaac-GR00T-1d6/scripts/lerobot_conversion/convert_robotwin2_to_v2.py"
NUM_PROCESSES = 8
# =================================

def setup_environment():
    """设置 LD_LIBRARY_PATH 环境变量"""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise EnvironmentError("CONDA_PREFIX 未设置，请在 conda 环境中运行此脚本")
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib"
    return env

def process_task(task, input_base, output_base, input_subdir, env):
    """处理单个任务的转换"""
    input_dir = Path(input_base) / task / input_subdir
    output_dir = Path(output_base) / task
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        if result.returncode != 0:
            error_msg = f"Task '{task}' failed with code {result.returncode}\nSTDERR: {result.stderr[:500]}"
            return (task, False, error_msg)
        return (task, True, f"Completed in {len(result.stdout.splitlines())} lines output")
    except subprocess.TimeoutExpired:
        return (task, False, "Timeout after 1 hour")
    except Exception as e:
        return (task, False, f"Exception: {str(e)}")

def main():
    for i in range(len(INPUT_BASE_DIRS)):
        INPUT_BASE_DIR, OUTPUT_BASE_DIR, INPUT_SUBDIR = INPUT_BASE_DIRS[i], OUTPUT_BASE_DIRS[i], INPUT_SUBDIRS[i]
        print(f"[{datetime.now()}] 开始转换任务，共 {len(TASKS)} 个任务，使用 {NUM_PROCESSES} 个进程")
        print(f"输入基础目录: {INPUT_BASE_DIR}")
        print(f"输出基础目录: {OUTPUT_BASE_DIR}\n")
        
        env = setup_environment()
        
        # 准备部分函数以传入固定参数
        worker = partial(
            process_task,
            input_base=INPUT_BASE_DIR,
            output_base=OUTPUT_BASE_DIR,
            input_subdir=INPUT_SUBDIR,
            env=env
        )
        
        # 创建进程池并执行
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.imap_unordered(worker, TASKS)
            
            success_count = 0
            failure_count = 0
            
            for i, (task, success, msg) in enumerate(results, 1):
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"[{i}/{len(TASKS)}] {status}: {task}")
                if not success:
                    print(f"  └─ 错误详情: {msg}")
                    failure_count += 1
                else:
                    success_count += 1
        
        # 汇总结果
        print("\n" + "="*60)
        print(f"转换完成: 成功 {success_count}/{len(TASKS)} | 失败 {failure_count}/{len(TASKS)}")
        print(f"输出目录: {OUTPUT_BASE_DIR}")
        print("="*60)
        
        return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())