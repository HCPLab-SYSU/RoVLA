import json
import os
import pathlib
import shutil
from typing import Dict, List

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import sys
from pathlib import Path

# ======================
# 配置区
# ======================
ROBOT_TYPE = "robotwin_aloha_agilex"
FPS = 20
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 1000  # episodes per chunk

CAMERA_NAMES = ["head_camera", "left_camera", "right_camera"]
CAMERA_KEYS = {
    "head_camera": "observation.images.head_view",
    "left_camera": "observation.images.left_wrist_view",
    "right_camera": "observation.images.right_wrist_view",
}

JOINT_NAMES = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper"
]


def process_state(f):
    left_arm = f["joint_action/left_arm"][:-1]
    left_gripper = f["joint_action/left_gripper"][:-1].reshape(-1, 1)
    right_arm = f["joint_action/right_arm"][:-1]
    right_gripper = f["joint_action/right_gripper"][:-1].reshape(-1, 1)
    return np.hstack([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float64)


def process_action(f):
    left_arm = f["joint_action/left_arm"][1:]
    left_gripper = f["joint_action/left_gripper"][1:].reshape(-1, 1)
    right_arm = f["joint_action/right_arm"][1:]
    right_gripper = f["joint_action/right_gripper"][1:].reshape(-1, 1)
    return np.hstack([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float64)


def decode_and_resize_image(img, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    # 解码图像数据
    if isinstance(img, bytes):
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)#RGB
    
    # 获取原始图像尺寸和目标尺寸
    old_size = img.shape[:2]  # (height, width)
    ratio = min(float(target_size[0])/old_size[0], float(target_size[1])/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size[::-1]])  # 注意宽高顺序
    
    # 等比例缩放图像
    img = cv2.resize(img, (new_size[0], new_size[1]))
    
    # 创建一个带有padding的新图像（底色为黑色）
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]  # 黑色边框
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


def process_images_rgb(f, camera_name: str):
    """Return (T, H, W, 3) uint8 array in RGB"""
    rgb_data = f["observation"][camera_name]["rgb"]
    images = []
    for i in range(len(rgb_data) - 1):
        img = decode_and_resize_image(rgb_data[i])
        images.append(img)
    return np.stack(images) if images else np.empty((0, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


def write_video(frames: np.ndarray, output_path: pathlib.Path, fps: int = 20):
    """
    Write uint8 RGB frames (T, H, W, 3) to MP4 using FFmpeg via subprocess.
    Much faster than OpenCV for batch writing.
    """
    if frames.size == 0:
        print(f"⚠️ Skipping empty video: {output_path}")
        return

    T, H, W, C = frames.shape
    assert C == 3 and frames.dtype == np.uint8, "Expected uint8 RGB frames of shape (T, H, W, 3)"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without prompt
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",  # input from stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",  # for compatibility
        "-preset", "fast",      # balance speed/size
        "-crf", "23",           # visual quality (lower = better, 18-28 typical)
        str(output_path)
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10**8  # large buffer for video data
        )
        # Feed all frames as raw RGB bytes
        process.stdin.write(frames.tobytes())
        process.stdin.close()
        stderr = process.stderr.read()
        ret = process.wait()

        if ret != 0:
            raise RuntimeError(f"FFmpeg failed with code {ret}. Stderr:\n{stderr.decode()}")

    except FileNotFoundError:
        raise EnvironmentError("FFmpeg not found. Please install FFmpeg (e.g., `apt install ffmpeg` or `brew install ffmpeg`).")
    except Exception as e:
        print(f"❌ Failed to write video {output_path}: {e}")
        raise


def load_task_instruction(episode_path: str) -> str:
    ep_num = int(os.path.basename(episode_path).replace("episode", "").replace(".hdf5", ""))
    instruction_dir = os.path.join(os.path.dirname(os.path.dirname(episode_path)), "instructions")
    instruction_path = os.path.join(instruction_dir, f"episode{ep_num}.json")
    with open(instruction_path) as f:
        instr_dict = json.load(f)
        seen_instrs = instr_dict.get("seen", [])
        if not seen_instrs:
            seen_instrs = instr_dict.get("unseen", [])
        if not seen_instrs:
            raise ValueError("No instructions found")
        new_seen_instrs = []
        if not isinstance(seen_instrs, list):
            seen_instrs = [seen_instrs]
        for instr in seen_instrs:
            if not instr.endswith("."):
                instr += "."
            new_seen_instrs.append(instr)
        return new_seen_instrs


def save_episode_parquet(
    episode_index: int,
    states: np.ndarray,
    actions: np.ndarray,
    task_index: int,
    output_chunk_dir: pathlib.Path
):
    T = len(states)
    data = {
        "observation.state": [s for s in states],
        "action": [a for a in actions],
        "timestamp": np.arange(T, dtype=np.float64) / FPS,
        "task_index": np.full(T, task_index, dtype=np.int64),
        "episode_index": np.full(T, episode_index, dtype=np.int64),
        "index": np.arange(T, dtype=np.int64),
        "next.done": np.concatenate([np.zeros(T - 1, dtype=bool), [True]]),
        "next.reward": np.concatenate([np.zeros(T - 1), [1]]),
    }

    schema = pa.schema([
        ("observation.state", pa.list_(pa.float64(), STATE_DIM)),
        ("action", pa.list_(pa.float64(), ACTION_DIM)),
        ("timestamp", pa.float64()),
        ("task_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("next.done", pa.bool_()),
        ("next.reward", pa.float64()),
    ])

    table = pa.table(data, schema=schema)
    parquet_path = output_chunk_dir / f"episode_{episode_index:06d}.parquet"
    pq.write_table(table, parquet_path)


def main(input_path: str, output_path: str):
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    # Collect episodes
    episode_dir = input_path / "data"
    episode_files = sorted([f for f in episode_dir.glob("episode*.hdf5")], key=lambda x: int(x.stem[7:]))
    total_episodes = len(episode_files)
    info_path = meta_dir / "info.json"
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        if 'clean' in str(input_path) and info['total_episodes']==50:
            print(f"skip, input_path:{input_path}") 
            return
        elif 'clean' not in str(input_path) and info['total_episodes']==500:
            print(f"skip, input_path:{input_path}") 
            return
    # First pass: collect unique tasks
    task_to_index: Dict[str, int] = {}
    task_to_all_instructions_list = {}
    all_instructions = []
    for ep_file in episode_files:
        instrs = load_task_instruction(str(ep_file))
        instr = instrs[0]
        all_instructions.append(instr)
        if instr not in task_to_index:
            task_to_index[instr] = len(task_to_index)
            task_to_all_instructions_list[instr]=instrs
    index_to_task = {v: k for k, v in task_to_index.items()}
    # Write tasks.jsonl
    with open(meta_dir / "tasks_raw.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task, "task_list":task_to_all_instructions_list[task]}) + "\n")
    total_frames = 0

    episode_path = meta_dir / "episodes.jsonl"
    if episode_path.exists():
        episode_path.unlink()
        print(f"Deleted: {episode_path}")
    # Second pass: convert each episode
    for ep_idx, ep_file in enumerate(episode_files):
        print(f"Processing episode {ep_idx}: {ep_file.name}")

        with h5py.File(ep_file, "r") as f:
            states = process_state(f)
            actions = process_action(f)
            head_imgs = process_images_rgb(f, "head_camera")
            left_imgs = process_images_rgb(f, "left_camera")
            right_imgs = process_images_rgb(f, "right_camera")

        T = len(states)
        total_frames += T
        task_index = task_to_index[all_instructions[ep_idx]]

        # Compute chunk
        chunk_id = ep_idx // CHUNK_SIZE
        data_chunk_dir = output_path / "data" / f"chunk-{chunk_id:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)

        video_base_dir = output_path / "videos" / f"chunk-{chunk_id:03d}"

        # Save videos for each camera
        for cam_name, frames in zip(CAMERA_NAMES, [head_imgs, left_imgs, right_imgs]):
            video_key = CAMERA_KEYS[cam_name]
            video_dir = video_base_dir / video_key
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
            write_video(frames, video_path, fps=FPS)

        # Save parquet (no images!)
        save_episode_parquet(ep_idx, states, actions, task_index, data_chunk_dir)

        # Append to episodes.jsonl
        with open(episode_path, "a") as f_ep:
            json.dump({
                "episode_index": ep_idx,
                "tasks": index_to_task[task_index],
                "length": T
            }, f_ep)
            f_ep.write("\n")
    # info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": len(CAMERA_NAMES)*total_episodes,
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": float(FPS),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.head_view": {
                "dtype": "video",
                "shape": [
                    256,
                    256,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
                "video_info": {
                    "video.fps": 20.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.left_wrist_view": {
                "dtype": "video",
                "shape": [
                    256,
                    256,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
                "video_info": {
                    "video.fps": 20.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.right_wrist_view": {
                "dtype": "video",
                "shape": [
                    256,
                    256,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
                "video_info": {
                    "video.fps": 20.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [STATE_DIM],
                "names": JOINT_NAMES,
            },
            "action": {
                "dtype": "float64",
                "shape": [ACTION_DIM],
                "names": JOINT_NAMES,
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        }
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json
    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 6},
            "left_gripper": {"start": 6, "end": 7},
            "right_arm": {"start": 7, "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "action": {
            "left_arm": {"start": 0, "end": 6},
            "left_gripper": {"start": 6, "end": 7},
            "right_arm": {"start": 7, "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "video": {
            "head_view": {"original_key": CAMERA_KEYS["head_camera"]},
            "left_wrist_view": {"original_key": CAMERA_KEYS["left_camera"]},
            "right_wrist_view": {"original_key": CAMERA_KEYS["right_camera"]},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        }
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)


    print(f"\n✅ Conversion complete!")
    print(f"   Output dir: {output_path}")
    print(f"   Episodes: {total_episodes}, Frames: {total_frames}, Tasks: {len(task_to_index)}")
    print(f"   Chunks: {(total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to RobotWin2 dataset root")
    parser.add_argument("--output_dir", required=True, help="Output LeRobot v2 dataset path")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)