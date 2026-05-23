"""
LIBERO environment

This file wraps the original LIBERO as a Gymnasium environment,
and registers it so that it can be instantiated via gym.make(...) and work
using our distributed evaluation.
"""

import math
import os
import json
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero import benchmark, get_default_path_dict


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
import numpy as np
import random
def clean_text(text):
    start_list = ['scene'] #小写，取后一个
    end_list = [str(i) for i in range(10)]#小写，取数字前一个
    """
    清洗文本，保留 start_list 与 end_list 之间的内容（不含边界词），
    去除尾部的数字，并去除多余空格。

    参数:
        text (str): 输入的原始字符串，如 "LIVING ROOM SCENE2 put apple view 0 0"
        start_list (list of str): 起始关键词列表
        end_list (list of str): 结束关键词列表

    返回:
        str: 清洗后的字符串，如 "put apple view"
    """
    if not text.strip():
        return ""

    words = text.split(' ')
    start_idx = 0
    end_idx = len(words)

    # 查找起始位置（第一个匹配 start_list 的词）
    for i, word in enumerate(words):
        if any([sw in word.lower() for sw in start_list]):
            start_idx = i+1
            break

    # 从起始位置之后查找结束位置
    for i in range(start_idx, len(words)):
        if any([ew in words[i].lower() for ew in end_list]):
            if words[i].isdigit():
                end_idx = i-1
            else:
                end_idx = i
            break
    # 提取中间部分
    segment = words[start_idx:end_idx]

    # 如果 segment 为空，则返回空
    if not segment:
        return ""

    # 重新组合并去除多余空格（其实 split+join 已经处理了）
    cleaned = ' '.join(segment)
    return cleaned.strip()
def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


class LiberoPlusEnv(gym.Env):
    """LanguageTable env."""

    def __init__(self, task_bddl_file: str, task_description: str, task_suite_name: str, task_category: str):
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        self._task_description = clean_text(task_description)
        print(f'task description:{self._task_description}')
        self._task_category = task_category
        self._task_suite_name = task_suite_name
        # Convert Gym action space to Gymnasium.
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                "video.wrist_image": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.roll": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.pitch": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.yaw": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.gripper": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.gripper": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def close(self):
        self._env.close()

    def _process_observation(self, obs):
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        new_obs = {
            "video.image": obs["agentview_image"][::-1, ::-1],
            "video.wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.x": [xyz[0]],
            "state.y": [xyz[1]],
            "state.z": [xyz[2]],
            "state.roll": [rpy[0]],
            "state.pitch": [rpy[1]],
            "state.yaw": [rpy[2]],
            "state.gripper": gripper,
            "annotation.human.action.task_description": self._task_description,
        }
        return new_obs

    def reset(self, seed=None, options=None):
        self.seed(seed)
        observation = self._env.reset()
        observation = self._process_observation(observation)
        info = {"success": self._env.check_success()}
        benchmark_info = {
            "task_suite_name": self._task_suite_name,
            "task_category": self._task_category,
        }
        info['benchmark_info'] = benchmark_info
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                action["action.gripper"],
            ],
            axis=0,
        )
        # action_vector = normalize_gripper_action(action_vector)
        # action_vector = invert_gripper_action(action_vector)
        action_vector[...,-1] = np.clip(action_vector[...,-1], -1, 1)
        observation, reward, done, info = self._env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = self._env.check_success()
        benchmark_info = {
            "task_suite_name": self._task_suite_name,
            "task_category": self._task_category,
        }
        info['benchmark_info'] = benchmark_info
        truncated = False
        return observation, reward, done, truncated, info
    def get_sim_state(self):
        return self._env.get_sim_state()

    def set_state(self, mujoco_state):
        self._env.set_state(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self._env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self._env.seed(seed)

    def set_init_state(self, init_state):
        return self._env.set_init_state(init_state)
    def get_env(self):
        return self._env

def register_liberoplus_envs():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_classification_file = os.path.join(get_default_path_dict()['benchmark_root'],'benchmark/task_classification.json')
    task_id_to_category = {}
    task_suite_names = [
        "libero_10",
        "libero_spatial",
        "libero_object",
        "libero_goal",
        # "libero_90", #libero_90 is not supported in LiberoPlus
    ]
    if os.path.exists(task_classification_file):
        with open(task_classification_file) as f:
            task_classification = json.load(f)
    for task_suite_name in task_suite_names:
        for task_id, task_cls in enumerate(task_classification[task_suite_name]):
            task_id_to_category[task_id] = task_cls['category']
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            task_name = task.name
            task_description = task.language
            task_category = task_id_to_category[task_id]
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            register(
                id=f"liberoplus_sim/{task_name}",
                entry_point="gr00t.eval.sim.LIBEROPLUS.liberoplus_env:LiberoPlusEnv",
                kwargs={
                    "task_bddl_file": task_bddl_file,
                    "task_description": task_description,
                    "task_suite_name": task_suite_name,
                    "task_category": task_category,
                },
            )


if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    for key in ["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"]:
        for task_name in benchmark_dict[key]().get_task_names():
            print(f"- {key}/{task_name}")

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print(
        f"[info] retrieving task {task_id} from suite {task_suite_name}, the "
        + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}"
    )

    # step over the environment
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(
        task_id
    )  # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])

    dummy_action = [0.0] * 7
    for step in range(10):
        obs, reward, done, info = env.step(dummy_action)
        print("step", step, "obs", obs.keys())
    env.close()
