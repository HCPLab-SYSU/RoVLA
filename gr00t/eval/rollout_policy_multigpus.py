import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time
from typing import Any
import os
import csv
import torch
import json
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Queue, Lock
import threading
import signal
import sys
import yaml
SEED=42
# 配置：每个任务使用的 GPU 比例（例如 0.25 表示 1 GPU 可跑 4 个实例）
PER_GPU_RATIO = 1
# PER_GPU_RATIO = 0.125

# 使用threading.local()确保每个进程中的每个线程都有独立的状态
_local = threading.local()

# 全局变量用于跟踪进程
running_processes = []

# 全局中断标志
interrupted = False

def signal_handler(signum, frame):
    """处理中断信号"""
    global interrupted
    print(f"\nReceived signal {signum}. Setting interrupt flag...")
    interrupted = True
    # 终止所有正在运行的进程
    for p in running_processes:
        if p.is_alive():
            p.terminate()
    for p in running_processes:
        p.join(timeout=1)
    sys.exit(0)


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        return gym.make(env_name, enable_render=True)

    return env_fn


def get_groot_locomanip_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
        from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
        from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
        import robocasa  # noqa: F401

        gym_env = gym.make(
            env_name,
            onscreen=False,
            offscreen=True,
            enable_waist=True,
            randomize_cameras=False,
            camera_names=[
                "robot0_oak_egoview",
                "robot0_rs_tppview",
            ],
        )
        wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
        gym_env = WholeBodyControlWrapper(gym_env, wbc_config)
        return gym_env

    return env_fn


def _ensure_local_attrs():
    """确保_local对象有需要的属性"""
    if not hasattr(_local, 'libero_registered'):
        _local.libero_registered = False
    if not hasattr(_local, 'liberoplus_registered'):
        _local.liberoplus_registered = False
    if not hasattr(_local, 'simpler_registered'):
        _local.simpler_registered = False
    if not hasattr(_local, 'behavior_registered'):
        _local.behavior_registered = False
    if not hasattr(_local, 'robotwin_registered'):
        _local.robotwin_registered = False


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        _ensure_local_attrs()
        if not _local.simpler_registered:
            from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
            register_simpler_envs()
            _local.simpler_registered = True
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        _ensure_local_attrs()
        if not _local.libero_registered:
            from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
            register_libero_envs()
            _local.libero_registered = True
        return gym.make(env_name)

    return env_fn

def get_liberoplus_env_fn(
    env_name: str,
):
    def env_fn():
        _ensure_local_attrs()
        if not _local.liberoplus_registered:
            from gr00t.eval.sim.LIBEROPLUS.liberoplus_env import register_liberoplus_envs
            register_liberoplus_envs()
            _local.liberoplus_registered = True
        return gym.make(env_name)

    return env_fn

def get_behavior_env_fn(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
):
    def env_fn():
        _ensure_local_attrs()
        if not _local.behavior_registered:
            from gr00t.eval.sim.BEHAVIOR.behavior_env import register_behavior_envs
            register_behavior_envs()
            _local.behavior_registered = True
        return gym.make(env_name, env_idx=env_idx, total_n_envs=total_n_envs)

    return env_fn

def get_robotwin_env_fn(
    env_name: str,
):
    def env_fn():
        _ensure_local_attrs()
        if not _local.robotwin_registered:
            from gr00t.eval.sim.RoboTwin.robotwin_env import register_robotwin_envs
            register_robotwin_envs()
            _local.robotwin_registered = True
        return gym.make(env_name)

    return env_fn
def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (
        EmbodimentTag.GR1,
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
    ):
        env_fn = get_robocasa_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.UNITREE_G1,):
        env_fn = get_groot_locomanip_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.OXE_GOOGLE, EmbodimentTag.OXE_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)
    elif env_embodiment in (EmbodimentTag.LIBEROPLUS_PANDA,):
        env_fn = get_liberoplus_env_fn(env_name)
    elif env_embodiment in (EmbodimentTag.BEHAVIOR_R1_PRO,):
        env_fn = get_behavior_env_fn(env_name, env_idx, total_n_envs)
    elif env_embodiment in (EmbodimentTag.ROBOTWIN_ALOHA_AGILEX,):
        env_fn = get_robotwin_env_fn(env_name)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """
    if env_name.startswith('robotwin'):
        file_dir = Path(__file__).parent.resolve()
        step_lim_path = os.path.join(str(file_dir), 'sim/RoboTwin/_eval_step_limit.yml')
        with open(step_lim_path, "r", encoding="utf-8") as f:
            step_lim_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        step_lim = step_lim_dict['_'.join(env_name.split('/')[-1].split('_')[2:])]
        wrapper_configs.video.max_episode_steps = step_lim
        wrapper_configs.multistep.max_episode_steps = step_lim
    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy_fn: Function that creates a policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]
    try:
        if n_envs == 1:
            env = gym.vector.SyncVectorEnv(env_fns)
        else:
            env = gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )
    except Exception as e:
        print(e)
        print('gym.vector.SyncVectorEnv error')
        import traceback
        traceback.print_exc()
    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset
    env_speacial_id = sum([ord(c) for c in env_name])  #简单的hash方法
    observations, _ = env.reset(seed=env_speacial_id+SEED)
    policy.reset()
    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    
    # 检查全局中断标志
    global interrupted
    while completed_episodes < n_episodes and not interrupted:
        actions, _ = policy.get_action(observations)
        try:
            next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        except Exception as e:
            print(e)
            print('env.step error')
            import traceback
            traceback.print_exc()
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "benchmark_info" in env_infos:
                    episode_infos['benchmark_info'].append(env_infos["benchmark_info"][env_idx][0])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs

    pbar.close()
    env.close()

    if interrupted:
        print("Rollout was interrupted. Exiting...")
        return env_name, [], {}

    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}
    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=0,
            )
        )
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str = None,
):
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)
    env_name_flatten = env_name.replace("/", "_")
    if video_dir:
        # 使用指定的视频目录
        video_dir = os.path.join(video_dir, f"{env_name_flatten}_ac{n_action_steps}")
    elif model_path:
        video_dir = (
            os.path.join(model_path, f"eval/sim_eval_videos_ac{n_action_steps}")
        )
    else:
        video_dir = f"/tmp/sim_eval_videos_{env_name_flatten}_ac{n_action_steps}"
    if env_name.startswith("sim_behavior_r1_pro"):
        # BEHAVIOR sim will crash if decord is imported in video_utils.py
        video_dir = None
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


def get_benchmark_env_names(benchmark_name: str):
    """Get all environment names for a given benchmark."""
    # 使用局部状态确保每个进程只注册一次
    _ensure_local_attrs()
    
    if benchmark_name.startswith("liberoplus"):
        # Handle liberoplus benchmarks
        if not _local.liberoplus_registered:
            from gr00t.eval.sim.LIBEROPLUS.liberoplus_env import register_liberoplus_envs
            register_liberoplus_envs()
            _local.liberoplus_registered = True
        
        parts = benchmark_name.split("/")
        if len(parts) == 1:
            # Just "liberoplus" - run all task suites
            task_suites = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
        elif len(parts) == 2:
            # "liberoplus/suite_name" - run specific suite
            suite_map = {
                "all": ["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                "spatial": ["libero_spatial"],
                "goal": ["libero_goal"],
                "object": ["libero_object"],
                "10": ["libero_10"],
                "long": ["libero_10"]
            }
            suite_name = parts[1]
            task_suites = suite_map.get(suite_name, [suite_name])
        else:
            raise ValueError(f"Invalid benchmark name format: {benchmark_name}")
            
        # Get environment names for each task suite
        env_names = []
        for suite in task_suites:
            # We need to actually get the registered env names
            benchmark_dict = __import__('libero.libero.benchmark', fromlist=['benchmark']).get_benchmark_dict()
            task_suite = benchmark_dict[suite]()
            for task_id in range(task_suite.get_num_tasks()):
                task = task_suite.get_task(task_id)
                task_name = task.name
                env_names.append(f"liberoplus_sim/{task_name}")
                
        return env_names
    elif benchmark_name.startswith("libero"):
        # Handle libero benchmarks
        if not _local.libero_registered:
            from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
            register_libero_envs()
            _local.libero_registered = True
        
        parts = benchmark_name.split("/")
        if len(parts) == 1:
            # Just "libero" - run all task suites
            task_suites = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
        elif len(parts) == 2:
            # "libero/suite_name" - run specific suite
            suite_map = {
                "all": ["libero_10", "libero_spatial", "libero_object", "libero_goal"],
                "spatial": ["libero_spatial"],
                "goal": ["libero_goal"],
                "object": ["libero_object"],
                "10": ["libero_10"],
                "long": ["libero_10"]
            }
            suite_name = parts[1]
            task_suites = suite_map.get(suite_name, [suite_name])
        else:
            raise ValueError(f"Invalid benchmark name format: {benchmark_name}")
            
        # Get environment names for each task suite
        env_names = []
        for suite in task_suites:
            # We need to actually get the registered env names
            benchmark_dict = __import__('libero.libero.benchmark', fromlist=['benchmark']).get_benchmark_dict()
            task_suite = benchmark_dict[suite]()
            for task_id in range(task_suite.get_num_tasks()):
                task = task_suite.get_task(task_id)
                task_name = task.name
                env_names.append(f"libero_sim/{task_name}")
                
        return env_names
    elif benchmark_name.startswith("robocasa") and "tabletop" in benchmark_name:
        # robocasa-gr1-tabletop-tasks
        # List all robocasa environments
        env_names = [
            "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
            "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
        ]
        return env_names
    elif benchmark_name.startswith("robocasa_panda_omron"):
        # List all robocasa environments
        env_names = [
            "robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env",
            "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env", 
            "robocasa_panda_omron/CoffeePressButton_PandaOmron_Env",
            "robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env",
            "robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env",
            "robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env",
            "robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env",
            "robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
            "robocasa_panda_omron/CloseDrawer_PandaOmron_Env",
            "robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env",
            "robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env",
            "robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env",
            "robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env",
            "robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env",
            "robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env",
            "robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env",
            "robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env",
            "robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env",
            "robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env",
            "robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env",
            "robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env",
            "robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env",
            "robocasa_panda_omron/TurnOnStove_PandaOmron_Env",
            "robocasa_panda_omron/TurnOffStove_PandaOmron_Env"
        ]
        return [f"robocasa_panda_omron/{name}" for name in env_names]
    elif benchmark_name.startswith("robotwin"):
        from gr00t.eval.sim.RoboTwin.all_task_names import robotwin_all_task_names
        if 'demo_clean' in benchmark_name:
            return [f"robotwin_sim/demo_clean_{task_name}" for task_name in robotwin_all_task_names]
        elif 'demo_randomized' in benchmark_name:
            return [f"robotwin_sim/demo_randomized_{task_name}" for task_name in robotwin_all_task_names]
        elif 'all' in benchmark_name:
            return [f"robotwin_sim/demo_clean_{task_name}" for task_name in robotwin_all_task_names] + [f"robotwin_sim/demo_randomized_{task_name}" for task_name in robotwin_all_task_names]
        else:
            raise ValueError(f"Unknown benchmark name: {benchmark_name}")
    else:
        # For single environment
        return [benchmark_name]
def robotwin_save_aggregated_results_to_csv(results, output_path: str):
    """
    Save aggregated results for RobotTwin benchmark to CSV.
    
    Expected structure of `results`:
        {
            "env_0": (..., [success1, success2, ...], [info1, info2, ...]),
            ...
        }
    Each info dict must contain: info['benchmark_info'] = {
        "task_name": str,
        "task_config": str  # either "demo_clean" or "demo_randomized"
    }
    """
    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)

    # Fixed task configs we care about
    TASK_CONFIGS = ["demo_clean", "demo_randomized"]
    
    # Aggregation dicts
    success_count = {}  # success_count[task_name][task_config] = int
    total_count = {}    # total_count[task_name][task_config] = int

    # Initialize 'ALL' entry
    success_count['ALL'] = {config: 0 for config in TASK_CONFIGS}
    total_count['ALL'] = {config: 0 for config in TASK_CONFIGS}

    # Process each environment result
    for env_name, (_, successes, infos) in results.items():
        if not successes:
            continue  # skip empty

        # All episodes in one env share the same benchmark_info
        task_config = '_'.join(env_name.split('/')[-1].split('_')[:2])
        task_name = '_'.join(env_name.split('/')[-1].split('_')[2:])
        

        if task_config not in TASK_CONFIGS:
            raise ValueError(f"Unexpected task_config: {task_config}. Expected one of {TASK_CONFIGS}")

        # Initialize task_name entry if not exists
        if task_name not in success_count:
            success_count[task_name] = {config: 0 for config in TASK_CONFIGS}
            total_count[task_name] = {config: 0 for config in TASK_CONFIGS}

        num_success = sum(successes)
        num_total = len(successes)

        # Update per-task counts
        success_count[task_name][task_config] += num_success
        total_count[task_name][task_config] += num_total

        # Update ALL
        success_count['ALL'][task_config] += num_success
        total_count['ALL'][task_config] += num_total

    # Compute success rates (%)
    success_rates = {}
    for task_name in success_count:
        success_rates[task_name] = {}
        for config in TASK_CONFIGS:
            total = total_count[task_name][config]
            if total > 0:
                rate = (success_count[task_name][config] / total) * 100
            else:
                rate = 0.0
            success_rates[task_name][config] = rate

    # Get all unique task names (excluding 'ALL'), sorted for consistency
    task_names = sorted([name for name in success_count.keys() if name != 'ALL'])
    task_names.append('ALL')  # Append ALL at the end

    # Headers
    headers = ["Task Name"] + TASK_CONFIGS + ["Average"]

    # Write success rate CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for task in task_names:
            row = [task]
            rates = [success_rates[task][config] for config in TASK_CONFIGS]
            row.extend([f"{rate:.1f}" for rate in rates])
            
            # Compute average across the two configs (only if both have data)
            valid_rates = []
            valid_totals = []
            for config in TASK_CONFIGS:
                total = total_count[task][config]
                if total > 0:
                    valid_rates.append(success_rates[task][config])
                    valid_totals.append(total)
            
            if valid_rates:
                # Weighted average by episode count
                weighted_sum = sum(rate * count for rate, count in zip(valid_rates, valid_totals))
                total_episodes = sum(valid_totals)
                avg_rate = weighted_sum / total_episodes
            else:
                avg_rate = 0.0

            row.append(f"{avg_rate:.1f}")
            writer.writerow(row)

    print(f"RobotTwin aggregated success rates saved to {output_path}")

    # Write count CSV
    count_output_path = output_path.replace('.csv', '_count.csv')
    with open(count_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for task in task_names:
            row = [task]
            counts = [total_count[task][config] for config in TASK_CONFIGS]
            row.extend(counts)
            
            # Total count across configs
            total_all = sum(counts)
            row.append(total_all)
            writer.writerow(row)

    print(f"RobotTwin aggregated counts saved to {count_output_path}")

def liberoplus_save_aggregated_results_to_csv(results, output_path):
    """Save aggregated results to CSV file."""
    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)
    # Define category mapping (original -> CSV column names)
    category_mapping = {
        "Background Textures": "Background",
        "Camera Viewpoints": "Camera",
        "Language Instructions": "Language",
        "Light Conditions": "Light",
        "Objects Layout": "Layout",
        "Robot Initial States": "Robot",
        "Sensor Noise": "Noise"
    }
    
    # Define task name mapping with fixed order
    task_name_mapping = {
        "libero_spatial": "spatial",
        "libero_goal": "Goal",
        "libero_10": "long",
        "libero_object": "object"
    }
    
    # Initialize data structures for aggregation
    success_count = {}  # success_count[task_suite][category] = number of successes
    total_count = {}    # total_count[task_suite][category] = total count
    
    # Initialize 'all' task suite
    success_count['ALL'] = {}
    total_count['ALL'] = {}
    
    # Process each environment result
    for env_name, (_, successes, infos) in results.items():
        # Extract task suite and category information from env info
        benchmark_info = infos["benchmark_info"][0]  # Take first element as they're all the same
        task_suite = task_name_mapping[benchmark_info["task_suite_name"]]
        category = category_mapping[benchmark_info["task_category"]]  # Fix: use category_mapping instead of task_name_mapping
        
        # Initialize nested dictionaries if needed
        if task_suite not in success_count:
            success_count[task_suite] = {}
            total_count[task_suite] = {}
        
        # Update counts for specific task_suite-category pair
        if category not in success_count[task_suite]:
            success_count[task_suite][category] = 0
            total_count[task_suite][category] = 0
        
        success_count[task_suite][category] += sum(successes)
        total_count[task_suite][category] += len(successes)
        
        # Update 'all' task suite
        if category not in success_count['ALL']:
            success_count['ALL'][category] = 0
            total_count['ALL'][category] = 0
        
        success_count['ALL'][category] += sum(successes)
        total_count['ALL'][category] += len(successes)
    
    # Calculate success rates
    success_rates = {}
    for task_suite in success_count:
        success_rates[task_suite] = {}
        for category in success_count[task_suite]:
            if total_count[task_suite][category] > 0:
                success_rates[task_suite][category] = (success_count[task_suite][category] / total_count[task_suite][category]) * 100
            else:
                success_rates[task_suite][category] = 0.0
    
    # Define table headers
    headers = ["Type", "Background", "Camera", "Language", "Light", "Layout", "Robot", "Noise", "Total"]
    
    # Helper function to get value safely
    def get_rate_or_zero(task_suite, category):
        return success_rates.get(task_suite, {}).get(category, 0.0)
    
    def get_count_or_zero(task_suite, category):
        return total_count.get(task_suite, {}).get(category, 0)
    
    # Write results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Write each task suite result in the required order
        ordered_task_names = ["spatial", "Goal", "long", "object"]
        for task_display_name in ordered_task_names:
            if task_display_name in success_rates:
                # Calculate total success rate for this task suite
                total_successes = sum(success_count[task_display_name].values()) if task_display_name in success_count else 0
                total_samples = sum(total_count[task_display_name].values()) if task_display_name in total_count else 1
                total_rate = (total_successes / total_samples) * 100 if total_samples > 0 else 0.0
                
                writer.writerow([
                    task_display_name,
                    f"{get_rate_or_zero(task_display_name, 'Background'):.1f}",
                    f"{get_rate_or_zero(task_display_name, 'Camera'):.1f}",
                    f"{get_rate_or_zero(task_display_name, 'Language'):.1f}",
                    f"{get_rate_or_zero(task_display_name, 'Light'):.1f}",
                    f"{get_rate_or_zero(task_display_name, 'Layout'):.1f}",
                    f"{get_rate_or_zero(task_display_name, 'Robot'):.1f}",
                    f"{get_rate_or_zero(task_display_name, 'Noise'):.1f}",
                    f"{total_rate:.1f}"
                ])
            else:
                # If no data for this task type, show zeros
                writer.writerow([task_display_name, "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"])
        
        # Write overall average row
        total_successes_all = sum(success_count['ALL'].values())
        total_samples_all = sum(total_count['ALL'].values())
        total_avg_rate = (total_successes_all / total_samples_all) * 100 if total_samples_all > 0 else 0.0
        
        writer.writerow([
            "ALL",
            f"{get_rate_or_zero('ALL', 'Background'):.1f}",
            f"{get_rate_or_zero('ALL', 'Camera'):.1f}",
            f"{get_rate_or_zero('ALL', 'Language'):.1f}",
            f"{get_rate_or_zero('ALL', 'Light'):.1f}",
            f"{get_rate_or_zero('ALL', 'Layout'):.1f}",
            f"{get_rate_or_zero('ALL', 'Robot'):.1f}",
            f"{get_rate_or_zero('ALL', 'Noise'):.1f}",
            f"{total_avg_rate:.1f}"
        ])
    
    print(f"Aggregated results saved to {output_path}")
    
    # Write count results to CSV
    count_output_path = output_path.replace('.csv', '_count.csv')
    with open(count_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Write each task suite count result in the required order
        ordered_task_names = ["spatial", "Goal", "long", "object"]
        for task_display_name in ordered_task_names:
            if task_display_name in total_count:
                total_task_count = sum(total_count[task_display_name].values())
                
                writer.writerow([
                    task_display_name,
                    get_count_or_zero(task_display_name, 'Background'),
                    get_count_or_zero(task_display_name, 'Camera'),
                    get_count_or_zero(task_display_name, 'Language'),
                    get_count_or_zero(task_display_name, 'Light'),
                    get_count_or_zero(task_display_name, 'Layout'),
                    get_count_or_zero(task_display_name, 'Robot'),
                    get_count_or_zero(task_display_name, 'Noise'),
                    total_task_count
                ])
            else:
                # If no data for this task type, show zeros
                writer.writerow([task_display_name, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Write overall count row
        total_all_count = sum(total_count['ALL'].values())
        
        writer.writerow([
            "ALL",
            get_count_or_zero('ALL', 'Background'),
            get_count_or_zero('ALL', 'Camera'),
            get_count_or_zero('ALL', 'Language'),
            get_count_or_zero('ALL', 'Light'),
            get_count_or_zero('ALL', 'Layout'),
            get_count_or_zero('ALL', 'Robot'),
            get_count_or_zero('ALL', 'Noise'),
            total_all_count
        ])
    
    print(f"Aggregated count results saved to {count_output_path}")


def robocasa_save_aggregated_results_to_csv(results, output_path):
    """Save aggregated results for robocasa benchmark to CSV file."""
    
    # Collect data for each environment
    env_data = []
    total_successes = []
    total_counts = []
    
    for env_name, (_, successes, _) in results.items():
        success_rate = np.mean(successes) * 100
        total_successes.extend(successes)
        episode_count = len(successes)
        total_counts.append(episode_count)
        
        # Extract just the environment/task name part
        if "/" in env_name:
            env_display_name = env_name.split("/")[-1]
        else:
            env_display_name = env_name
            
        env_data.append({
            'env_name': env_display_name,
            'success_rate': success_rate,
            'num_episodes': episode_count
        })
    
    # Calculate overall success rate
    overall_success_rate = np.mean(total_successes) * 100 if total_successes else 0.0
    
    # Write results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Environment", "Success Rate (%)", "Num Episodes"])
        
        # Write each environment's results
        for data in env_data:
            writer.writerow([
                data['env_name'],
                f"{data['success_rate']:.1f}",
                data['num_episodes']
            ])
        
        # Write overall average
        writer.writerow([
            "Overall",
            f"{overall_success_rate:.1f}",
            len(total_successes)
        ])
    
    print(f"Robocasa aggregated results saved to {output_path}")


def run_single_env_evaluation(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    worker_id: int = 0,
    gpu_id: int = 0,
    video_dir: str = None,  # 新增参数
):
    """Run evaluation on a single environment with specified GPU."""
    print(f"[Worker {worker_id}] Evaluating environment: {env_name} on GPU {gpu_id}")
    
    # Set CUDA device for this process
    # torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # 因为只看到一个设备，就是0
    print(f"[Worker {worker_id}] Using GPU {gpu_id} for policy client...")
    
    global interrupted
    if interrupted:
        print(f"[Worker {worker_id}] Interrupted before starting {env_name}")
        return None
    
    results = run_gr00t_sim_policy(
        env_name=env_name,
        n_episodes=n_episodes,
        max_episode_steps=max_episode_steps,
        model_path=model_path,
        policy_client_host=policy_client_host,
        policy_client_port=policy_client_port,
        n_envs=n_envs,
        n_action_steps=n_action_steps,
        video_dir=video_dir,  # 传递视频目录参数
    )
    print(f"[Worker {worker_id}] Completed evaluation for {env_name}")
    return env_name, results


def run_spawn_evaluation_worker(
    queue_in: Queue,
    queue_out: Queue,
    lock: Lock,
    worker_id: int,
    gpu_id: int,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str,
    policy_client_host: str,
    policy_client_port: int,
    n_envs: int,
    n_action_steps: int,
    video_dir: str,
):
    """Worker process for spawn-based evaluation."""
    print(f"[Worker {worker_id}] Started on GPU {gpu_id}")
    
    global interrupted
    while not interrupted:
        try:
            # Get an environment name from the queue
            item = queue_in.get(timeout=0.1)
            if item is None:  # Poison pill to stop the worker
                break
            
            env_name = item
            print(f"[Worker {worker_id}] Processing environment: {env_name}")
            
            # Run the evaluation
            result = run_single_env_evaluation(
                env_name=env_name,
                n_episodes=n_episodes,
                max_episode_steps=max_episode_steps,
                model_path=model_path,
                policy_client_host=policy_client_host,
                policy_client_port=policy_client_port,
                n_envs=n_envs,
                n_action_steps=n_action_steps,
                worker_id=worker_id,
                gpu_id=gpu_id,
                video_dir=video_dir,
            )
            
            if result is not None:
                # Put the result in the output queue
                queue_out.put(result)
                print(f"[Worker {worker_id}] Finished {env_name}")
        except:
            # Timeout - continue loop
            continue
    
    print(f"[Worker {worker_id}] Shutting down")


def run_spawn_batch_evaluation(
    benchmark_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    experiment_path: str = "",  # Add experiment_path parameter
):
    """Run spawn-based evaluation on a batch of environments belonging to a benchmark."""
    
    # Get all environment names for the benchmark
    env_names = get_benchmark_env_names(benchmark_name)


    n = len(env_names)
    inter_step = 10
    inter_step = 1 if inter_step>=n else inter_step
    env_names_reordered = []
    for i in range(inter_step):
        env_names_reordered += env_names[i::inter_step]
    env_names = env_names_reordered
    # env_names = env_names[1:]
    # Use experiment_path for output directories if provided, otherwise fall back to model_path
    output_path = experiment_path if experiment_path else model_path
    
    # Create output directories
    if output_path:
        benchmark_output_dir = os.path.join(output_path, f'eval_{benchmark_name}')
        os.makedirs(benchmark_output_dir, exist_ok=True)
        video_dir = os.path.join(benchmark_output_dir, 'rollout_video')
        os.makedirs(video_dir, exist_ok=True)
        results_file = os.path.join(benchmark_output_dir, 'all_results.json')
    else:
        video_dir = None
        results_file = None
    
    # Load existing results if available
    all_results = {}
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing results from {results_file}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    for env_name in all_results:
        # all_results[env_name]: (env_name, [episode_success, ...], info_dict)
        if len(all_results[env_name][1]) != n_episodes:
            print(f"Warning: Existing results for {env_name} have {len(all_results[env_name][1])} episodes, expected {n_episodes}. Re-evaluating this environment.")
            del all_results[env_name]
    # Determine which environments still need to be evaluated
    remaining_env_names = [env_name for env_name in env_names if (env_name not in all_results or len(all_results[env_name][1]) != n_episodes)]
    print(f"Environments to evaluate: {len(remaining_env_names)} (out of {len(env_names)} total)")
    
    if not remaining_env_names:
        print("All environments have already been evaluated. Skipping evaluation.")
        return all_results
    
    # Determine GPU allocation
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No GPU found. Spawn evaluation requires GPU.")
    
    # Calculate number of processes per GPU based on PER_GPU_RATIO
    processes_per_gpu = int(1 / PER_GPU_RATIO)  # For PER_GPU_RATIO=0.25, this gives 4
    total_workers = int(n_gpu * processes_per_gpu)
    
    print(f"Detected {n_gpu} GPUs. Launching {total_workers} workers (each using {PER_GPU_RATIO} GPU).")
    print(f"Each GPU will run {processes_per_gpu} processes.")
    
    # Limit number of workers to the number of environments
    num_workers = min(total_workers, len(remaining_env_names))
    
    # Create queues for communication
    task_queue = Queue()
    result_queue = Queue()
    lock = Lock()
    
    # Add tasks to the queue
    for env_name in remaining_env_names:
        task_queue.put(env_name)
    
    # Add poison pills to signal workers to stop
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Create and start worker processes
    processes = []
    all_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if all_gpu_id is None:
        all_gpu_id = [i for i in range(n_gpu)]
    else:
        all_gpu_id = all_gpu_id.split(',')
    for i in range(num_workers):
        gpu_id = i % n_gpu  # Round-robin assignment of GPUs
        gpu_id = all_gpu_id[gpu_id]
        worker_id = i
        
        p = Process(
            target=run_spawn_evaluation_worker,
            args=(
                task_queue,
                result_queue,
                lock,
                worker_id,
                gpu_id,
                n_episodes,
                max_episode_steps,
                model_path,
                policy_client_host,
                policy_client_port,
                n_envs,
                n_action_steps,
                video_dir,
            )
        )
        p.start()
        processes.append(p)
        running_processes.append(p)  # Track for signal handling
    
    # Collect results
    completed_results = {}
    total_completed = 0
    
    global interrupted
    while total_completed < len(remaining_env_names) and not interrupted:
        try:
            env_name, results = result_queue.get(timeout=1)
            if results is not None:
                total_completed += 1
                completed_results[env_name] = results
                all_results[env_name] = results
                # Save results after each environment evaluation
                if results_file:
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                if total_completed % 100 == 0 or benchmark_name.startswith("robocasa") or benchmark_name.startswith("robotwin"):
                    csv_output_path = os.path.join(benchmark_output_dir, 'all_results.csv')
                    if benchmark_name.startswith("liberoplus"):
                        liberoplus_save_aggregated_results_to_csv(all_results, csv_output_path)
                    elif benchmark_name.startswith("robocasa"):
                        robocasa_save_aggregated_results_to_csv(all_results, csv_output_path)
                    elif benchmark_name.startswith("robotwin"):
                        robotwin_save_aggregated_results_to_csv(all_results, csv_output_path)
                print(f"Progress: {total_completed}/{len(remaining_env_names)}. Completed evaluation for {env_name}: success rate = {np.mean(results[1]):.2f};")
        except:
            # Continue waiting for results
            continue

    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join()
    
    return all_results


def run_batch_evaluation(
    benchmark_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    experiment_path: str = "",  # Add experiment_path parameter
):
    """Run evaluation on a batch of environments belonging to a benchmark."""
    
    # Get all environment names for the benchmark
    env_names = get_benchmark_env_names(benchmark_name)
    
    # Use experiment_path for output directories if provided, otherwise fall back to model_path
    output_path = experiment_path if experiment_path else model_path
    
    # Create output directories
    if output_path:
        benchmark_output_dir = os.path.join(output_path, f'eval_{benchmark_name}')
        os.makedirs(benchmark_output_dir, exist_ok=True)
        video_dir = os.path.join(benchmark_output_dir, 'rollout_video')
        os.makedirs(video_dir, exist_ok=True)
        results_file = os.path.join(benchmark_output_dir, 'all_results.json')
    else:
        video_dir = None
        results_file = None
    
    # Load existing results if available
    all_results = {}
    if results_file and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {results_file}")
    for env_name in all_results:
        # all_results[env_name]: (env_name, [episode_success, ...], info_dict)
        if len(all_results[env_name][1]) != n_episodes:
            print(f"Warning: Existing results for {env_name} have {len(all_results[env_name][1])} episodes, expected {n_episodes}. Re-evaluating this environment.")
            del all_results[env_name]
    # Determine which environments still need to be evaluated
    remaining_env_names = [env_name for env_name in env_names if (env_name not in all_results or len(all_results[env_name][1]) != n_episodes)]
    print(f"Environments to evaluate: {len(remaining_env_names)} (out of {len(env_names)} total)")
    
    # Run evaluation for each remaining environment
    global interrupted
    for i, env_name in enumerate(remaining_env_names):
        if interrupted:
            print("\nEvaluation interrupted by user. Stopping...")
            break
            
        print(f"\n[{i+1}/{len(remaining_env_names)}] Evaluating environment: {env_name}")
        results = run_gr00t_sim_policy(
            env_name=env_name,
            n_episodes=n_episodes,
            max_episode_steps=max_episode_steps,
            model_path=model_path,
            policy_client_host=policy_client_host,
            policy_client_port=policy_client_port,
            n_envs=n_envs,
            n_action_steps=n_action_steps,
            video_dir=video_dir,
        )
        all_results[env_name] = results
        print(f"Results for {env_name}: success rate = {np.mean(results[1]):.2f}")
        
        # Save results after each environment evaluation
        if results_file:
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--experinment_path",
        type=str,
        default="",
    )
    parser.add_argument("--policy_client_host", type=str, default="")
    parser.add_argument("--policy_client_port", type=int, default=None)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="",
        help="Benchmark name for batch evaluation (e.g., liberoplus/all, liberoplus/spatial)"
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument(
        "--use_spawn",
        action="store_true",
        help="Use spawn for multi-process evaluation across multiple GPUs"
    )

    args = parser.parse_args()

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy_client_host and --policy_client_port, and set --model_path ""\n'
        '  - To use model path: set --model_path, and set --policy_client_host "" (and leave --policy_client_port unset)'
    )

    # Determine whether to run single env or batch evaluation
    if args.benchmark_name:
        # Batch evaluation mode
        if args.use_spawn:
            results = run_spawn_batch_evaluation(
                benchmark_name=args.benchmark_name,
                n_episodes=args.n_episodes,
                max_episode_steps=args.max_episode_steps,
                model_path=args.model_path,
                policy_client_host=args.policy_client_host,
                policy_client_port=args.policy_client_port,
                n_envs=args.n_envs,
                n_action_steps=args.n_action_steps,
                experiment_path=args.experinment_path,  # Pass experiment_path
            )
        else:
            results = run_batch_evaluation(
                benchmark_name=args.benchmark_name,
                n_episodes=args.n_episodes,
                max_episode_steps=args.max_episode_steps,
                model_path=args.model_path,
                policy_client_host=args.policy_client_host,
                policy_client_port=args.policy_client_port,
                n_envs=args.n_envs,
                n_action_steps=args.n_action_steps,
                experiment_path=args.experinment_path,  # Pass experiment_path
            )
        
        print("\n" + "="*50)
        print("BATCH EVALUATION RESULTS")
        print("="*50)
        
        # Print individual results
        for env_name, (_, successes, _) in results.items():
            success_rate = np.mean(successes) * 100
            print(f"{env_name}: {success_rate:.2f}% ({sum(successes)}/{len(successes)})")
        
        # Save aggregated results to CSV if either experiment_path or model_path is provided
        output_path = args.experinment_path if args.experinment_path else args.model_path
        if output_path:
            benchmark_output_dir = os.path.join(output_path, f'eval_{args.benchmark_name}')
            os.makedirs(benchmark_output_dir, exist_ok=True)
            csv_output_path = os.path.join(benchmark_output_dir, 'all_results.csv')
            # If it's a liberoplus benchmark, also show aggregated results
            if args.benchmark_name.startswith("liberoplus"):
                liberoplus_save_aggregated_results_to_csv(results, csv_output_path)
            if args.benchmark_name.startswith("robocasa"):
                robocasa_save_aggregated_results_to_csv(results, csv_output_path)
                
    else:
        # Single environment evaluation mode (existing behavior)
        results = run_gr00t_sim_policy(
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            max_episode_steps=args.max_episode_steps,
            model_path=args.model_path,
            policy_client_host=args.policy_client_host,
            policy_client_port=args.policy_client_port,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
        )
        print("results: ", results)
        print("success rate: ", np.mean(results[1]))



