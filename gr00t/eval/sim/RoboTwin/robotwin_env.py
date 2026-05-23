"""
RobotWin environment wrapped as a Gymnasium environment for GR00TN1.6.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import importlib
import os
import sys
from typing import Dict, Any, Tuple, Optional
import yaml
import cv2
# Assume the original envs are in ./envs/
"""
Since the RobotWin 2.0 codebase makes extensive use of relative paths, 
it cannot be directly utilized as a standalone library; consequently, 
our code currently does not support testing with RobotWin 2.0.
If you wish to enable support, you can manually modify the `ROBOTWIN_PACKAGE_PATH` 
and adjust the paths within RoboTwin in response to any error messages.
"""
ROBOTWIN_PACKAGE_PATH = '/xxx/RoboTwin'
if ROBOTWIN_PACKAGE_PATH not in sys.path:
    sys.path.insert(0, ROBOTWIN_PACKAGE_PATH)

# from robotwin
from envs import CONFIGS_PATH
from description.utils.generate_episode_instructions import generate_episode_descriptions
from envs.utils.create_actor import UnStableError
ENV_MODULE_BASE = os.path.join(ROBOTWIN_PACKAGE_PATH,"envs")

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    # if not os.path.isabs(embodiment_args['urdf_path']):
    #     embodiment_args['urdf_path'] = os.path.join(ROBOTWIN_PACKAGE_PATH, embodiment_args['urdf_path'])
    # if not os.path.isabs(embodiment_args['srdf_path']):
    #     embodiment_args['srdf_path'] = os.path.join(ROBOTWIN_PACKAGE_PATH, embodiment_args['srdf_path'])
    return embodiment_args
class RoboTwinEnv(gym.Env):
    """
    Gymnasium wrapper for RobotWin tasks.
    
    Observation space matches the expected input of GR00T N1.6 policy.
    Action space is joint positions + gripper states (14-dim).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task_name: str,
        seed: Optional[int] = None,
        instruction: Optional[str] = None,
        task_config: str = 'demo_clean',
        instruction_type: str = 'unseen',
        test_num: int = 100,
        step_lim: int = 1700,
    ):
        super().__init__()
        self._task_name = task_name
        self._target_size = (256, 256)
        self._instruction = instruction
        self._instruction_type = instruction_type
        self._args = self.load_args(task_name, task_config)
        # Load the original task environment
        try:
            env_module = importlib.import_module(f"envs.{task_name}")
            env_class = getattr(env_module, task_name)
            self._env = env_class()
        except Exception as e:
            raise ValueError(f"Failed to load RobotWin task '{task_name}': {e}")

        # Determine action dimension from embodiment config
        # We assume 7 arm per arm + 1 gripper per arm = 16? But your code uses 14.
        # From your eval: left_arm (6) + left_grip (1) + right_arm (6) + right_grip (1) = 14
        self._action_dim = 14  # adjust if needed based on actual robot

        # Define observation space
        H, W = self._target_size
        self.observation_space = spaces.Dict({
            "video.head_view": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "video.left_wrist_view": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "video.right_wrist_view": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "state.left_arm": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "state.left_gripper": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "state.right_arm": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "state.right_gripper": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "annotation.human.action.task_description": spaces.Text(max_length=512),
        })

        # Define action space: qpos (14-dim)
        self.action_space = spaces.Dict({
            "action.left_arm": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "action.left_gripper": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "action.right_arm": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "action.right_gripper": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
        # Internal state
        self._current_seed = seed
        self._episode_step = 0
        self._current_episode_num = 0
        self._test_num = test_num
        self.step_lim = step_lim
    def _process_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert raw RobotWin observation to GR00TN1.6 format."""
        # Extract RGBs
        obs_dict = raw_obs["observation"]
        head_rgb = self._process_image(obs_dict["head_camera"]["rgb"])
        left_rgb = self._process_image(obs_dict["left_camera"]["rgb"])
        right_rgb = self._process_image(obs_dict["right_camera"]["rgb"])

        # Extract state vector
        q = np.asarray(raw_obs["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        
        D = q.shape[0]
        arm = (D - 2) // 2
        
        if 2 * arm + 2 > D:
            raise ValueError(f"Cannot split state vector dim={D}. Please adjust slicing.")
        if 2 * arm + 2 != D:
            arm = 7

        left_arm  = q[0:arm]
        left_grip = q[arm:arm+1]
        right_arm = q[arm+1:arm+1+arm]
        right_grip = q[arm+1+arm:arm+1+arm+1]

        return {
            "video.head_view": head_rgb,
            "video.left_wrist_view": left_rgb,
            "video.right_wrist_view": right_rgb,
            "state.left_arm": left_arm.astype(np.float32),
            "state.left_gripper": left_grip.astype(np.float32),
            "state.right_arm": right_arm.astype(np.float32),
            "state.right_gripper": right_grip.astype(np.float32),
            "annotation.human.action.task_description": self._env.instruction,
        }

    def _process_image(self, img_input) -> np.ndarray:
        if isinstance(img_input, bytes):
            # 如果输入是字节类型，则先解码为图像数组
            img = cv2.imdecode(np.frombuffer(img_input, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(img_input, np.ndarray):
            # 如果输入已经是ndarray，则直接使用
            img = img_input
        else:
            raise ValueError("Unsupported image input type. Expected bytes or ndarray.")
        # 获取原始图像尺寸和目标尺寸
        old_size = img.shape[:2]  # (height, width)
        ratio = min(float(self._target_size[1])/old_size[0], float(self._target_size[0])/old_size[1])
        new_size = tuple([int(x*ratio) for x in old_size[::-1]])  # 注意宽高顺序
        
        # 等比例缩放图像
        img_resized = cv2.resize(img, (new_size[0], new_size[1]), interpolation=cv2.INTER_LINEAR)
        
        # 创建一个带有padding的新图像（底色为黑色）
        delta_w = self._target_size[0] - new_size[0]
        delta_h = self._target_size[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]  # 黑色边框
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img_padded

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self._current_episode_num>0:
            self._env.close_env(clear_cache=((self._current_episode_num + 1) % self.args['clear_cache_freq'] == 0))
        if seed is not None:
            self._current_seed = seed
        try_count = 0
        while True and try_count<1000:
            try_count += 1
            try:
                self._env.setup_demo(now_ep_num=self._current_episode_num, seed=self._current_seed, is_test=True, **self._args)
                episode_info = self._env.play_once()
                self._env.close_env()
                if self._env.plan_success and self._env.check_success():
                    break
                else:
                    self._current_seed += 1
                    continue
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                self._env.close_env()
                self._current_seed += 1
                continue
            except Exception as e:
                import traceback
                stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"Error in: {self._task_name}", stack_trace)
                print(" -------------")
                self._env.close_env()
                self._current_seed += 1
                print(f"error occurs {self._task_name}!")
                continue
        
        self._env.setup_demo(now_ep_num=self._current_episode_num, seed=self._current_seed, is_test=True, **self._args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(self._args["task_name"], episode_info_list, self._test_num)
        instruction = np.random.choice(results[0][self._instruction_type])
        self._instruction = instruction
        self._env.set_instruction(instruction=instruction)  # set language instruction

        # Get initial observation
        raw_obs = self._env.get_obs()
        obs = self._process_observation(raw_obs)

        self._episode_step = 0
        info = {
            "success": False
        }
        benchmark_info = {
            "task_name": self._task_name,
            "task_config": self._args["task_config"],
        }
        info['benchmark_info'] = benchmark_info
        self._current_seed += 1
        return obs, info

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action_vector = np.concatenate(
            [
                action["action.left_arm"],
                action["action.left_gripper"],
                action["action.right_arm"],
                action["action.right_gripper"],
            ],
            axis=0,
        ) #(14,)
        if action_vector.shape != (self._action_dim,):
            raise ValueError(f"Action must be shape ({self._action_dim},), got {action_vector.shape}")
        # Execute action
        self._env.take_action(action_vector, action_type="qpos")
        raw_obs = self._env.get_obs()
        obs = self._process_observation(raw_obs)

        # Check success
        success = self._env.eval_success

        self._episode_step += 1
        truncated = self._episode_step >= self.step_lim
        terminated = success or truncated

        reward = 1.0 if success else 0.0
        info = {
            "success": success,
        }
        benchmark_info = {
            "task_name": self._task_name,
            "task_config": self._args["task_config"],
        }
        info['benchmark_info'] = benchmark_info

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self._env, "close_env"):
            self._env.close_env()
        elif hasattr(self._env, "close"):
            self._env.close()
    def load_args(self, task_name, task_config):
        with open(os.path.join(ROBOTWIN_PACKAGE_PATH, f"task_config/{task_config}.yml"), "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        args['task_name'] = task_name
        args["task_config"] = task_config

        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            robot_file = os.path.join(ROBOTWIN_PACKAGE_PATH, robot_file.replace('./',''))
            if robot_file is None:
                raise "No embodiment files"
            return robot_file

        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise "embodiment items should be 1 or 3"

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
        args['eval_video_log'] = False
        args['save_path']=None
        args["eval_mode"] = True
        return args
# Optional: auto-register all RobotWin tasks if you have a task list
def register_robotwin_envs():
    """
    Register all RobotWin tasks as gym environments.
    """
    from gymnasium.envs.registration import register
    from gr00t.eval.sim.RoboTwin.all_task_names import robotwin_all_task_names
    with open(os.path.join(ROBOTWIN_PACKAGE_PATH, f"task_config/_eval_step_limit.yml"), "r", encoding="utf-8") as f:
        step_lim_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for task_name in robotwin_all_task_names:
        for task_config in ['demo_clean', 'demo_randomized']:
            register(
                id=f"robotwin_sim/{task_config}_{task_name}",
                entry_point="gr00t.eval.sim.RoboTwin.robotwin_env:RoboTwinEnv",
                kwargs={"task_name": task_name, 'task_config': task_config, 'step_lim': step_lim_dict[task_name]},
            )
def normalize_gripper_action(action, binarize=True):
    # Just normalize the last action to [-1,+1].
    action[..., -1] = (action[..., -1]>=0.5).astype(np.float32)
    return action