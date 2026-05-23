"""
PANDA Real-Robot GR00T Policy Evaluation Script (Franky + PyRealSense2 Version)
- QUATERNION FORMAT: Unified to xyzw [x, y, z, w] throughout the entire script.
- Uses franky-control for Robot and Gripper control.
- Uses pyrealsense2 for camera capture.
- Maintains Gymnasium API (reset/step interfaces).
- Added Keyboard Control: 's' to start/run, 'e' to stop/abort, 'q' to quit.
"""

# =============================================================================
# Imports
# =============================================================================
import sys
sys.path.insert(0, "/media/hcp/disk/projects/Isaac-GR00T-1d6")

import numpy as np
import time
import os
import cv2
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import threading
import tty
import termios
import select

# GR00T & Policy Imports
from gr00t.policy.server_client import PolicyClient
# Franky Control Library - Updated imports based on documentation
import franky
from franky import (
    Robot, Gripper, 
    CartesianMotion, 
    Affine, RobotPose, 
    ReferenceType,
)
from scipy.spatial.transform import Rotation

# RealSense Camera Library
import pyrealsense2 as rs

import imageio.v3 as iio

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class CameraConfig:
    """RealSense camera configuration."""
    width: int = 848
    height: int = 480
    fps: int = 30
    enable_depth: bool = False


@dataclass
class ControlConfig:
    """Control parameters and workspace bounds."""
    eef_x_min: float = -1.0
    eef_x_max: float = 1.0
    eef_y_min: float = -1.0
    eef_y_max: float = 1.0
    eef_z_min: float = -1.0
    eef_z_max: float = 1.0
    
    use_closed_loop: bool = False  # Franky handles its own closed loop internally
    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.05  # Radians
    
    gripper_max_width: float = 0.08
    gripper_speed: float = 0.05  # [m/s]
    gripper_force: float = 80.0  # [N]
    
    # Motion parameters
    velocity_rel: float = 0.05  # Relative velocity (0.0 - 1.0)
    acceleration_rel: float = 0.05  # Relative acceleration (0.0 - 1.0)
    jerk_rel: float = 0.05  # Relative jerk (0.0 - 1.0)
    relative_dynamics_factor: float=0.05

@dataclass
class InitialStateConfig:
    """Initial state configuration for environment reset."""
    # x: float = 0.5864516253870733
    # y: float = -0.02157497058695771
    # z: float = 0.47211122199859734
    # roll: float = 2.1779807567955665
    # pitch: float = -0.03807105668675128
    # yaw: float = 0.021796417599355956
    # x,y,z,roll,pitch,yaw = 0.5231996995314867, -0.008530103814094643, 0.44673143435016494, 3.109076281994777, 0.015278916788788122, 0.10931428189307169
    # x,y,z,roll,pitch,yaw = 0.5, 0., 0.5, 3, 0., 0.

    # for drawer
    x,y,z,roll,pitch,yaw = 0.53, 0, 0.5070390104309448, 3.1078813513213532, 0., 0.
    gripper_width: float = 0.08


@dataclass
class EvalConfig:
    """Main evaluation configuration."""
    policy_host: str = "localhost"
    policy_port: int = 5555
    lang_instruction: str = "Pick up the red block."
    experiment_path: str = "./experiments"
    record_video: bool = False
    max_step: int = 800
    
    robot_ip: str = "172.16.0.2"
    gripper_ip: str = "172.16.0.2"  # Usually same as robot IP
    
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)


# =============================================================================
# Helper Functions (xyzw NATIVE)
# =============================================================================

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """Add batch (B=1) and time (T=1) dimensions to numpy arrays in obs dict."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                obs[key] = val[np.newaxis, np.newaxis, ...]
            elif val.ndim == 3:
                obs[key] = val[np.newaxis, np.newaxis, ...]
            else:
                obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [[val]] if not isinstance(val, list) else [val]
    return obs


def quaternion_distance_xyzw(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Calculate angular distance between two quaternions in XYZW format.
    q = [x, y, z, w]
    """
    # Ensure normalized
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    
    dot = np.dot(q1, q2)
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angle = 2 * arccos(|dot|)
    angle = 2.0 * np.arccos(abs(dot))
    return angle


def euler_to_quaternion_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (RPY) to quaternion (XYZW format).
    
    参数:
        roll (float): 绕 X 轴的旋转角 (弧度)
        pitch (float): 绕 Y 轴的旋转角 (弧度)
        yaw (float): 绕 Z 轴的旋转角 (弧度)
        
    返回:
        np.ndarray: 四元数 [x, y, z, w]
    """
    # 'xyz' 对应原代码 axes='xyzs' (静态轴 X-Y-Z 顺序)
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    
    # as_xyzw() 返回 numpy.ndarray [x, y, z, w]
    return r.as_quat()


def quaternion_to_euler_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (XYZW) to Euler angles (RPY).
    
    参数:
        quat_xyzw (np.ndarray): 四元数 [x, y, z, w]
        
    返回:
        np.ndarray: 欧拉角 [roll, pitch, yaw] (弧度)
    """
    # from_quat 接受 [x, y, z, w] 格式
    r = Rotation.from_quat(quat_xyzw)
    
    # as_euler 返回 numpy.ndarray [roll, pitch, yaw]
    # 顺序 'xyz' 必须与 from_euler 时保持一致
    return r.as_euler('xyz')


# =============================================================================
# Keyboard Controller (Non-blocking)
# =============================================================================

class KeyboardController:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        self.enabled = False

    def enable(self):
        """Set terminal to cbreak mode for non-blocking input."""
        tty.setcbreak(self.fd)
        self.enabled = True

    def disable(self):
        """Restore terminal settings."""
        if self.enabled:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.enabled = False

    def get_key(self) -> Optional[str]:
        """
        Get a key press if available, otherwise return None.
        Non-blocking.
        """
        if not self.enabled:
            return None
        
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


# =============================================================================
# Gym-like Environment with Correct Franky Usage
# =============================================================================

class FrankaEnv:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.control_cfg = cfg.control
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Franky Robot Arm
        self.logger.info(f"Connecting to Franka robot arm at {cfg.robot_ip}...")
        self.robot = Robot(cfg.robot_ip)
        
        
        # Set dynamics factor for safety (lower = slower/safer)
        self.robot.velocity_rel = self.control_cfg.velocity_rel
        self.robot.acceleration_rel = self.control_cfg.acceleration_rel
        self.robot.jerk_rel = self.control_cfg.jerk_rel
        self.robot.relative_dynamics_factor = self.control_cfg.relative_dynamics_factor
        self.logger.info("Franka robot arm connected successfully.")
        self.logger.info(f"Robot dynamics: vel={self.robot.velocity_rel}, acc={self.robot.acceleration_rel}")
        
        # Initialize Franky Gripper
        self.logger.info(f"Connecting to Franka gripper at {cfg.gripper_ip}...")
        self.gripper = Gripper(cfg.gripper_ip)
        self.gripper_speed = self.control_cfg.gripper_speed
        self.logger.info("Franka gripper connected successfully.")
        
        # Initialize RealSense Camera
        self.camera_config = cfg.camera_config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(
            rs.stream.color, 
            self.camera_config.width, 
            self.camera_config.height, 
            rs.format.bgr8, 
            self.camera_config.fps
        )
        
        if self.camera_config.enable_depth:
            self.config.enable_stream(
                rs.stream.depth, 
                self.camera_config.width, 
                self.camera_config.height, 
                rs.format.z16, 
                self.camera_config.fps
            )
        
        self.pipeline.start(self.config)
        self.logger.info("RealSense camera started.")
        
        # Warm up camera
        for _ in range(10):
            self.pipeline.wait_for_frames()
        time.sleep(0.5)
        
        # Video Recording State
        self.record_video = cfg.record_video
        self.video_frames: List[np.ndarray] = []
        self.current_video_path: Optional[str] = None
        
        # Step counting
        self.max_step = cfg.max_step
        self.current_step = 0
        
        # Initial state
        self.initial_pos = np.array([
            cfg.initial_state.x,
            cfg.initial_state.y,
            cfg.initial_state.z
        ])
        self.initial_rpy = np.array([
            cfg.initial_state.roll,
            cfg.initial_state.pitch,
            cfg.initial_state.yaw
        ])
        self.initial_gripper_width = cfg.initial_state.gripper_width
        
        self.logger.info(f"FrankaEnv initialized with max_step={self.max_step}")
        self.logger.info(f"Initial Pos: {self.initial_pos}, Initial RPY: {self.initial_rpy}")

    def _get_camera_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from RealSense camera."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
            
            color_image = np.asanyarray(color_frame.get_data())
            # Convert BGR to RGB
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            return color_rgb
        except Exception as e:
            self.logger.warning(f"Error capturing camera frame: {e}")
            return None
    
    def _get_robot_state(self) -> Optional[Dict]:
        """Get current robot state from franky. Returns Quat in xyzw."""
        try:
            # Read current pose from robot
            current_pose = self.robot.current_cartesian_state.pose.end_effector_pose
            position = current_pose.translation
            # Rotation matrix to quaternion
            quat_xyzw = current_pose.quaternion
            
            # Get gripper width
            gripper_width = self.gripper.width
            
            return {
                "eef_pos": np.array(position),
                "eef_quat_xyzw": quat_xyzw,
                "gripper": np.array([gripper_width/2, gripper_width/2])
            }
        except Exception as e:
            self.logger.warning(f"Error getting robot state: {e}")
            return None
    
    def _setup_video_recording(self):
        if not self.record_video:
            return

        safe_instruction = '_'.join(self.cfg.lang_instruction.lower().split())
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        base_dir = os.path.join(
            self.cfg.experiment_path, "eval", "hcp_panda", 
            safe_instruction, timestamp
        )
        # os.makedirs(base_dir, exist_ok=True)
        
        self.current_video_path = os.path.join(base_dir, "evaluation.mp4")
        self.video_frames = []
        self.logger.info(f"Video recording enabled. Saving to: {self.current_video_path}")

    def _save_video(self):
        if not self.record_video or len(self.video_frames) == 0:
            return
        
        os.makedirs(os.path.dirname(self.current_video_path), exist_ok=True)
        self.logger.info(f"Saving video to: {self.current_video_path}")

        try:
            iio.imwrite(
                self.current_video_path, 
                self.video_frames, 
                codec='libx264', 
                fps=20.0
            )
            self.logger.info(f"Video saved successfully.")
            land_file = os.path.join(os.path.dirname(self.current_video_path), 'instruction.txt')
            with open(land_file, 'w') as f:
                f.write(self.cfg.lang_instruction)
        except Exception as e:
            self.logger.error(f"Failed to save video: {e}")
        
        self.video_frames = []
    
    def _clear_video(self):
        """Clear video frames without saving."""
        self.video_frames = []
        self.current_video_path = None

    def get_observation(self, timeout_sec=2.0) -> Optional[Dict]:
        """Get current observation. Quat returned is xyzw."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            image = self._get_camera_frame()
            state = self._get_robot_state()
            
            if image is not None and state is not None:
                obs = {
                    "eef_pos": state["eef_pos"].copy(),
                    "eef_quat": state["eef_quat_xyzw"].copy(),
                    "gripper": state["gripper"].copy(),
                    "wrist_image": image.copy(),
                    "lang": self.cfg.lang_instruction
                }
                
                if self.record_video:
                    self.video_frames.append(self.resize_and_pad_image(image.copy()))
                
                return obs
            
            time.sleep(0.01)
        
        self.logger.warning("Timeout waiting for observation.")
        return None
    
    def resize_and_pad_image(self, img: np.ndarray) -> np.ndarray:
        target_size = (256, 256)
        h, w = img.shape[:2]
        ratio = min(float(target_size[1])/h, float(target_size[0])/w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        top = (target_size[1] - new_h) // 2
        left = (target_size[0] - new_w) // 2
        
        canvas[top:top+new_h, left:left+new_w] = resized
        return canvas
    
    def _move_gripper(self, width: float, blocking: bool = True) -> bool:
        """Move gripper to specific width."""
        width = float(np.clip(width, 0.0, self.control_cfg.gripper_max_width))
        
        try:
            if blocking:
                success = self.gripper.move(width, self.gripper_speed)
                # success = self.gripper.grasp(width, self.gripper_speed, self.control_cfg.gripper_force, 0.01)
                return success
            else:
                # Non-blocking async move
                future = self.gripper.move_async(width, self.gripper_speed)
                # future = self.gripper.grasp_async(width, self.gripper_speed, self.control_cfg.gripper_force, 0.01)
                # Wait a short time but don't block forever
                if future.wait(2.0):
                    return future.get()
                return True  # Assume success if we don't wait
        except Exception as e:
            self.logger.warning(f"Gripper move failed: {e}")
            # try:
            #     self.robot.recover_from_errors()
            # except:
            #     pass
            return False

    def _move_to_initial_state(self, timeout_sec=10.0) -> bool:
        """Move robot to initial state using CartesianMotion."""
        # Convert RPY to xyzw Quaternion
        init_quat_xyzw = euler_to_quaternion_xyzw(
            self.initial_rpy[0], self.initial_rpy[1], self.initial_rpy[2]
        )
        
        self.logger.info(f"Moving to initial state: pos={self.initial_pos}, quat_xyzw={init_quat_xyzw}")
        
        try:
            # Move gripper to initial width
            self._move_gripper(self.initial_gripper_width, blocking=True)
            self.logger.info("Successfully moved to initial state.")

            # Create Affine transformation (position + quaternion)
            # Affine constructor: Affine(position, quaternion_xyzw)
            target_affine = Affine(self.initial_pos, init_quat_xyzw)
            
            # # Create RobotPose with optional elbow state
            # target_pose = RobotPose(target_affine)
            
            # Create CartesianMotion with absolute reference
            motion = CartesianMotion(target_affine)
            
            # Execute motion (blocking)
            self.robot.move(motion)
            


            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reach initial state: {e}")
            return False

    def reset(self) -> Optional[Dict]:
        """Reset the environment."""
        self.logger.info("Environment resetting...")
        
        self.current_step = 0
        
        if self.record_video:
            # If there was a previous video, save it only if it was a completed run.
            # However, the logic for saving is now handled in main loop based on completion.
            # Here we just clear old frames if any, and setup new path.
            self.video_frames = [] 
            self._setup_video_recording()
        
        # # Recover from any errors before reset
        # self.robot.recover_from_errors()
        
        success = self._move_to_initial_state()
        if not success:
            self.logger.warning("Failed to move to initial state precisely, continuing anyway.")
        

        time.sleep(1)
        obs = self.get_observation(timeout_sec=5.0)
        if obs is None:
            self.logger.error("Failed to get initial observation after reset.")
            return None
            
        self.logger.info(f"Environment reset successful. Step: {self.current_step}/{self.max_step}")
        return obs

    def step(self, action_dict: Dict) -> Tuple[Optional[Dict], bool, bool, str]:
        """Execute one step."""
        self.current_step += 1
        
        target_pos = action_dict['target_pos']
        target_quat_xyzw = action_dict['target_quat']  # xyzw format
        gripper_width = action_dict.get('gripper_width', 0.0)
        
        # Execute motion
        success = self._execute_motion(target_pos, target_quat_xyzw)
        
        # Control gripper (non-blocking to allow parallel execution)
        if success:
            self._move_gripper(gripper_width, blocking=True)
        
        # Get new observation
        obs = self.get_observation(timeout_sec=1.0)
        
        # Check truncation
        truncated = self.current_step >= self.max_step
        
        if truncated:
            info = f"Episode truncated: reached max_step ({self.max_step})"
        elif success:
            info = "Success"
        else:
            info = "Motion did not converge"
        self.logger.info(f"Step {self.current_step}/{self.max_step}: success={success}, truncated={truncated}")
        self.logger.info(f"Step {self.current_step}/{self.max_step}: step_action={action_dict}")
        return obs, success, truncated, info

    def _execute_motion(self, target_pos: np.ndarray, target_quat_xyzw: np.ndarray) -> bool:
        """
        Execute motion using franky CartesianMotion.
        Franky handles its own internal closed-loop control.
        """
        try:
            # Safety clamping
            target_pos = np.clip(
                target_pos, 
                [self.control_cfg.eef_x_min, self.control_cfg.eef_y_min, self.control_cfg.eef_z_min],
                [self.control_cfg.eef_x_max, self.control_cfg.eef_y_max, self.control_cfg.eef_z_max]
            )
            
            # Normalize quaternion
            # target_quat_xyzw = target_quat_xyzw / (np.linalg.norm(target_quat_xyzw) + 1e-8)
            # Create Affine: Affine(position, quaternion_xyzw)
            target_affine = Affine(target_pos, target_quat_xyzw)
            
            # # Create pose
            # target_pose = RobotPose(target_affine)
            
            # Create Cartesian motion with absolute reference
            motion = CartesianMotion(target_affine)
            
            # Execute (blocking until motion completes or fails)
            self.robot.move(motion)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Motion execution failed: {e}")
            # Try to recover from errors
            try:
                self.robot.recover_from_errors()
            except:
                pass
            return False

    def destroy(self):
        """Cleanup resources."""
        self.logger.info("Shutting down environment...")
        
        # Do NOT save video here automatically. Saving is controlled by the main loop.
        # If the program crashes or is killed, we might lose the video, which is acceptable
        # for aborted runs. For critical saves, the main loop should handle it.
        
        try:
            self.pipeline.stop()
            self.logger.info("RealSense pipeline stopped.")
        except Exception as e:
            self.logger.warning(f"Error stopping camera: {e}")
        
        # Note: franky doesn't require explicit disconnect, but we can cleanup
        self.logger.info("Environment destroyed.")


# =============================================================================
# Policy Adapter
# =============================================================================

class PolicyAdapter:
    def __init__(self, policy_client: PolicyClient, env: FrankaEnv):
        self.policy = policy_client
        self.env = env
        self.camera_keys = ["wrist_image"]
        self.control_cfg = env.control_cfg
        self.image_height, self.image_width = 256, 256
        self.action_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        self.current_gripper = 1
    def resize_and_pad_image(self, img: np.ndarray) -> np.ndarray:
        target_size = (self.image_width, self.image_height)
        h, w = img.shape[:2]
        ratio = min(float(target_size[1])/h, float(target_size[0])/w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        top = (target_size[1] - new_h) // 2
        left = (target_size[0] - new_w) // 2
        
        canvas[top:top+new_h, left:left+new_w] = resized
        return canvas
    
    def obs_to_policy_inputs(self, obs: Dict) -> Dict:
        """
        Convert observation to policy inputs.
        Obs quat is xyzw, convert to RPY for state input.
        """
        model_obs = {}
        model_obs["video"] = {k: self.resize_and_pad_image(obs[k]) for k in self.camera_keys}
        
        eef_pos = obs["eef_pos"].astype(np.float32)
        
        # Input is xyzw, transforms3d quat2euler expects xyzw
        quat_xyzw = obs["eef_quat"].astype(np.float64)
        rpy = quaternion_to_euler_xyzw(quat_xyzw).astype(np.float32)
        
        gripper_state = obs["gripper"].astype(np.float32)
        model_obs["state"] = {
            "x": eef_pos[0:1], 
            "y": eef_pos[1:2], 
            "z": eef_pos[2:3],
            "roll": rpy[0:1], 
            "pitch": rpy[1:2], 
            "yaw": rpy[2:3],
            "gripper": gripper_state,
        }
        model_obs["language"] = {
            "annotation.human.action.task_description": obs["lang"]
        }
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def get_action_commands(self, obs: Dict) -> List[Dict]:
        """
        Get action commands from policy.
        Policy outputs RPY, convert to xyzw for the Environment.
        """
        try:
            model_input = self.obs_to_policy_inputs(obs)
            action_chunk, info = self.policy.get_action(model_input)

            cmd_list = []
            action_chunk = np.concatenate(
                [action_chunk[k] for k in self.action_names], axis=2
            )
            action_horizon = action_chunk[0].shape[0]
            for t in range(action_horizon):
                action_vec = action_chunk[0][t]
                raw_pos = action_vec[:3]
                raw_rot_rpy = action_vec[3:6]  # Policy outputs RPY
                raw_gripper = action_vec[6]
                
                # Convert RPY -> xyzw
                target_quat_xyzw = euler_to_quaternion_xyzw(
                    raw_rot_rpy[0], raw_rot_rpy[1], raw_rot_rpy[2]
                )
                target_quat_xyzw = target_quat_xyzw / np.linalg.norm(target_quat_xyzw)
                
                # Safety Clamping
                target_pos = np.clip(
                    raw_pos, 
                    [self.control_cfg.eef_x_min, self.control_cfg.eef_y_min, self.control_cfg.eef_z_min],
                    [self.control_cfg.eef_x_max, self.control_cfg.eef_y_max, self.control_cfg.eef_z_max]
                )
                
                gripper_width = raw_gripper * self.control_cfg.gripper_max_width
                gripper_width = np.clip(gripper_width, 0.0, self.control_cfg.gripper_max_width)

                cmd_list.append({
                    "target_pos": target_pos,
                    "target_quat": target_quat_xyzw,  # Pass xyzw to Env
                    "gripper_width": gripper_width
                })
            return cmd_list
            
        except Exception as e:
            self.env.logger.error(f"Policy inference failed: {e}")
            import traceback
            traceback.print_exc()
            return []


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PANDA Real-Robot GR00T Policy Evaluation (Franky + RealSense) with Keyboard Control"
    )
    parser.add_argument("--policy_host", type=str, default="localhost")
    parser.add_argument("--policy_port", type=int, default=5555)
    parser.add_argument("--lang_instruction", type=str, default="pick up the banana.")
    parser.add_argument("--experiment_path", type=str, default="./experiments")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--max_step", type=int, default=800)
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2")
    parser.add_argument("--gripper_ip", type=str, default="172.16.0.2")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=60)
    parser.add_argument("--velocity_rel", type=float, default=0.05)
    
    args = parser.parse_args()
    
    cfg = EvalConfig(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        lang_instruction=args.lang_instruction,
        experiment_path=args.experiment_path,
        record_video=args.record_video,
        max_step=args.max_step,
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        camera_config=CameraConfig(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps
        ),
    )
    
    # Update control config with velocity parameter
    cfg.control.velocity_rel = args.velocity_rel
    cfg.control.acceleration_rel = args.velocity_rel
    cfg.control.jerk_rel = args.velocity_rel
    cfg.control.relative_dynamics_factor = args.velocity_rel
    
    env = None
    keyboard = KeyboardController()
    
    try:
        print(f"Connecting to policy server at {cfg.policy_host}:{cfg.policy_port}...")
        policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        print("Policy connected.")
        
        env = FrankaEnv(cfg)
        policy_adapter = PolicyAdapter(policy_client, env)
        
        print("\n=== Keyboard Controls ===")
        print("  [s] Start execution (reset -> step until done)")
        print("  [e] Stop/Abort current execution (save video)")
        print("  [q] Quit program (do not save video)")
        print("=========================\n")
        print("Press 's' to begin...")
        
        # Enable keyboard listening
        keyboard.enable()
        
        running = True
        is_executing = False
        
        while running:
            # Check for key presses
            key = keyboard.get_key()
            
            if key:
                key = key.lower()
                if key == 'q':
                    print("\nQuitting program...")
                    running = False
                    break
                elif key == 'e':
                    if is_executing:
                        print("\nAborting current execution...")
                        is_executing = False
                        # Clear video frames for this aborted run
                        if cfg.record_video:
                            env._save_video()
                            env._clear_video()
                    else:
                        print("No execution in progress to abort.")
                elif key == 's':
                    if is_executing:
                        # User pressed 's' while already running: abort current and restart
                        print("\nRestarting execution (aborting previous)...")
                        if cfg.record_video:
                            env._clear_video()
                        # We let the loop below detect is_executing=False and break, 
                        # then this block will re-initiate.
                        is_executing = False
                    else:
                        # Start new execution
                        print("\nStarting new execution...")
                        obs = env.reset()
                        if obs is None:
                            print("Failed to initialize environment. Waiting for command...")
                            continue
                        
                        is_executing = True
                        done = False
                        
                        # Execution Loop
                        while is_executing and not done:
                            # Check for interrupt keys inside the execution loop
                            key = keyboard.get_key()
                            if key:
                                key = key.lower()
                                if key == 's':
                                    print("\nRestart requested during execution. Aborting current...")
                                    if cfg.record_video:
                                        env._clear_video()
                                    is_executing = False # Break inner loop
                                    break # Break out of step loop to restart
                                elif key == 'e':
                                    print("\nStop requested during execution. Aborting...")
                                    if cfg.record_video:
                                        env._save_video()
                                        env._clear_video()
                                    is_executing = False
                                    break
                                elif key == 'q':
                                    print("\nQuit requested during execution.")
                                    running = False
                                    is_executing = False
                                    break
                            
                            cmd_list = policy_adapter.get_action_commands(obs)
                            
                            if not cmd_list:
                                env.logger.warning("No actions generated. Waiting...")
                                time.sleep(0.5)
                                obs = env.get_observation()
                                continue
                            
                            # Execute action horizon
                            for cmd in cmd_list[:8]:
                                # Check for interrupts even during step execution
                                key = keyboard.get_key()
                                if key:
                                    key = key.lower()
                                    if key in ['s', 'e', 'q']:
                                        if key == 's':
                                            print("\nRestart requested during step. Aborting...")
                                            if cfg.record_video:
                                                env._clear_video()
                                            is_executing = False
                                        elif key == 'e':
                                            print("\nStop requested during step. Aborting...")
                                            if cfg.record_video:
                                                env._save_video()
                                                env._clear_video()
                                            is_executing = False
                                        elif key == 'q':
                                            print("\nQuit requested during step.")
                                            running = False
                                            is_executing = False
                                        break # Break step loop
                                
                                if not is_executing or not running:
                                    break
                                
                                tic = time.time()
                                env.logger.info(f"state.gripper: {obs['gripper']}")
                                next_obs, success, truncated, info = env.step(cmd)
                                # env.logger.info(f"state_pos={obs['eef_pos']}, state_quat={obs['eef_quat']}")
                                # env.logger.info(f"action={cmd}")
                                if next_obs:
                                    obs = next_obs
                                else:
                                    env.logger.error("Lost observation during step. Breaking.")
                                    is_executing = False
                                    break
                                
                                if truncated:
                                    print(f"Episode truncated: {info}")
                                    done = True
                                    break
                                
                                # Rate limiting
                                toc = time.time()
                                action_fps = 20
                                elapsed = toc - tic
                                if elapsed < 1.0 / action_fps:
                                    time.sleep(1.0 / action_fps - elapsed)
                            
                            if done and is_executing:
                                print("Episode finished successfully.")
                                if cfg.record_video:
                                    env._save_video()
                                is_executing = False # Mark as finished to return to idle
                                done = False # Reset for next potential run
                                # Break out to main loop to wait for next command
                                break 

                        if not running:
                            break
                        # If we broke out due to 's' (restart), the outer loop continues 
                        # and will see is_executing=False, then immediately trigger 's' logic again?
                        # No, the 's' key was consumed. We need to explicitly restart if 's' was the cause.
                        # The current logic sets is_executing=False and breaks. 
                        # The outer loop will then wait for a NEW key press.
                        # To make 's' during execution instantly restart, we can set a flag.
                        
                        # Let's refine: if 's' triggered the break, we want to restart immediately.
                        # We can check why we broke. But simpler: just let user press 's' again?
                        # Requirement: "当在执行阶段输入 s，代表这次执行废弃，直接重新执行"
                        # So if 's' was pressed, we should restart immediately without waiting for another 's'.
                        
                        # Re-evaluating the flow for 's' during execution:
                        # If key=='s' detected inside, we set is_executing=False, clear video, and break.
                        # After breaking the inner `while is_executing` loop, we are back in the `while running` loop.
                        # We need to know if we should restart immediately.
                        # Let's use a variable `restart_requested`.
                        
                        # Actually, the structure above has a flaw for immediate restart.
                        # Let's restructure slightly:
                        pass # Placeholder, logic handled below with a flag

            # If we are not executing, we just loop waiting for keys.
            # The key handling is done at the top of the loop.
            if not is_executing and running:
                time.sleep(0.05) # Small sleep to prevent CPU spinning when idle

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        keyboard.disable()
        if env:
            # Do not auto-save on exit unless explicitly finished
            env.destroy()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()