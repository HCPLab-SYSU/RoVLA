"""
PANDA Real-Robot GR00T Policy Evaluation Script (ROS 2 Version)
- Uses ROS 2 for communication (Subscribers/Publishers).
- Uses only Wrist Camera (/camera/camera/color/image_rect_raw).
- Synchronizes latest state and image data using ApproximateTimeSynchronizer.
- Controls Franka via Cartesian Pose commands (6D pose + Gripper).
- Implements closed-loop position control.
"""

# =============================================================================
# Imports
# =============================================================================
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from pprint import pformat

# GR00T & Policy Imports
from gr00t.policy.server_client import PolicyClient
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qmult, rotate_vector, mat2quat, euler2quat

# ROS 2 Message Types
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32
from franka_msgs.msg import RobotState
from franka_msgs.action import Grasp, Move
from cv_bridge import CvBridge
# Message Filters for synchronization
import message_filters
import argparse
# =============================================================================
# Configuration - All Topic Names and Parameters Here
# =============================================================================

@dataclass
class TopicConfig:
    """Unified configuration for all ROS 2 topic names and types."""
    
    # === Subscribers ===
    # Image topic (Wrist Camera)
    image_topic: str = "/camera/camera/color/image_rect_raw"
    image_msg_type: str = "sensor_msgs/msg/Image"
    
    # End-effector Pose topic (from franka_robot_state_broadcaster)
    pose_topic: str = "/franka_robot_state_broadcaster/current_pose"
    pose_msg_type: str = "geometry_msgs/msg/PoseStamped"
    
    # Gripper joint states topic
    gripper_joint_state_topic: str = "/franka_gripper/joint_states"
    gripper_joint_state_msg_type: str = "sensor_msgs/msg/JointState"
    
    # Alternative: Direct gripper state (if available)
    # gripper_state_topic: str = "/franka_gripper/current_state"
    
    # === Publishers ===
    # Cartesian Pose Command (for pose-based controllers)
    cartesian_pose_command_topic: str = "/cartesian_pose_example_controller/pose_command"
    cartesian_pose_command_msg_type: str = "geometry_msgs/msg/PoseStamped"
    
    # === Actions ===
    # Gripper Action Server
    gripper_action_name: str = "/franka_gripper/move"
    gripper_grasp_action_name: str = "/franka_gripper/grasp"
    
    # === QoS Settings ===
    # For sensors (images)
    sensor_qos_depth: int = 10
    sensor_qos_reliability: str = "BEST_EFFORT"
    
    # For state (pose, joint states)
    state_qos_depth: int = 10
    state_qos_reliability: str = "RELIABLE"
    
    # Time synchronizer settings
    sync_queue_size: int = 10
    sync_slop: float = 0.1  # seconds
    action_horizon: int = 8

@dataclass
class ControlConfig:
    """Control parameters and workspace bounds."""
    
    # Workspace bounds (meters)
    eef_x_min: float = -1.0
    eef_x_max: float = 1.0
    eef_y_min: float = -1.0
    eef_y_max: float = 1.0
    eef_z_min: float = -1.0
    eef_z_max: float = 1.0
    
    
    # Closed-loop control parameters
    use_closed_loop: bool = True
    position_tolerance: float = 0.01  # meters (1cm)
    orientation_tolerance: float = 0.05  # radians (~3 degrees)
    
    # PID gains for closed-loop control (simplified P controller)
    kp_position: float = 2.0
    kp_orientation: float = 1.5
    
    # Max velocity limits (safety)
    max_linear_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 0.5  # rad/s
    
    # Gripper settings
    gripper_max_width: float = 0.08  # meters
    gripper_speed: float = 0.03  # m/s
    gripper_force: float = 50.0  # Newtons


@dataclass
class EvalConfig:
    """Main evaluation configuration."""
    policy_host: str = "localhost"
    policy_port: int = 5555
    lang_instruction: str = "Pick up the red block."
    
    # Include sub-configs
    topics: TopicConfig = field(default_factory=TopicConfig)
    control: ControlConfig = field(default_factory=ControlConfig)


# =============================================================================
# Helper Functions
# =============================================================================

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """Add batch (B=1) and time (T=1) dimensions to numpy arrays in obs dict."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                obs[key] = val[np.newaxis, np.newaxis, ...]
            elif val.ndim == 3:  # Image H, W, C -> 1, 1, H, W, C
                obs[key] = val[np.newaxis, np.newaxis, ...]
            else:
                obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [[val]] if not isinstance(val, list) else [val]
    return obs


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Calculate angular distance between two quaternions (xyzw format)."""
    # Convert to wxyz for calculation
    q1_wxyz = np.roll(q1, 1)
    q2_wxyz = np.roll(q2, 1)
    
    # Dot product
    dot = np.dot(q1_wxyz, q2_wxyz)
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angle
    angle = 2.0 * np.arccos(abs(dot))
    return angle


# =============================================================================
# ROS 2 Adapter Class with Closed-Loop Control
# =============================================================================

class FrankaRos2Adapter(Node):
    def __init__(self, policy_client: PolicyClient, cfg: EvalConfig):
        super().__init__('franka_gr00t_ros2_adapter')
        
        self.cfg = cfg
        self.topics = cfg.topics
        self.control_cfg = cfg.control
        self.policy = policy_client
        self.bridge = CvBridge()
        
        # Callback group for action clients
        self.callback_group = ReentrantCallbackGroup()
        
        # --- State Buffers ---
        self.latest_eef_pos: Optional[np.ndarray] = None
        self.latest_eef_quat: Optional[np.ndarray] = None  # xyzw
        self.latest_gripper: Optional[np.ndarray] = None
        self.latest_image: Optional[np.ndarray] = None
        
        self.state_received = False
        self.image_received = False
        
        # Closed-loop control state
        self.target_pose: Optional[PoseStamped] = None
        self.control_active = False
        
        # --- QoS Profiles ---
        sensor_qos = QoSProfile(
            depth=self.topics.sync_queue_size,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )
        
        state_qos = QoSProfile(
            depth=self.topics.sync_queue_size,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # --- Subscribers with Message Filters ---
        # 1. Image Subscriber
        self.sub_image = message_filters.Subscriber(
            self, 
            Image, 
            self.topics.image_topic,
            qos_profile=sensor_qos
        )
        
        # 2. Pose Subscriber (End-effector pose from robot state broadcaster)
        self.sub_pose = message_filters.Subscriber(
            self,
            PoseStamped,
            self.topics.pose_topic,
            qos_profile=state_qos
        )
        
        # 3. Gripper Joint State Subscriber
        self.sub_gripper = message_filters.Subscriber(
            self,
            JointState,
            self.topics.gripper_joint_state_topic,
            qos_profile=state_qos
        )
        
        # Time Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_pose, self.sub_gripper],
            queue_size=self.topics.sync_queue_size,
            slop=self.topics.sync_slop
        )
        self.ts.registerCallback(self._synchronized_callback)
        
        # --- Publishers ---
        # Cartesian Pose Command Publisher
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.topics.cartesian_pose_command_topic,
            10
        )
        
        # --- Action Clients for Gripper ---
        self.gripper_move_client = ActionClient(
            self,
            Move,
            self.topics.gripper_action_name,
            callback_group=self.callback_group
        )
        
        self.gripper_grasp_client = ActionClient(
            self,
            Grasp,
            self.topics.gripper_grasp_action_name,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f"FrankaRos2Adapter initialized.")
        self.get_logger().info(f"  Image topic: {self.topics.image_topic}")
        self.get_logger().info(f"  Pose topic: {self.topics.pose_topic}")
        self.get_logger().info(f"  Gripper topic: {self.topics.gripper_joint_state_topic}")
        self.get_logger().info(f"  Pose command topic: {self.topics.cartesian_pose_command_topic}")
        self.get_logger().info("Waiting for synchronized data...")

    def _synchronized_callback(self, image_msg: Image, pose_msg: PoseStamped, gripper_msg: JointState):
        """Callback for synchronized messages."""
        try:
            # Process Image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            self.latest_image = cv_image
            
            # Process Pose
            pos = np.array([
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z
            ])
            quat = np.array([
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w
            ])
            
            self.latest_eef_pos = pos
            self.latest_eef_quat = quat
            
            # Process Gripper State
            # Franka gripper has two finger joints, width is sum of both positions
            if len(gripper_msg.position) == 2:
                self.latest_gripper = np.array(
                    gripper_msg.position[0],
                    gripper_msg.position[1]
                )
            elif len(gripper_msg.position) == 1:
                self.latest_gripper = np.array(
                    gripper_msg.position[0]/2,
                    gripper_msg.position[0]/2
                )
            
            self.state_received = True
            self.image_received = True
            
        except Exception as e:
            self.get_logger().warn(f"Error in synchronized callback: {e}")

    def get_synchronized_observation(self, timeout_sec=2.0) -> Optional[Dict]:
        """Wait for synchronized observation data."""
        start_time = time.time()
        
        while not (self.state_received and self.image_received):
            if time.time() - start_time > timeout_sec:
                self.get_logger().error("Timeout waiting for synchronized sensor data.")
                return None
            time.sleep(0.01)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.latest_eef_pos is None or self.latest_image is None:
            return None
        
        obs = {
            "eef_pos": self.latest_eef_pos.copy(),
            "eef_quat": self.latest_eef_quat.copy(),
            "gripper": self.latest_gripper.copy(),
            "wrist_image": self.latest_image.copy(),
            "lang": self.cfg.lang_instruction
        }
        
        return obs

    def send_pose_command(self, position: np.ndarray, quaternion: np.ndarray):
        """Send a Cartesian pose command to the robot."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "franka_link0"  # Base frame
        
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.z = float(quaternion[3])
        
        self.pose_pub.publish(pose_msg)
        
        # Update target for closed-loop monitoring
        self.target_pose = pose_msg

    def send_gripper_command(self, width: float, use_grasp: bool = False):
        """Send gripper command using Action interface."""
        width = np.clip(width, 0.0, self.control_cfg.gripper_max_width)
        
        if use_grasp:
            # Use grasp action for grasping objects
            goal_msg = Grasp.Goal()
            goal_msg.width = width
            goal_msg.speed = self.control_cfg.gripper_speed
            goal_msg.force = self.control_cfg.gripper_force
            goal_msg.epsilon.inner = 0.05
            goal_msg.epsilon.outer = 0.05
            
            self.gripper_grasp_client.send_goal_async(goal_msg)
        else:
            # Use move action for simple positioning
            goal_msg = Move.Goal()
            goal_msg.width = width
            goal_msg.speed = self.control_cfg.gripper_speed
            
            self.gripper_move_client.send_goal_async(goal_msg)

    def check_pose_convergence(self, current_pos: np.ndarray, current_quat: np.ndarray,
                                target_pos: np.ndarray, target_quat: np.ndarray) -> Tuple[bool, float, float]:
        """
        Check if current pose has converged to target pose.
        Returns: (is_converged, position_error, orientation_error)
        """
        pos_error = np.linalg.norm(current_pos - target_pos)
        orient_error = quaternion_distance(current_quat, target_quat)
        
        pos_converged = pos_error < self.control_cfg.position_tolerance
        orient_converged = orient_error < self.control_cfg.orientation_tolerance
        
        is_converged = pos_converged and orient_converged
        
        return is_converged, pos_error, orient_error

    def execute_closed_loop_control(self, target_pos: np.ndarray, target_quat: np.ndarray, 
                                     max_steps: int = 50) -> bool:
        """
        Execute closed-loop pose control until convergence or max steps.
        Returns True if converged successfully.
        """
        if not self.control_cfg.use_closed_loop:
            # Open-loop: just send once
            self.send_pose_command(target_pos, target_quat)
            return True
        
        self.get_logger().info(f"Starting closed-loop control to target: pos={target_pos}, quat={target_quat}")
        
        for step in range(max_steps):
            # Get current state
            obs = self.get_synchronized_observation(timeout_sec=0.5)
            if obs is None:
                self.get_logger().warn("Lost sensor data during closed-loop control.")
                return False
            
            current_pos = obs["eef_pos"]
            current_quat = obs["eef_quat"]
            
            # Check convergence
            converged, pos_err, orient_err = self.check_pose_convergence(
                current_pos, current_quat, target_pos, target_quat
            )
            
            if converged:
                self.get_logger().info(f"Pose converged at step {step}: pos_err={pos_err:.4f}m, orient_err={orient_err:.4f}rad")
                # Send final command
                self.send_pose_command(target_pos, target_quat)
                return True
            
            # P-controller: compute velocity command based on error
            pos_error_vec = target_pos - current_pos
            orient_error = quaternion_distance(current_quat, target_quat)
            
            # Simple proportional control
            linear_vel = np.clip(pos_error_vec * self.control_cfg.kp_position, 
                                  -self.control_cfg.max_linear_velocity,
                                  self.control_cfg.max_linear_velocity)
            
            # For orientation, we'll just send the target directly (simplified)
            # A full implementation would compute angular velocity from quaternion error
            
            # Send intermediate pose command (blending current and target)
            # This is a simplified approach - ideally use a proper trajectory generator
            blend_factor = min(1.0, np.linalg.norm(linear_vel) / self.control_cfg.max_linear_velocity)
            intermediate_pos = current_pos + linear_vel * 0.02  # 20ms step
            
            self.send_pose_command(intermediate_pos, target_quat)
            
            self.get_logger().debug(f"Step {step}: pos_err={pos_err:.4f}m, orient_err={orient_err:.4f}rad")
            
            time.sleep(0.02)  # 50Hz control loop
        
        self.get_logger().warn(f"Closed-loop control did not converge within {max_steps} steps.")
        return False


# =============================================================================
# Policy Adapter Logic
# =============================================================================

class PolicyAdapter:
    def __init__(self, policy_client: PolicyClient, ros_adapter: FrankaRos2Adapter):
        self.policy = policy_client
        self.ros_adapter = ros_adapter
        self.camera_keys = ["wrist_image"]
        self.control_cfg = ros_adapter.control_cfg

    def obs_to_policy_inputs(self, obs: Dict) -> Dict:
        model_obs = {f'video.{k}': obs[k] for k in self.camera_keys}
        
        eef_pos = obs["eef_pos"]
        eef_quat = obs["eef_quat"]  # xyzw
        
        # Convert xyzw to wxyz for transforms3d
        quat_wxyz = np.roll(eef_quat, 1)
        rpy = np.array(quat2euler(quat_wxyz, axes='sxyz'))
        
        gripper_state = obs["gripper"]
        
        state = {
            "state.x": [eef_pos[0]], 
            "state.y": [eef_pos[1]], 
            "state.z": [eef_pos[2]],
            "state.roll": [rpy[0]], 
            "state.pitch": [rpy[1]], 
            "state.yaw": [rpy[2]],
            "state.gripper": gripper_state,
        }
        model_obs.update(state)
        model_obs["annotation.human.action.task_description"] = obs["lang"]
        
        model_obs = recursive_add_extra_dim(model_obs)
        
        return model_obs

    def get_action_and_convert(self, obs: Dict) -> Optional[Dict]:
        try:
            model_input = self.obs_to_policy_inputs(obs)
            action_chunk, info = self.policy.get_action(model_input)
            
            if "action" not in action_chunk:
                raise KeyError('Policy output missing "action"')
            
            cmd_list = []
            action_horizon = action_chunk["action"][0].shape[0]
            for t in range(action_horizon):
                action_vec = action_chunk["action"][0][t]  # Shape (7,) -> [x, y, z, roll, pitch, yaw, gripper]
                
                # =================================================================
                # 1. Parse Action Vector (Direct 6D Pose Interpretation)
                # =================================================================
                # 动作向量结构: [x, y, z, r, p, y, gripper]
                # 注意：这里假设模型输出的是【绝对坐标】(Absolute Coordinates)
                # gripper在训练中中输出的是归一化值(0~1)
                
                raw_pos = action_vec[:3]       # [x, y, z]
                raw_rot = action_vec[3:6]      # [roll, pitch, yaw] (Euler angles)
                raw_gripper = action_vec[6]    # Gripper state
                
                current_pos = obs["eef_pos"]
                current_quat = obs["eef_quat"] # xyzw
                
                # --- 位置处理 (Position) ---
                # 模型直接输出绝对世界坐标 (米)
                target_pos = raw_pos 
                
                # 直接使用输出的欧拉角作为目标绝对姿态
                target_quat_wxyz = euler2quat(raw_rot[0], raw_rot[1], raw_rot[2], axes='sxyz')
                
                # 转换回 xyzw 格式 (ROS 标准)
                target_quat_xyzw = np.roll(target_quat_wxyz, -1)
                
                # 归一化四元数 (防止数值漂移)
                target_quat_xyzw = target_quat_xyzw / np.linalg.norm(target_quat_xyzw)
                
                # =================================================================
                # 2. Safety Constraints & Clamping
                # =================================================================
                
                # 限制位置在工作空间内
                target_pos = np.clip(target_pos, 
                                    [self.control_cfg.eef_x_min, self.control_cfg.eef_y_min, self.control_cfg.eef_z_min],
                                    [self.control_cfg.eef_x_max, self.control_cfg.eef_y_max, self.control_cfg.eef_z_max])
                
                # 限制夹爪宽度
                # 模型输出宽度百分比：0=close 1=open
                gripper_width = raw_gripper*self.control_cfg.gripper_max_width
                    
                gripper_width = np.clip(gripper_width, 0.0, self.control_cfg.gripper_max_width)
                cmd_list.append(
                    {
                        "target_pos": target_pos,
                        "target_quat": target_quat_xyzw,
                        "gripper_width": gripper_width
                    }
                )
            return cmd_list
            
        except Exception as e:
            self.ros_adapter.get_logger().error(f"Policy inference failed: {e}")
            import traceback
            traceback.print_exc()
            return []


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = EvalConfig()
    #用户命令行输入参数控制host,port,lang_instruction
    parser = argparse.ArgumentParser(description="PANDA Real-Robot GR00T Policy Evaluation")
    parser.add_argument("--policy_host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=5555, help="Policy server port")
    parser.add_argument("--lang_instruction", type=str, default="pick the banana", help="Language instruction for the task")
    args = parser.parse_args()
    cfg.policy_host = args.policy_host
    cfg.policy_port = args.policy_port
    cfg.lang_instruction = args.lang_instruction
    
    rclpy.init()
    node = None
    
    try:
        # 1. Connect to Policy Server
        print(f"Connecting to policy server at {cfg.policy_host}:{cfg.policy_port}...")
        policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        print("Policy connected.")
        
        # 2. Initialize ROS 2 Node and Adapter
        node = FrankaRos2Adapter(policy_client, cfg)
        policy_adapter = PolicyAdapter(policy_client, node)
        
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        # Spin to establish connections
        for _ in range(20):
            executor.spin_once(timeout_sec=0.1)
        
        # Wait for gripper action servers
        if not node.gripper_move_client.wait_for_server(timeout_sec=5.0):
            node.get_logger().warn("Gripper move action server not available.")
        
        print("Starting evaluation loop... Press Ctrl+C to stop.")
        
        while rclpy.ok():
            # A. Get Synchronized Observation
            obs = node.get_synchronized_observation(timeout_sec=1.0)
            if obs is None:
                node.get_logger().warn("Skipping step due to missing observation.")
                continue
            
            # B. Inference
            cmd_list = policy_adapter.get_action_and_convert(obs)
            if len(cmd_list)==0:
                continue
            for cmd in cmd_list[:cfg.action_horizon]:
                tic = time.time()
                # C. Execute Closed-Loop Pose Control
                node.get_logger().info(f"Executing motion to pos={cmd['target_pos']}, gripper={cmd['gripper_width']:.3f}")
                
                success = node.execute_closed_loop_control(
                    cmd["target_pos"],
                    cmd["target_quat"]
                )
                
                if success:
                    # Send gripper command after reaching pose
                    node.send_gripper_command(cmd["gripper_width"])
                else:
                    node.get_logger().warn("Motion did not converge, skipping gripper command.")
                
                # D. Rate Limit (e.g., 10Hz for high-level planning)
                toc = time.time()
                action_fps = 20
                if toc - tic < 1.0 / action_fps:
                    time.sleep(1.0 / action_fps - (toc - tic))
            
    except KeyboardInterrupt:
        print("Evaluation interrupted.")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()