#!/bin/bash

# ==============================================================================
# 一键启动 Franka + RealSense + 控制器 脚本
# 作者：Assistant
# 日期：2026-03-11
# ==============================================================================

# --- 配置区域 ---
# 日志存储根目录
LOG_ROOT_DIR="/media/hcp/disk/projects/hcp_logs"
# 本次运行的时间戳文件夹 (格式: YYYY-MM-DD_HH-MM-SS)
TIMESTAMP=$(date +"%Y-%m-%d")
RUN_LOG_DIR="${LOG_ROOT_DIR}/franka_run_${TIMESTAMP}"

# 机器人参数
ARM_ID="fr3"
NAMESPACE=""
ROBOT_IP="172.16.0.2"
LOAD_GRIPPER="true"

# --- 初始化 ---
echo "=================================================="
echo "正在初始化启动脚本..."
echo "日志将保存至: ${RUN_LOG_DIR}"
echo "=================================================="

# 1. 创建日志目录 (如果不存在则创建，-p 保证父目录也创建)
mkdir -p "${RUN_LOG_DIR}"

if [ ! -d "${LOG_ROOT_DIR}" ]; then
    echo "错误：根日志目录 ${LOG_ROOT_DIR} 不存在且无法创建。请检查磁盘挂载情况。"
    exit 1
fi

# 定义环境加载函数 (避免重复代码)
# 注意：在后台进程 (&) 中，当前 shell 的环境变量有时不会完美继承，
# 最稳妥的方式是在每个后台子进程中重新 source 必要的环境。
load_env() {
    source $HOME/miniforge3/bin/activate datacollect
    source /opt/ros/humble/setup.bash
    source ~/franka_ros2_ws/install/setup.bash
}

# --- 启动节点 ---

# 2. 启动 Franka Bringup
# 使用 { ... } & 结构在后台运行，并重定向日志
echo "[1/4] 启动 Franka Bringup (arm_id: ${ARM_ID}, ip: ${ROBOT_IP}) ..."
{
    load_env
    echo "[$(date)] Starting franka_bringup..."
    ros2 launch franka_bringup franka.launch.py \
        arm_id:=${ARM_ID} \
        namespace:=${NAMESPACE} \
        robot_ip:=${ROBOT_IP} \
        load_gripper:=${LOAD_GRIPPER}
} > "${RUN_LOG_DIR}/franka_bringup.log" 2>&1 &
PID_FRANKA=$!

# 3. 启动 RealSense Camera
echo "[2/4] 启动 RealSense Camera ..."
{
    load_env
    echo "[$(date)] Starting realsense2_camera..."
    ros2 launch realsense2_camera rs_launch.py
} > "${RUN_LOG_DIR}/realsense_camera.log" 2>&1 &
PID_REALSENSE=$!

# 4. 等待系统准备就绪
# Franka 和 RealSense 启动需要时间，直接运行 spawner 可能会因为 controller_manager 未就绪而失败
echo "[3/4] 等待控制器管理器 (controller_manager) 就绪..."
# 这里简单等待 5 秒，实际生产中建议用 while 循环检测 ros2 service list /controller_manager/list_controllers
sleep 8 

# 5. 启动 Cartesian Pose Controller
echo "[4/4] 启动 Cartesian Pose Controller ..."
{
    load_env
    echo "[$(date)] Spawning cartesian_pose_example_controller..."
    ros2 run controller_manager spawner cartesian_pose_example_controller --controller-manager /controller_manager
} > "${RUN_LOG_DIR}/spawner_cartesian.log" 2>&1 &
PID_SPAWNER_CART=$!

# 6. 启动 Gripper Controller
echo "[5/4] 启动 Gripper Controller ..."
{
    load_env
    echo "[$(date)] Spawning gripper_example_controller..."
    ros2 run controller_manager spawner gripper_example_controller --controller-manager /controller_manager
} > "${RUN_LOG_DIR}/spawner_gripper.log" 2>&1 &
PID_SPAWNER_GRIP=$!

# --- 结束与监控 ---
echo ""
echo "=================================================="
echo "所有节点已后台启动!"
echo "Franka PID: ${PID_FRANKA}"
echo "RealSense PID: ${PID_REALSENSE}"
echo "Spawner (Cart) PID: ${PID_SPAWNER_CART}"
echo "Spawner (Grip) PID: ${PID_SPAWNER_GRIP}"
echo ""
echo "日志实时查看命令:"
echo "tail -f ${RUN_LOG_DIR}/*.log"
echo ""
echo "停止所有相关进程命令 (可选):"
echo "kill ${PID_FRANKA} ${PID_REALSENSE} ${PID_SPAWNER_CART} ${PID_SPAWNER_GRIP}"
echo "=================================================="

# 可选：如果你希望脚本不要立即退出，而是按住前台直到用户 Ctrl+C，然后清理进程，取消下面注释
wait ${PID_FRANKA} ${PID_REALSENSE} ${PID_SPAWNER_CART} ${PID_SPAWNER_GRIP}
# echo "检测到中断，正在停止进程..."
kill ${PID_FRANKA} ${PID_REALSENSE} ${PID_SPAWNER_CART} ${PID_SPAWNER_GRIP} 2>/dev/null