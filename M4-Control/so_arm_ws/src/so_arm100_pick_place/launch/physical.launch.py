"""
physical.launch.py
==================
Lanza el pipeline completo para el robot físico SO-ARM100.

Nodos que inicia:
  1. robot_state_publisher   — URDF → árbol TF (sin Gazebo)
  2. controller_manager      — ros2_control con mock_components
  3. joint_state_broadcaster — publica /joint_states
  4. arm_controller          — FollowJointTrajectory para el brazo
  5. gripper_controller      — FollowJointTrajectory para el gripper
  6. move_group (MoveIt2)    — planificación de trayectorias
  7. realsense_node          — RealSense D435 RGB + depth
  8. vision_detector         — detección CV → /vision/best_object
  9. lqr_visual_servoing     — control LQR → MoveIt2 → brazo
 10. RViz2 (opcional)

Uso:
  ros2 launch so_arm100_pick_place physical.launch.py
  ros2 launch so_arm100_pick_place physical.launch.py rviz:=false
  ros2 launch so_arm100_pick_place physical.launch.py log_csv:=false

NOTA: mock_components refleja los comandos de posición como joint_states.
      Para feedback real del brazo, conectar el bridge Feetech → ros2_control.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml(pkg, filepath):
    pkg_path = get_package_share_directory(pkg)
    with open(os.path.join(pkg_path, filepath), "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    desc_pkg   = get_package_share_directory("so_arm100_description")
    pick_pkg   = get_package_share_directory("so_arm100_pick_place")

    urdf_xacro = os.path.join(desc_pkg, "urdf", "so_arm100.urdf.xacro")
    ctrl_yaml  = os.path.join(desc_pkg, "config", "ros2_controllers.yaml")
    rviz_cfg   = os.path.join(pick_pkg, "rviz", "vision_pick.rviz")

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", urdf_xacro, " use_sim:=false"]),
            value_type=str,
        )
    }

    # MoveIt2 params
    srdf_file = os.path.join(
        get_package_share_directory("so_arm100_moveit_config"), "config", "so_arm100.srdf"
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            Command(["cat ", srdf_file]), value_type=str
        )
    }
    kinematics_yaml    = load_yaml("so_arm100_moveit_config", "config/kinematics.yaml")
    ompl_yaml          = load_yaml("so_arm100_moveit_config", "config/ompl_planning.yaml")
    joint_limits_yaml  = load_yaml("so_arm100_moveit_config", "config/joint_limits.yaml")
    controllers_yaml   = load_yaml("so_arm100_moveit_config", "config/moveit_controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml["moveit_simple_controller_manager"],
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    return LaunchDescription([
        # ── Args ────────────────────────────────────────────────────────────
        DeclareLaunchArgument("rviz",       default_value="true"),
        DeclareLaunchArgument("log_csv",    default_value="true"),
        DeclareLaunchArgument("realsense",  default_value="true",
                              description="false = skip RealSense (sin cámara conectada)"),

        # ── 1. robot_state_publisher ─────────────────────────────────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description, {"use_sim_time": False}],
        ),

        # ── 2. controller_manager (mock hardware) ────────────────────────────
        # Remapeamos joint_states→joint_states_mock a NIVEL DEL NODO
        # para que feetech_bridge sea el único publicador de /joint_states reales
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[robot_description, ctrl_yaml, {"use_sim_time": False}],
            remappings=[("joint_states", "joint_states_mock")],
        ),

        # ── 3-5. Spawner de controllers ──────────────────────────────────────
        TimerAction(period=2.0, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),
        TimerAction(period=3.0, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["arm_controller", "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),
        TimerAction(period=3.5, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 6. MoveIt2 move_group ────────────────────────────────────────────
        TimerAction(period=4.0, actions=[
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    {"robot_description_kinematics": kinematics_yaml},
                    {"robot_description_planning": {"joint_limits": joint_limits_yaml}},
                    ompl_yaml,
                    moveit_controllers,
                    {"use_sim_time": False},
                    {"publish_robot_description_semantic": True},
                    {"planning_plugin": "ompl_interface/OMPLPlanner"},
                ],
            ),
        ]),

        # ── 7. Feetech bridge — conecta ros2_control ↔ motores físicos ──────────
        Node(
            package="so_arm100_pick_place",
            executable="feetech_bridge.py",
            name="feetech_bridge",
            output="screen",
            parameters=[{
                "port":       "/dev/ttyACM0",
                "move_time":  0.15,
                "deadband_deg": 0.5,
            }],
        ),

        # ── 8. RealSense D435 ─────────────────────────────────────────────────
        Node(
            package="so_arm100_pick_place",
            executable="realsense_node.py",
            name="realsense_node",
            output="screen",
            parameters=[{"width": 640, "height": 480, "fps": 30}],
            condition=IfCondition(LaunchConfiguration("realsense")),
        ),

        # ── 8. Vision detector (espera que la cámara arranque) ────────────────
        TimerAction(period=3.0, actions=[
            Node(
                package="so_arm100_pick_place",
                executable="vision_detector.py",
                name="vision_detector",
                output="screen",
                parameters=[{
                    "min_depth_m":  0.05,
                    "max_depth_m":  1.20,
                    "camera_frame": "camera_color_optical_frame",
                    "world_frame":  "world",
                    "patch_size":   7,
                }],
            ),
        ]),

        # ── 9. LQR Visual Servoing ────────────────────────────────────────────
        TimerAction(period=6.0, actions=[
            Node(
                package="so_arm100_pick_place",
                executable="lqr_visual_servoing.py",
                name="lqr_visual_servoing",
                output="screen",
                parameters=[{
                    "ee_link":     "tcp",
                    "world_frame": "world",
                    "log_csv":     LaunchConfiguration("log_csv"),
                    "auto_start":  True,
                }],
            ),
        ]),

        # ── 10. RViz2 ─────────────────────────────────────────────────────────
        TimerAction(period=4.0, actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_cfg],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]),
    ])
