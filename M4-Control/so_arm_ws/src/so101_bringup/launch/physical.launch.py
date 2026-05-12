"""
physical.launch.py
──────────────────
Lanza el SO-ARM101 físico sin Gazebo.

Stack:
  1. robot_state_publisher   — URDF → árbol TF
  2. controller_manager      — ros2_control con mock_components
     (joint_states → joint_states_mock para que feetech_bridge sea el
      único publicador de /joint_states reales)
  3. joint_state_broadcaster — publica /joint_states_mock
  4. arm_controller          — JointTrajectoryController (brazo)
  5. gripper_controller      — JointTrajectoryController (gripper)
  6. feetech_bridge          — espeja comandos del mock en motores STS3215
                               y publica la posición REAL en /joint_states

Uso:
  ros2 launch so101_bringup physical.launch.py
  ros2 launch so101_bringup physical.launch.py port:=/dev/ttyACM1
  ros2 launch so101_bringup physical.launch.py rviz:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg   = get_package_share_directory("so101_description")
    urdf_xacro = os.path.join(desc_pkg, "urdf", "so101.urdf.xacro")
    ctrl_yaml  = os.path.join(desc_pkg, "config", "ros2_controllers.yaml")
    rviz_cfg   = os.path.join(desc_pkg, "rviz", "so101.rviz")

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", urdf_xacro, " use_sim:=false"]),
            value_type=str,
        )
    }

    return LaunchDescription([
        DeclareLaunchArgument("port",  default_value="/dev/ttyACM0",
                              description="Puerto serie de los motores Feetech"),
        DeclareLaunchArgument("rviz",  default_value="false",
                              description="Lanzar RViz2"),

        # ── 1. robot_state_publisher ─────────────────────────────────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description, {"use_sim_time": False}],
        ),

        # ── 2. controller_manager (mock hardware) ────────────────────────────
        # Remapeamos joint_states → joint_states_mock a nivel de nodo;
        # feetech_bridge es el único publicador de /joint_states reales.
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[robot_description, ctrl_yaml, {"use_sim_time": False}],
            remappings=[("joint_states", "joint_states_mock")],
        ),

        # ── 3. joint_state_broadcaster ───────────────────────────────────────
        TimerAction(period=2.0, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 4. arm_controller ────────────────────────────────────────────────
        TimerAction(period=3.0, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["arm_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 5. gripper_controller ────────────────────────────────────────────
        TimerAction(period=3.5, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["gripper_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 6. feetech_bridge ────────────────────────────────────────────────
        # Se suscribe a /joint_states_mock y publica posiciones reales en /joint_states
        Node(
            package="so101_bringup",
            executable="feetech_bridge.py",
            name="feetech_bridge",
            output="screen",
            parameters=[{
                "port":         LaunchConfiguration("port"),
                "cmd_topic":    "joint_states_mock",
                "publish_rate": 20.0,
                "deadband_deg": 0.3,
            }],
        ),

        # ── 7. RViz2 (opcional) ──────────────────────────────────────────────
        TimerAction(period=4.0, actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", rviz_cfg],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]),
    ])
