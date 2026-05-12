"""
sync.launch.py
──────────────
Simulación Gazebo + robot físico en SYNC.

Arquitectura:
  • Gazebo Classic corre como master — física completa, controladores normales.
  • feetech_bridge se suscribe a /joint_states (publicado por joint_state_broadcaster
    de Gazebo) y envía esas posiciones a los motores STS3215 físicos.
  • El robot físico sigue al robot simulado en tiempo real.
  • Las posiciones reales se publican en /physical_joint_states para diagnóstico.

Uso:
  ros2 launch so101_bringup sync.launch.py
  ros2 launch so101_bringup sync.launch.py port:=/dev/ttyACM1 gui:=false
"""

import os
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg   = get_package_share_directory("so101_description")
    urdf_xacro = os.path.join(desc_pkg, "urdf", "so101.urdf.xacro")
    rviz_cfg   = os.path.join(desc_pkg, "rviz", "so101.rviz")

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", urdf_xacro, " use_sim:=true"]),
            value_type=str,
        )
    }

    return LaunchDescription([
        DeclareLaunchArgument("gui",  default_value="true",
                              description="Lanzar ventana gráfica de Gazebo"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Lanzar RViz2"),
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0",
                              description="Puerto serie de los motores Feetech"),

        # ── 1. Gazebo Classic ────────────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare("gazebo_ros"), "/launch/gazebo.launch.py"
            ]),
            launch_arguments={
                "verbose": "false",
                "pause":   "false",
                "gui":     LaunchConfiguration("gui"),
            }.items(),
        ),

        # ── 2. robot_state_publisher ─────────────────────────────────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description, {"use_sim_time": True}],
        ),

        # ── 3. Spawn robot ───────────────────────────────────────────────────
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-topic", "robot_description",
                "-entity", "so101",
                "-x", "0.0", "-y", "0.0", "-z", "0.0",
            ],
            output="screen",
        ),

        # ── 4a. joint_state_broadcaster ──────────────────────────────────────
        TimerAction(period=3.0, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 4b. arm_controller ───────────────────────────────────────────────
        TimerAction(period=4.0, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["arm_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 4c. gripper_controller ───────────────────────────────────────────
        TimerAction(period=4.5, actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["gripper_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen",
            ),
        ]),

        # ── 5. feetech_bridge (SYNC) ─────────────────────────────────────────
        # Se suscribe a /joint_states (de Gazebo) y comanda los motores físicos.
        # Publica estados reales en /physical_joint_states para comparación.
        TimerAction(period=5.0, actions=[
            Node(
                package="so101_bringup",
                executable="feetech_bridge.py",
                name="feetech_bridge",
                output="screen",
                parameters=[{
                    "port":              LaunchConfiguration("port"),
                    "cmd_topic":         "joint_states",
                    "state_topic":       "physical_joint_states",
                    "publish_rate":      20.0,
                    "deadband_deg":      0.3,
                }],
            ),
        ]),

        # ── 6. RViz2 ─────────────────────────────────────────────────────────
        TimerAction(period=5.0, actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", rviz_cfg],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]),
    ])
