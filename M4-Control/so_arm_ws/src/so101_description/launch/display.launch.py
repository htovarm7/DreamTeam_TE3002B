"""
display.launch.py
─────────────────
Visualiza el SO-ARM101 en RViz2 con joint_state_publisher_gui.
Útil para verificar el URDF antes de lanzar Gazebo o el robot físico.

Uso:
  ros2 launch so101_description display.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg   = get_package_share_directory("so101_description")
    urdf_xacro = os.path.join(desc_pkg, "urdf", "so101.urdf.xacro")
    rviz_cfg   = os.path.join(desc_pkg, "rviz", "so101.rviz")

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", urdf_xacro, " use_sim:=false"]),
            value_type=str,
        )
    }

    return LaunchDescription([
        DeclareLaunchArgument("rviz_config", default_value=rviz_cfg),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),

        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
        ),
    ])
