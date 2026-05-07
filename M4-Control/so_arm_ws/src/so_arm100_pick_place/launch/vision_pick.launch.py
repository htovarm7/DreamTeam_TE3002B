"""
vision_pick.launch.py
──────────────────────
Launches the full vision + RViz pipeline for SO-ARM100 pick-and-place.

What starts:
  1. robot_state_publisher  — publishes /robot_description + TF from URDF
  2. realsense_node         — D435 RGB + aligned depth topics
  3. vision_detector        — HSV detector → /vision/markers + /vision/best_object
  4. rviz2                  — shows robot, object markers, camera images

Usage:
  ros2 launch so_arm100_pick_place vision_pick.launch.py

Optional args:
  rviz:=false       skip RViz (headless)
  port:=/dev/ttyACM0
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg  = get_package_share_directory("so_arm100_description")
    pick_pkg  = get_package_share_directory("so_arm100_pick_place")

    urdf_file   = os.path.join(desc_pkg, "urdf", "so_arm100.urdf.xacro")
    rviz_config = os.path.join(pick_pkg, "rviz", "vision_pick.rviz")

    robot_description = {"robot_description": Command(["xacro ", urdf_file])}

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch RViz2"),
        DeclareLaunchArgument("port", default_value="/dev/ttyACM0",
                              description="Serial port for the arm"),

        # ── 1. Robot state publisher (TF + /robot_description) ───────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),

        # ── 2. RealSense D435 ─────────────────────────────────────────────────
        Node(
            package="so_arm100_pick_place",
            executable="realsense_node.py",
            name="realsense_node",
            output="screen",
            parameters=[{
                "width":  640,
                "height": 480,
                "fps":    30,
            }],
        ),

        # ── 3. Vision detector (slight delay for camera to start) ─────────────
        TimerAction(period=2.0, actions=[
            Node(
                package="so_arm100_pick_place",
                executable="vision_detector.py",
                name="vision_detector",
                output="screen",
                parameters=[{
                    "min_depth_m":  0.10,
                    "max_depth_m":  1.20,
                    "camera_frame": "camera_color_optical_frame",
                }],
            ),
        ]),

        # ── 4. RViz2 ──────────────────────────────────────────────────────────
        TimerAction(period=3.0, actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]),
    ])
