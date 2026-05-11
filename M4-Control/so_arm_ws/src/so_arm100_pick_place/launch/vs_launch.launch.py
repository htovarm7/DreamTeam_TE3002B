"""
vs_launch.launch.py
====================
Complete LQR Visual Servoing pipeline for SO-ARM100.

What starts:
  1. robot_state_publisher   — URDF → TF tree
  2. realsense_node          — D435 RGB + aligned-depth topics
  3. vision_detector         — classical CV: HSV + contours + image moments
                               → /vision/best_object (PoseStamped, camera frame)
  4. lqr_visual_servoing     — PBVS LQR controller (this project's contribution)
                               reads /vision/best_object, solves DARE, drives arm
  5. RViz2                   — live visualisation (markers + debug image + TF)

For simulation (Gazebo must already be running with sim.launch.py):
  ros2 launch so_arm100_pick_place vs_launch.launch.py sim:=true

For real hardware (RealSense + physical arm):
  ros2 launch so_arm100_pick_place vs_launch.launch.py

LQR parameters (can be overridden on command line):
  q_xy        position error weight (x and y)  default 10.0
  q_z         position error weight (z)         default 15.0
  r           control effort weight             default 1.0
  v_max       Cartesian velocity limit (m/s)    default 0.12
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg = get_package_share_directory("so_arm100_description")
    pick_pkg = get_package_share_directory("so_arm100_pick_place")

    urdf_file   = os.path.join(desc_pkg, "urdf", "so_arm100.urdf.xacro")
    rviz_config = os.path.join(pick_pkg, "rviz", "vision_pick.rviz")

    robot_description = {"robot_description": Command(["xacro ", urdf_file])}

    return LaunchDescription([
        # ── Launch arguments ─────────────────────────────────────────────────
        DeclareLaunchArgument("rviz",    default_value="true",
                              description="Launch RViz2"),
        DeclareLaunchArgument("sim",     default_value="false",
                              description="true = Gazebo simulation (skip RealSense)"),
        DeclareLaunchArgument("port",    default_value="/dev/ttyACM0",
                              description="Serial port for physical arm"),
        DeclareLaunchArgument("log_csv", default_value="true",
                              description="Save LQR metrics to ~/vs_logs/*.csv"),

        # ── 1. Robot state publisher (TF + /robot_description) ───────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
            condition=UnlessCondition(LaunchConfiguration("sim")),
        ),

        # ── 2. RealSense D435 (hardware only) ────────────────────────────────
        Node(
            package="so_arm100_pick_place",
            executable="realsense_node.py",
            name="realsense_node",
            output="screen",
            parameters=[{"width": 640, "height": 480, "fps": 30}],
            condition=UnlessCondition(LaunchConfiguration("sim")),
        ),

        # ── 3. Classical CV vision detector ──────────────────────────────────
        #    HSV thresholding → morphological ops → contour detection →
        #    image moments (centroid) → depth backprojection → 3-D position
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
                    "world_frame":  "world",
                    "patch_size":   7,
                }],
            ),
        ]),

        # ── 4. LQR Visual Servoing controller ────────────────────────────────
        #    Subscribes: /vision/best_object
        #    Computes:   DARE → gain K → u* = −K e
        #    Commands:   incremental Cartesian goals → MoveIt2 → arm
        TimerAction(period=4.0, actions=[
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

        # ── 5. RViz2 ──────────────────────────────────────────────────────────
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
