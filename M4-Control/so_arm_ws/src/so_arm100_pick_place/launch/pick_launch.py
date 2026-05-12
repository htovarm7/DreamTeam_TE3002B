"""
pick_launch.py
==============
Launch mínimo para pick_demo.py:
  - realsense_node   → cámara RGB + depth
  - vision_detector  → detección CV → /vision/detections

NO incluye feetech_bridge ni move_group.
El control del brazo lo hace pick_demo.py directamente vía lerobot.

Uso:
  # Terminal 1:
  ros2 launch so_arm100_pick_place pick_launch.py

  # Terminal 2:
  ros2 run so_arm100_pick_place pick_demo.py          # automático
  ros2 run so_arm100_pick_place pick_demo.py --manual # paso a paso
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="false"),

        # ── 1. RealSense D435 ─────────────────────────────────────────────────
        Node(
            package="so_arm100_pick_place",
            executable="realsense_node.py",
            name="realsense_node",
            output="screen",
            parameters=[{"width": 640, "height": 480, "fps": 30}],
        ),

        # ── 2. Vision detector (espera 2s que arranque la cámara) ─────────────
        TimerAction(period=2.0, actions=[
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

        # ── 3. rqt viewers opcionales ─────────────────────────────────────────
        TimerAction(period=3.0, actions=[
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                name="rgb_view",
                arguments=["/camera/color/image_raw"],
                output="screen",
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]),
        TimerAction(period=3.0, actions=[
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                name="debug_view",
                arguments=["/vision/debug_image"],
                output="screen",
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]),
    ])
