from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="so_arm100_pick_place",
            executable="vision_node.py",
            name="vision_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="so_arm100_pick_place",
            executable="pick_place_node.py",
            name="pick_place_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
    ])
