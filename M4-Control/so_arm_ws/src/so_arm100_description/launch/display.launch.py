from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("so_arm100_description")
    urdf_file = PathJoinSubstitution([pkg, "urdf", "so_arm100.urdf.xacro"])
    rviz_config = PathJoinSubstitution([pkg, "rviz", "so_arm100.rviz"])

    robot_description = {"robot_description": Command(["xacro ", urdf_file])}

    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[robot_description],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
        ),
    ])
