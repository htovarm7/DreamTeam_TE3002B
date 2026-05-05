"""
test_joints.launch.py
─────────────────────
Launch the joint_commander in interactive mode.

After launching, send commands from another terminal:

  # Named poses
  ros2 topic pub --once /joint_commander/target std_msgs/msg/String "data: home"
  ros2 topic pub --once /joint_commander/target std_msgs/msg/String "data: ready"
  ros2 topic pub --once /joint_commander/target std_msgs/msg/String "data: pre_grasp"
  ros2 topic pub --once /joint_commander/target std_msgs/msg/String "data: open"
  ros2 topic pub --once /joint_commander/target std_msgs/msg/String "data: closed"

  # Custom arm joints (radians): j1, j2, j3, j4, j5
  ros2 topic pub --once /joint_commander/target std_msgs/msg/String "data: 0.5,-0.4,0.8,0.0,0.0"

Or launch with a one-shot pose:
  ros2 launch so_arm100_pick_place test_joints.launch.py pose:=home
  ros2 launch so_arm100_pick_place test_joints.launch.py pose:=ready duration:=3.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("pose",     default_value="",    description="Named pose to move to at startup"),
        DeclareLaunchArgument("joints",   default_value="",    description="Comma-separated joint values (rad) e.g. 0.0,0.0,0.0,0.0,0.0"),
        DeclareLaunchArgument("duration", default_value="2.0", description="Trajectory duration in seconds"),

        Node(
            package="so_arm100_pick_place",
            executable="joint_commander.py",
            name="joint_commander",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "pose":     LaunchConfiguration("pose"),
                "duration": LaunchConfiguration("duration"),
                "interactive": True,
            }],
        ),
    ])
