import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    desc_pkg   = get_package_share_directory("so_arm100_description")
    gazebo_pkg = get_package_share_directory("so_arm100_gazebo")

    urdf_file  = os.path.join(desc_pkg,   "urdf",   "so_arm100.urdf.xacro")
    world_file = os.path.join(gazebo_pkg, "worlds", "pick_place.world")

    robot_description = {
        "robot_description": ParameterValue(Command(["xacro ", urdf_file]), value_type=str)
    }

    # ── Gazebo ──────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file, "verbose": "false", "pause": "false"}.items(),
    )

    # ── robot_state_publisher ────────────────────────────────────
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # ── Spawn robot ──────────────────────────────────────────────
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "so_arm100"],
        output="screen",
    )

    # ── Controllers (spawner, waits for controller_manager) ─────
    jsb = TimerAction(period=3.0, actions=[Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )])

    arm_ctrl = TimerAction(period=4.0, actions=[Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )])

    gripper_ctrl = TimerAction(period=5.0, actions=[Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        gazebo,
        rsp,
        spawn,
        jsb,
        arm_ctrl,
        gripper_ctrl,
    ])
