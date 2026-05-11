"""
Main simulation launch:
  - Gazebo with pick_place.world
  - robot_state_publisher
  - spawn SO-ARM100
  - ros2_control controllers
  - MoveIt2 move_group
  - RViz2
"""
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    desc_pkg    = get_package_share_directory("so_arm100_description")
    gazebo_pkg  = get_package_share_directory("so_arm100_gazebo")
    moveit_pkg  = get_package_share_directory("so_arm100_moveit_config")

    urdf_file  = os.path.join(desc_pkg, "urdf", "so_arm100.urdf.xacro")
    world_file = os.path.join(gazebo_pkg, "worlds", "pick_place.world")
    ctrl_yaml  = os.path.join(gazebo_pkg, "config", "ros2_controllers.yaml")

    robot_description_content = ParameterValue(Command(["xacro ", urdf_file]), value_type=str)
    robot_description = {"robot_description": robot_description_content}

    # ── Gazebo ──────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py"
            )
        ),
        launch_arguments={
            "world": world_file,
            "verbose": "false",
            "pause": "false",
        }.items(),
    )

    # ── robot_state_publisher ────────────────────────────────
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # ── Spawn robot in Gazebo ────────────────────────────────
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "so_arm100",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.0",
        ],
        output="screen",
    )

    # ── ros2_control: load controllers after spawn ───────────
    load_jsb = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "active",
             "joint_state_broadcaster"],
        output="screen",
    )
    load_arm = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "active",
             "arm_controller"],
        output="screen",
    )
    load_gripper = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "active",
             "gripper_controller"],
        output="screen",
    )

    # Chain: spawn → jsb → arm → gripper
    after_spawn_jsb     = RegisterEventHandler(OnProcessExit(
        target_action=spawn_robot, on_exit=[load_jsb]))
    after_jsb_arm       = RegisterEventHandler(OnProcessExit(
        target_action=load_jsb,    on_exit=[load_arm]))
    after_arm_gripper   = RegisterEventHandler(OnProcessExit(
        target_action=load_arm,    on_exit=[load_gripper]))

    # ── MoveIt2 move_group (delayed 5 s to let controllers settle) ──
    move_group = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_pkg, "launch", "move_group.launch.py")
                ),
                launch_arguments={"use_sim_time": "true"}.items(),
            )
        ],
    )

    # ── RViz2 ────────────────────────────────────────────────
    rviz_config = os.path.join(desc_pkg, "rviz", "so_arm100.rviz")
    rviz = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        gazebo,
        rsp,
        spawn_robot,
        after_spawn_jsb,
        after_jsb_arm,
        after_arm_gripper,
        move_group,
        rviz,
    ])
