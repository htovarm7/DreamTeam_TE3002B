from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import os
import yaml


def load_yaml(pkg, filepath):
    pkg_path = FindPackageShare(pkg).find(pkg)
    abs_path = os.path.join(pkg_path, filepath)
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    desc_pkg = FindPackageShare("so_arm100_description")
    moveit_pkg = FindPackageShare("so_arm100_moveit_config")

    urdf_file = PathJoinSubstitution([desc_pkg, "urdf", "so_arm100.urdf.xacro"])
    srdf_file = PathJoinSubstitution([moveit_pkg, "config", "so_arm100.srdf"])

    robot_description = {"robot_description": ParameterValue(Command(["xacro ", urdf_file]), value_type=str)}
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(Command(["cat ", srdf_file]), value_type=str)
    }

    kinematics_yaml      = load_yaml("so_arm100_moveit_config", "config/kinematics.yaml")
    ompl_planning_yaml   = load_yaml("so_arm100_moveit_config", "config/ompl_planning.yaml")
    joint_limits_yaml    = load_yaml("so_arm100_moveit_config", "config/joint_limits.yaml")
    controllers_yaml     = load_yaml("so_arm100_moveit_config", "config/moveit_controllers.yaml")

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml["moveit_simple_controller_manager"],
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                robot_description,
                robot_description_semantic,
                {"robot_description_kinematics": kinematics_yaml},
                {"robot_description_planning": {"joint_limits": joint_limits_yaml}},
                ompl_planning_yaml,
                moveit_controllers,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"publish_robot_description_semantic": True},
            ],
        ),
    ])
