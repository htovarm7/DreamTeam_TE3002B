#!/usr/bin/env python3
"""
joint_commander.py
──────────────────
Send the SO-ARM100 to named positions or explicit joint angles.

Named positions (defined below, can be overridden via ROS params):
  home       — all zeros
  ready      — slightly raised, neutral
  pre_grasp  — arm reaching forward-down
  open       — gripper open
  closed     — gripper closed

Usage examples
──────────────
# Move to home via CLI argument:
ros2 run so_arm100_pick_place joint_commander.py --ros-args -p pose:=home

# Move to a custom joint config (radians, 5 arm joints):
ros2 run so_arm100_pick_place joint_commander.py \
  --ros-args -p joints:="[0.5, -0.4, 0.8, 0.0, 0.0]"

# Interactive mode (no params) — reads from /joint_commander/target topic:
ros2 run so_arm100_pick_place joint_commander.py
ros2 topic pub /joint_commander/target std_msgs/msg/String "data: home"
ros2 topic pub /joint_commander/target std_msgs/msg/String "data: 0.5,-0.4,0.8,0.0,0.0"
"""

import sys
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration


# ── Named positions (radians) ─────────────────────────────────────────────────

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_JOINTS = ["gripper_left_joint", "gripper_right_joint"]

NAMED_ARM_POSES = {
    "home":      [0.0,   0.0,  0.0,   0.0,  0.0],
    "ready":     [0.0,  -0.5,  0.8,  -0.3,  0.0],
    "pre_grasp": [0.0,  -0.8,  1.2,  -0.4,  0.0],
    "left":      [ 1.57, 0.0,  0.0,   0.0,  0.0],
    "right":     [-1.57, 0.0,  0.0,   0.0,  0.0],
    "up":        [0.0,  -1.0,  0.0,   0.0,  0.0],
}

NAMED_GRIPPER_POSES = {
    "open":   [-0.025, -0.025],
    "closed": [ 0.0,    0.0],
}


# ── Node ─────────────────────────────────────────────────────────────────────

class JointCommander(Node):

    def __init__(self):
        super().__init__("joint_commander")

        # Params
        self.declare_parameter("pose",     "")        # named pose
        self.declare_parameter("joints",   [0.0]*5)   # explicit arm joints (rad)
        self.declare_parameter("duration", 2.0)       # trajectory duration (s)
        self.declare_parameter("interactive", True)   # listen to topic

        # Action clients
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory"
        )
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory"
        )

        # Topic interface for interactive use
        self.create_subscription(String, "/joint_commander/target",
                                 self._target_cb, 10)

        # Status feedback
        self.status_pub = self.create_publisher(String, "/joint_commander/status", 10)

        self.get_logger().info("JointCommander ready.")
        self.get_logger().info(
            "Named arm poses: " + ", ".join(NAMED_ARM_POSES.keys()))
        self.get_logger().info(
            "Named gripper poses: " + ", ".join(NAMED_GRIPPER_POSES.keys()))
        self.get_logger().info(
            "Or subscribe to /joint_commander/target with e.g. 'home' or '0.5,-0.4,0.8,0.0,0.0'")

        # One-shot execution if params were given
        self._startup_timer = self.create_timer(1.0, self._startup_once)

    # ── startup one-shot ──────────────────────────────────────────────────────

    def _startup_once(self):
        self._startup_timer.cancel()

        pose_name = self.get_parameter("pose").get_parameter_value().string_value
        joints    = self.get_parameter("joints").get_parameter_value().double_array_value

        if pose_name:
            self._execute_named(pose_name)
        elif any(j != 0.0 for j in joints):
            self._execute_arm_joints(list(joints))

    # ── topic callback ────────────────────────────────────────────────────────

    def _target_cb(self, msg: String):
        text = msg.data.strip()
        self.get_logger().info(f"Received target: '{text}'")

        # Named pose?
        if text in NAMED_ARM_POSES or text in NAMED_GRIPPER_POSES:
            self._execute_named(text)
            return

        # Comma-separated floats → arm joints
        try:
            values = [float(v.strip()) for v in text.split(",")]
            if len(values) == 5:
                self._execute_arm_joints(values)
            elif len(values) == 2:
                self._execute_gripper_joints(values)
            else:
                self.get_logger().error(
                    f"Expected 5 arm joints or 2 gripper joints, got {len(values)}")
        except ValueError:
            self.get_logger().error(
                f"Unknown pose '{text}'. "
                f"Use a name ({', '.join({**NAMED_ARM_POSES, **NAMED_GRIPPER_POSES})}) "
                f"or comma-separated radians.")

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _execute_named(self, name: str):
        if name in NAMED_ARM_POSES:
            self.get_logger().info(f"Moving arm to named pose: '{name}'")
            self._execute_arm_joints(NAMED_ARM_POSES[name])
        elif name in NAMED_GRIPPER_POSES:
            self.get_logger().info(f"Moving gripper to named pose: '{name}'")
            self._execute_gripper_joints(NAMED_GRIPPER_POSES[name])
        else:
            self.get_logger().error(f"Unknown named pose: '{name}'")

    # ── arm trajectory ────────────────────────────────────────────────────────

    def _execute_arm_joints(self, positions: list):
        if len(positions) != 5:
            self.get_logger().error(f"Need 5 joint values, got {len(positions)}")
            return

        self.get_logger().info(
            "Arm joints (rad): " +
            "  ".join(f"j{i+1}={v:.4f}" for i, v in enumerate(positions))
        )
        self.get_logger().info(
            "Arm joints (deg): " +
            "  ".join(f"j{i+1}={math.degrees(v):.1f}°" for i, v in enumerate(positions))
        )

        dur = self.get_parameter("duration").get_parameter_value().double_value
        traj = self._make_trajectory(ARM_JOINTS, positions, dur)
        self._send_trajectory(self._arm_client, traj, "arm")

    # ── gripper trajectory ────────────────────────────────────────────────────

    def _execute_gripper_joints(self, positions: list):
        dur = self.get_parameter("duration").get_parameter_value().double_value
        traj = self._make_trajectory(GRIPPER_JOINTS, positions, dur)
        self._send_trajectory(self._gripper_client, traj, "gripper")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_trajectory(self, joint_names: list, positions: list,
                         duration_sec: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions  = [float(p) for p in positions]
        pt.velocities = [0.0] * len(positions)
        secs = int(duration_sec)
        nsecs = int((duration_sec - secs) * 1e9)
        pt.time_from_start = RosDuration(sec=secs, nanosec=nsecs)

        traj.points = [pt]
        return traj

    def _send_trajectory(self, client: ActionClient,
                         traj: JointTrajectory, label: str):
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f"{label} action server not available. "
                "Is the simulation running and controllers loaded?")
            self._pub_status(f"ERROR: {label} controller unavailable")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self._pub_status(f"MOVING: {label}")
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._goal_response_cb(f, label))

    def _goal_response_cb(self, future, label: str):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error(f"{label} goal rejected by controller")
            self._pub_status(f"REJECTED: {label}")
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda f: self._result_cb(f, label))

    def _result_cb(self, future, label: str):
        result = future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"✓ {label} reached target position.")
            self._pub_status(f"DONE: {label}")
        else:
            self.get_logger().error(
                f"{label} trajectory failed: {result.error_string}")
            self._pub_status(f"FAILED: {label} — {result.error_string}")

    def _pub_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


# ── main ─────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = JointCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
