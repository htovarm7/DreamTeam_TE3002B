#!/usr/bin/env python3
"""
pick_place_node.py
──────────────────
State-machine pick-and-place for SO-ARM100 using MoveIt2 Python API.

States:
  IDLE → MOVE_HOME → OPEN_GRIPPER → DETECT_OBJECT
       → MOVE_PRE_GRASP → MOVE_GRASP → CLOSE_GRIPPER
       → MOVE_POST_GRASP → MOVE_PLACE → OPEN_GRIPPER
       → MOVE_HOME → IDLE  (loop)

Object pose comes from /vision/object_position (PointStamped in camera frame).
TF is used to transform that point into the 'world' frame before planning.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 – registers PointStamped transform

# MoveIt2 Python bindings
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


# ── helpers ──────────────────────────────────────────────────────────────────

def make_pose(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0, frame="world") -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = z
    ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return ps


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    """ZYX Euler → quaternion."""
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


# ── State machine ─────────────────────────────────────────────────────────────

class State:
    IDLE            = "IDLE"
    MOVE_HOME       = "MOVE_HOME"
    OPEN_GRIPPER    = "OPEN_GRIPPER"
    DETECT_OBJECT   = "DETECT_OBJECT"
    MOVE_PRE_GRASP  = "MOVE_PRE_GRASP"
    MOVE_GRASP      = "MOVE_GRASP"
    CLOSE_GRIPPER   = "CLOSE_GRIPPER"
    MOVE_POST_GRASP = "MOVE_POST_GRASP"
    MOVE_PLACE      = "MOVE_PLACE"
    DONE            = "DONE"


class PickPlaceNode(Node):
    # Fixed place pose in world frame (blue pad in world)
    PLACE_POSE = make_pose(0.30, -0.15, 0.12,
                           **vars(euler_to_quat(0, math.pi / 2, 0).__class__
                                  .__new__(Quaternion).__dict__))

    # We rebuild PLACE_POSE properly:
    _PLACE_XYZ = (0.30, -0.15, 0.12)
    _PLACE_RPY = (0.0, math.pi / 2, 0.0)

    # Pre-grasp offset above the object (metres)
    PRE_GRASP_Z_OFFSET = 0.10
    # Grasp offset – TCP touches the top of the cube (cube = 4 cm, half = 2 cm)
    GRASP_Z_OFFSET     = 0.02

    def __init__(self):
        super().__init__("pick_place_node")

        # ── MoveIt2 ────────────────────────────────────────────────────────
        self.moveit = MoveItPy(node_name="pick_place_moveit")
        self.arm     = self.moveit.get_planning_component("arm")
        self.gripper = self.moveit.get_planning_component("gripper")

        # ── TF ─────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── State machine ──────────────────────────────────────────────────
        self.state        = State.IDLE
        self.object_world: PointStamped | None = None   # object in world frame

        # ── Subscriptions / publishers ─────────────────────────────────────
        self.create_subscription(PointStamped, "/vision/object_position",
                                 self._object_cb, 10)
        self.status_pub = self.create_publisher(String, "/pick_place/status", 10)

        # ── Timer: state machine tick at 2 Hz ─────────────────────────────
        self.create_timer(0.5, self._tick)

        self.get_logger().info("PickPlaceNode ready.")

    # ── Vision callback ───────────────────────────────────────────────────────

    def _object_cb(self, msg: PointStamped):
        """Transform detected point from camera frame → world frame."""
        try:
            self.object_world = self.tf_buffer.transform(
                msg, "world", timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=2.0)

    # ── State machine tick ────────────────────────────────────────────────────

    def _tick(self):
        self._pub_status(self.state)

        if self.state == State.IDLE:
            self.get_logger().info("Starting pick-and-place cycle…")
            self.state = State.MOVE_HOME

        elif self.state == State.MOVE_HOME:
            if self._move_named("home"):
                self.state = State.OPEN_GRIPPER

        elif self.state == State.OPEN_GRIPPER:
            if self._move_gripper_named("open"):
                self.state = State.DETECT_OBJECT

        elif self.state == State.DETECT_OBJECT:
            if self.object_world is not None:
                self.get_logger().info(
                    f"Object at world: "
                    f"x={self.object_world.point.x:.3f} "
                    f"y={self.object_world.point.y:.3f} "
                    f"z={self.object_world.point.z:.3f}"
                )
                self.state = State.MOVE_PRE_GRASP
            else:
                self.get_logger().info("Waiting for object detection…",
                                       throttle_duration_sec=2.0)

        elif self.state == State.MOVE_PRE_GRASP:
            p = self.object_world.point
            pose = self._grasp_pose(p.x, p.y, p.z + self.PRE_GRASP_Z_OFFSET)
            if self._move_to_pose(pose):
                self.state = State.MOVE_GRASP

        elif self.state == State.MOVE_GRASP:
            p = self.object_world.point
            pose = self._grasp_pose(p.x, p.y, p.z + self.GRASP_Z_OFFSET)
            if self._move_to_pose(pose, velocity_scaling=0.2):
                self.state = State.CLOSE_GRIPPER

        elif self.state == State.CLOSE_GRIPPER:
            if self._move_gripper_named("closed"):
                time.sleep(0.5)
                self.state = State.MOVE_POST_GRASP

        elif self.state == State.MOVE_POST_GRASP:
            p = self.object_world.point
            pose = self._grasp_pose(p.x, p.y, p.z + self.PRE_GRASP_Z_OFFSET)
            if self._move_to_pose(pose, velocity_scaling=0.3):
                self.state = State.MOVE_PLACE

        elif self.state == State.MOVE_PLACE:
            q = euler_to_quat(*self._PLACE_RPY)
            pose = make_pose(*self._PLACE_XYZ, qx=q.x, qy=q.y, qz=q.z, qw=q.w)
            if self._move_to_pose(pose):
                self.state = State.OPEN_GRIPPER
                # After placing, loop back to HOME for next cycle
                self._scheduled_next = State.MOVE_HOME

        elif self.state == State.DONE:
            self.get_logger().info("Cycle complete.", throttle_duration_sec=5.0)

    # ── MoveIt2 helpers ───────────────────────────────────────────────────────

    def _move_named(self, name: str) -> bool:
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name=name)
        result = self.arm.plan()
        if not result:
            self.get_logger().error(f"Planning failed for named state '{name}'")
            return False
        self.moveit.execute(result.trajectory, controllers=[])
        return True

    def _move_gripper_named(self, name: str) -> bool:
        self.gripper.set_start_state_to_current_state()
        self.gripper.set_goal_state(configuration_name=name)
        result = self.gripper.plan()
        if not result:
            self.get_logger().error(f"Gripper planning failed for '{name}'")
            return False
        self.moveit.execute(result.trajectory, controllers=[])
        return True

    def _move_to_pose(self, pose: PoseStamped,
                      velocity_scaling: float = 0.5) -> bool:
        pose.header.stamp = self.get_clock().now().to_msg()
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pose, pose_link="tcp")

        plan_params = self.arm.get_planning_parameters()
        plan_params.max_velocity_scaling_factor     = velocity_scaling
        plan_params.max_acceleration_scaling_factor = velocity_scaling * 0.5

        result = self.arm.plan(plan_params)
        if not result:
            self.get_logger().error("Cartesian planning failed")
            return False
        self.moveit.execute(result.trajectory, controllers=[])
        return True

    def _grasp_pose(self, x: float, y: float, z: float) -> PoseStamped:
        """TCP pointing down (gripper approaches from above)."""
        q = euler_to_quat(0.0, math.pi / 2, 0.0)
        return make_pose(x, y, z, qx=q.x, qy=q.y, qz=q.z, qw=q.w)

    # ── util ──────────────────────────────────────────────────────────────────

    def _pub_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
