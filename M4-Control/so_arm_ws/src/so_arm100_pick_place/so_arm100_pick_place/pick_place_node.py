#!/usr/bin/env python3
"""
pick_place_node.py
──────────────────
State-machine pick-and-place para SO-ARM100 — compatible con ROS2 Humble.
Usa /compute_ik (MoveIt2 service) + FollowJointTrajectory (ros2_control).

States: IDLE → HOME → OPEN_GRIPPER → DETECT
      → PRE_GRASP → GRASP → CLOSE_GRIPPER
      → POST_GRASP → PLACE → OPEN_GRIPPER → HOME (loop)
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState

import tf2_ros
import tf2_geometry_msgs  # noqa: F401


ARM_JOINTS     = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINTS = ["gripper"]

PRE_GRASP_Z = 0.10
GRASP_Z     = 0.02
PLACE_XYZ   = (0.30, -0.15, 0.12)


def _euler_to_quat(roll, pitch, yaw) -> Quaternion:
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


class State:
    IDLE         = "IDLE"
    HOME         = "HOME"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    DETECT       = "DETECT"
    PRE_GRASP    = "PRE_GRASP"
    GRASP        = "GRASP"
    CLOSE_GRIP   = "CLOSE_GRIP"
    POST_GRASP   = "POST_GRASP"
    PLACE        = "PLACE"
    DONE         = "DONE"


class PickPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_place_node")

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)

        self._arm_client  = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self._grip_client = ActionClient(
            self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )
        self._ik_cli = self.create_client(GetPositionIK, "/compute_ik")

        self._js: JointState | None          = None
        self._object_world: PointStamped | None = None
        self._state  = State.IDLE
        self._busy   = False

        self.create_subscription(PointStamped, "/vision/best_object",
                                 self._object_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self._status_pub = self.create_publisher(String, "/pick_place/status", 10)

        self.create_timer(0.5, self._tick)
        self.get_logger().info("PickPlaceNode ready (Humble, IK+trajectory).")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        self._js = msg

    def _object_cb(self, msg: PointStamped):
        try:
            self._object_world = self._tf_buf.transform(
                msg, "world", timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF fail: {e}", throttle_duration_sec=2.0)

    # ── State machine ──────────────────────────────────────────────────────────

    def _tick(self):
        if self._busy:
            return
        self._pub_status(self._state)

        if self._state == State.IDLE:
            self._state = State.HOME

        elif self._state == State.HOME:
            self._move_joints([0.0, 0.0, 0.0, 0.0, 0.0], duration=2.0)
            self._state = State.OPEN_GRIPPER

        elif self._state == State.OPEN_GRIPPER:
            self._move_gripper(0.0)
            self._state = State.DETECT

        elif self._state == State.DETECT:
            if self._object_world is not None:
                self.get_logger().info(
                    f"Object: x={self._object_world.point.x:.3f} "
                    f"y={self._object_world.point.y:.3f} "
                    f"z={self._object_world.point.z:.3f}"
                )
                self._state = State.PRE_GRASP
            else:
                self.get_logger().info("Waiting for object…", throttle_duration_sec=2.0)

        elif self._state == State.PRE_GRASP:
            p = self._object_world.point
            ok = self._move_cartesian(p.x, p.y, p.z + PRE_GRASP_Z, duration=2.5)
            if ok:
                self._state = State.GRASP

        elif self._state == State.GRASP:
            p = self._object_world.point
            ok = self._move_cartesian(p.x, p.y, p.z + GRASP_Z, duration=2.0)
            if ok:
                self._state = State.CLOSE_GRIP

        elif self._state == State.CLOSE_GRIP:
            self._move_gripper(0.8)
            time.sleep(0.5)
            self._state = State.POST_GRASP

        elif self._state == State.POST_GRASP:
            p = self._object_world.point
            ok = self._move_cartesian(p.x, p.y, p.z + PRE_GRASP_Z, duration=2.5)
            if ok:
                self._state = State.PLACE

        elif self._state == State.PLACE:
            x, y, z = PLACE_XYZ
            ok = self._move_cartesian(x, y, z, duration=3.0)
            if ok:
                self._state = State.OPEN_GRIPPER

        elif self._state == State.DONE:
            self.get_logger().info("Cycle complete.", throttle_duration_sec=5.0)

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _move_cartesian(self, x: float, y: float, z: float,
                        duration: float = 2.0) -> bool:
        q = _euler_to_quat(0.0, math.pi / 2, 0.0)
        ps = PoseStamped()
        ps.header.frame_id  = "world"
        ps.header.stamp     = self.get_clock().now().to_msg()
        ps.pose.position.x  = x
        ps.pose.position.y  = y
        ps.pose.position.z  = z
        ps.pose.orientation = q

        joints = self._compute_ik(ps)
        if joints is None:
            self.get_logger().error(f"IK failed for ({x:.3f},{y:.3f},{z:.3f})")
            return False

        self._move_joints(joints, duration=duration)
        return True

    def _compute_ik(self, pose: PoseStamped) -> list | None:
        if not self._ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/compute_ik not available")
            return None

        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name       = "arm"
        req.ik_request.ik_link_name     = "tcp"
        req.ik_request.pose_stamped     = pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec      = 0
        req.ik_request.timeout.nanosec  = int(0.2 * 1e9)

        if self._js is not None:
            req.ik_request.robot_state = RobotState()
            req.ik_request.robot_state.joint_state = self._js

        future = self._ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

        if not future.done() or future.result().error_code.val != 1:
            return None

        js = future.result().solution.joint_state
        name_pos = dict(zip(js.name, js.position))
        try:
            return [name_pos[j] for j in ARM_JOINTS]
        except KeyError:
            return None

    def _move_joints(self, positions: list, duration: float = 2.0):
        if not self._arm_client.wait_for_server(timeout_sec=2.0):
            return
        traj = _make_traj(ARM_JOINTS, positions, duration)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=duration + 1.0)

    def _move_gripper(self, position: float, duration: float = 1.0):
        if not self._grip_client.wait_for_server(timeout_sec=2.0):
            return
        traj = _make_traj(GRIPPER_JOINTS, [position], duration)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._grip_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=duration + 1.0)

    def _pub_status(self, text: str):
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_traj(joint_names: list, positions: list, duration: float) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = joint_names
    pt = JointTrajectoryPoint()
    pt.positions  = [float(p) for p in positions]
    pt.velocities = [0.0] * len(positions)
    sec  = int(duration)
    nsec = int((duration - sec) * 1e9)
    pt.time_from_start = RosDuration(sec=sec, nanosec=nsec)
    traj.points = [pt]
    return traj


# ── Entry point ───────────────────────────────────────────────────────────────

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
