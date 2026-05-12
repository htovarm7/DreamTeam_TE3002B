#!/usr/bin/env python3
"""
lqr_visual_servoing.py
======================
Position-Based Visual Servoing (PBVS) with LQR optimal control
for SO-ARM100 pick-and-place — compatible with ROS2 Humble.

Control pipeline:
  RealSense D435
    → vision_detector.py  → /vision/best_object  (PoseStamped, camera frame)
    → TF camera → world   → object position in world
    → LQR controller      → u* = −K e  (Cartesian velocity)
    → /compute_ik (MoveIt2 service) → joint positions
    → /arm_controller/follow_joint_trajectory (action) → brazo físico

LQR formulation:
  State:    e[k] = p_ee[k] − p_des[k]   (Cartesian error, ℝ³)
  Input:    u[k] = Δp[k]                 (Cartesian step, ℝ³)
  Dynamics: e[k+1] = e[k] + u[k]
  Cost:     J = Σ eᵀQe + uᵀRu
  Optimal:  u* = −K e,  K from DARE
"""

import csv
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState

import tf2_ros
import tf2_geometry_msgs  # noqa: F401


# ── Tuning ────────────────────────────────────────────────────────────────────

DT          = 0.15          # control step, seconds
Q_DIAG      = [10.0, 10.0, 15.0]
R_DIAG      = [1.0,  1.0,  1.0]
V_MAX       = 0.10          # max Cartesian step per iteration, m
THRESH_PRE  = 0.012         # pre-grasp convergence, m
THRESH_GRP  = 0.007         # grasp convergence, m
PRE_GRASP_Z = 0.10          # approach height above object, m
GRASP_Z     = 0.02          # grasp contact height, m
PLACE_XYZ   = np.array([0.30, -0.15, 0.20])

ARM_JOINTS     = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINTS = ["gripper"]


# ── Phases ────────────────────────────────────────────────────────────────────

class Phase:
    WAIT       = "WAIT"
    PRE_GRASP  = "PRE_GRASP"
    GRASP      = "GRASP"
    CLOSE_GRIP = "CLOSE_GRIP"
    POST_GRASP = "POST_GRASP"
    PLACE      = "PLACE"
    OPEN_GRIP  = "OPEN_GRIP"
    DONE       = "DONE"


# ── Node ──────────────────────────────────────────────────────────────────────

class LQRVisualServoingNode(Node):

    def __init__(self):
        super().__init__("lqr_visual_servoing")

        self.declare_parameter("ee_link",     "tcp")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("log_csv",     True)
        self.declare_parameter("auto_start",  True)

        self._ee_link     = self.get_parameter("ee_link").value
        self._world_frame = self.get_parameter("world_frame").value
        self._log_csv     = self.get_parameter("log_csv").value
        self._auto_start  = self.get_parameter("auto_start").value

        # ── LQR gain ───────────────────────────────────────────────────────
        self._K, self._P = self._build_lqr(np.diag(Q_DIAG), np.diag(R_DIAG))
        self.get_logger().info(
            f"[LQR] K (diag) = {np.diag(self._K).round(4).tolist()}\n"
            f"[LQR] Closed-loop eigenvalues = "
            f"{np.linalg.eigvals(np.eye(3) - self._K).round(4).tolist()}"
        )

        # ── State ──────────────────────────────────────────────────────────
        self._lock        = threading.Lock()
        self._obj_world   = None
        self._joint_state = None     # latest JointState
        self._phase       = Phase.WAIT
        self._t0          = time.monotonic()
        self._history     = []

        # ── TF ─────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Action clients ─────────────────────────────────────────────────
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory"
        )
        self._grip_client = ActionClient(
            self, FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory"
        )

        # ── IK service ─────────────────────────────────────────────────────
        self._ik_cli = self.create_client(GetPositionIK, "/compute_ik")

        # ── Subscriptions ──────────────────────────────────────────────────
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=2,
        )
        self.create_subscription(PoseStamped, "/vision/best_object", self._vision_cb, be)
        self.create_subscription(JointState,  "/joint_states",        self._js_cb,     10)

        # ── Publishers ─────────────────────────────────────────────────────
        self._pub_status  = self.create_publisher(String,            "/vs/status",  10)
        self._pub_metrics = self.create_publisher(Float64MultiArray, "/vs/metrics", 10)

        # ── CSV ─────────────────────────────────────────────────────────────
        self._csv_writer = None
        self._csv_fh     = None
        if self._log_csv:
            log_dir  = Path.home() / "vs_logs"
            log_dir.mkdir(exist_ok=True)
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = log_dir / f"vs_{ts}.csv"
            self._csv_fh     = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_fh)
            self._csv_writer.writerow([
                "t_s", "ex_m", "ey_m", "ez_m", "err_norm_m",
                "vx_ms", "vy_ms", "vz_ms", "v_norm_ms", "phase",
            ])
            self.get_logger().info(f"[LQR] Logging to {csv_path}")

        # ── Control thread ──────────────────────────────────────────────────
        self._ctl = threading.Thread(target=self._control_loop, daemon=True)
        self._ctl.start()

        self.get_logger().info("[LQR] Visual Servoing Node ready (Humble, no MoveItPy).")

    # ── LQR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_lqr(Q, R):
        A = np.eye(3)
        B = np.eye(3)
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return K, P

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _vision_cb(self, msg: PoseStamped):
        try:
            pt = PointStamped()
            # Usar Time() = transform más reciente disponible (evita extrapolation error)
            pt.header.frame_id = msg.header.frame_id
            pt.header.stamp    = rclpy.time.Time().to_msg()
            pt.point           = msg.pose.position
            pt_world = self._tf_buf.transform(
                pt, self._world_frame, timeout=Duration(seconds=0.1)
            )
            p = np.array([pt_world.point.x, pt_world.point.y, pt_world.point.z])
            with self._lock:
                self._obj_world = p
        except Exception as e:
            self.get_logger().warn(f"[LQR] TF fail: {e}", throttle_duration_sec=2.0)

    def _js_cb(self, msg: JointState):
        with self._lock:
            self._joint_state = msg

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        while rclpy.ok():
            t0 = time.monotonic()
            try:
                self._step()
            except Exception as e:
                self.get_logger().error(f"[LQR] step error: {e}")
            time.sleep(max(0.0, DT - (time.monotonic() - t0)))

    def _step(self):
        with self._lock:
            obj   = self._obj_world.copy() if self._obj_world is not None else None
            phase = self._phase

        self._pub_status.publish(String(data=phase))

        if phase == Phase.WAIT:
            if obj is not None and self._auto_start:
                self.get_logger().info(f"[LQR] Object at {obj.round(3)} → PRE_GRASP")
                with self._lock:
                    self._phase = Phase.PRE_GRASP
                    self._t0    = time.monotonic()
            return

        if obj is None:
            self.get_logger().warn("[LQR] Object lost.", throttle_duration_sec=2.0)
            return

        if phase == Phase.PRE_GRASP:
            done = self._lqr_step(obj + np.array([0.0, 0.0, PRE_GRASP_Z]), THRESH_PRE, phase)
            if done:
                self.get_logger().info("[LQR] Pre-grasp → GRASP")
                with self._lock:
                    self._phase = Phase.GRASP

        elif phase == Phase.GRASP:
            done = self._lqr_step(obj + np.array([0.0, 0.0, GRASP_Z]), THRESH_GRP, phase)
            if done:
                self.get_logger().info("[LQR] Grasp → CLOSE_GRIP")
                with self._lock:
                    self._phase = Phase.CLOSE_GRIP

        elif phase == Phase.CLOSE_GRIP:
            self._gripper_move(0.8)   # closed ≈ 0.8 rad
            time.sleep(0.8)
            with self._lock:
                self._phase = Phase.POST_GRASP

        elif phase == Phase.POST_GRASP:
            done = self._lqr_step(obj + np.array([0.0, 0.0, PRE_GRASP_Z]), THRESH_PRE, phase)
            if done:
                with self._lock:
                    self._phase = Phase.PLACE

        elif phase == Phase.PLACE:
            done = self._lqr_step(PLACE_XYZ, THRESH_PRE, phase)
            if done:
                with self._lock:
                    self._phase = Phase.OPEN_GRIP

        elif phase == Phase.OPEN_GRIP:
            self._gripper_move(0.0)   # open ≈ 0 rad
            time.sleep(0.5)
            self._flush_csv()
            self.get_logger().info("[LQR] Cycle done → WAIT")
            with self._lock:
                self._phase     = Phase.WAIT
                self._obj_world = None

    # ── LQR step ──────────────────────────────────────────────────────────────

    def _lqr_step(self, target: np.ndarray, threshold: float, phase: str) -> bool:
        p_ee = self._get_ee_pos()
        if p_ee is None:
            return False

        e = p_ee - target
        u = -self._K @ e

        norm = np.linalg.norm(u)
        if norm > V_MAX:
            u *= V_MAX / norm

        p_next = p_ee + u

        self._command_cartesian(p_next)

        e_norm = float(np.linalg.norm(e))
        self._log(e, u, e_norm, phase)
        return e_norm < threshold

    # ── EE position and orientation via TF ───────────────────────────────────

    def _get_ee_tf(self):
        """Returns (position, quaternion) of TCP in world frame, or (None, None)."""
        try:
            tf = self._tf_buf.lookup_transform(
                self._world_frame, self._ee_link,
                rclpy.time.Time(), timeout=Duration(seconds=0.05),
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            return np.array([t.x, t.y, t.z]), r
        except Exception as e:
            self.get_logger().warn(f"[LQR] EE TF fail: {e}", throttle_duration_sec=2.0)
            return None, None

    def _get_ee_pos(self):
        pos, _ = self._get_ee_tf()
        return pos

    # ── Cartesian command via IK + trajectory ─────────────────────────────────

    def _command_cartesian(self, pos: np.ndarray):
        with self._lock:
            js = self._joint_state

        # Usar la orientación actual del TCP — 5-DOF no soporta orientación arbitraria
        _, current_rot = self._get_ee_tf()
        if current_rot is None:
            return

        joints = self._compute_ik(pos, js, current_rot)
        if joints is None:
            self.get_logger().warn("[LQR] IK failed, skipping step.", throttle_duration_sec=1.0)
            return

        self._send_arm_trajectory(joints, duration_sec=DT * 1.2)

    def _compute_ik(self, pos: np.ndarray, seed_js: JointState, orientation=None):
        if not self._ik_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("[LQR] /compute_ik not available.", throttle_duration_sec=5.0)
            return None

        # Orientación natural del robot al-zeros (0, 0.707, 0, 0.707)
        target_orientation = orientation if orientation is not None \
            else _euler_to_quat(0.0, math.pi / 2, 0.0)

        # Probar con seed actual Y con seed en zeros — LMA necesita buen punto inicial
        seeds_to_try = []
        if seed_js is not None:
            seeds_to_try.append(seed_js)
        # Seed en zeros (posición natural del modelo URDF)
        zero_seed = JointState()
        zero_seed.name     = list(ARM_JOINTS)
        zero_seed.position = [0.0] * len(ARM_JOINTS)
        seeds_to_try.append(zero_seed)

        for seed in seeds_to_try:
            req = GetPositionIK.Request()
            req.ik_request = PositionIKRequest()
            req.ik_request.group_name      = "arm"
            req.ik_request.ik_link_name    = self._ee_link
            req.ik_request.timeout.sec     = 0
            req.ik_request.timeout.nanosec = int(0.5 * 1e9)
            req.ik_request.avoid_collisions = False

            ps = PoseStamped()
            ps.header.frame_id = self._world_frame
            ps.header.stamp    = self.get_clock().now().to_msg()
            ps.pose.position.x = float(pos[0])
            ps.pose.position.y = float(pos[1])
            ps.pose.position.z = float(pos[2])
            ps.pose.orientation = target_orientation
            req.ik_request.pose_stamped = ps

            req.ik_request.robot_state = RobotState()
            req.ik_request.robot_state.joint_state = seed

            future = self._ik_cli.call_async(req)
            deadline = time.monotonic() + 1.2
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.005)

            if not future.done():
                continue  # intenta con otro seed

            resp = future.result()
            if resp.error_code.val == 1:
                js = resp.solution.joint_state
                name_to_pos = dict(zip(js.name, js.position))
                try:
                    result = [name_to_pos[j] for j in ARM_JOINTS]
                    return result
                except KeyError:
                    continue

            self.get_logger().warn(f"[LQR] IK error_code={resp.error_code.val} (seed retry)",
                                   throttle_duration_sec=2.0)

        return None

    def _send_arm_trajectory(self, positions: list, duration_sec: float):
        if not self._arm_client.wait_for_server(timeout_sec=1.0):
            return

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions  = [float(p) for p in positions]
        pt.velocities = [0.0] * len(positions)
        sec  = int(duration_sec)
        nsec = int((duration_sec - sec) * 1e9)
        pt.time_from_start = RosDuration(sec=sec, nanosec=nsec)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._arm_client.send_goal_async(goal)

    def _gripper_move(self, position: float):
        if not self._grip_client.wait_for_server(timeout_sec=1.0):
            return

        traj = JointTrajectory()
        traj.joint_names = GRIPPER_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions  = [position]
        pt.velocities = [0.0]
        pt.time_from_start = RosDuration(sec=1, nanosec=0)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._grip_client.send_goal_async(goal)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, e, u, e_norm, phase):
        t = round(time.monotonic() - self._t0, 4)
        row = [
            t,
            round(float(e[0]), 6), round(float(e[1]), 6), round(float(e[2]), 6),
            round(e_norm, 6),
            round(float(u[0]), 6), round(float(u[1]), 6), round(float(u[2]), 6),
            round(float(np.linalg.norm(u)), 6),
            phase,
        ]
        self._history.append(row)
        msg = Float64MultiArray()
        msg.data = [float(v) for v in row[:9]]
        self._pub_metrics.publish(msg)

    def _flush_csv(self):
        if self._csv_writer and self._history:
            for row in self._history:
                self._csv_writer.writerow(row)
            if self._csv_fh:
                self._csv_fh.flush()
            self.get_logger().info(f"[LQR] Wrote {len(self._history)} rows to CSV.")
            self._history.clear()

    def destroy_node(self):
        self._flush_csv()
        if self._csv_fh:
            self._csv_fh.close()
        super().destroy_node()


# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LQRVisualServoingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
