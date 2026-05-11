#!/usr/bin/env python3
"""
lqr_visual_servoing.py
======================
Position-Based Visual Servoing (PBVS) with LQR optimal control
for SO-ARM100 pick-and-place using classical computer vision.

╔══════════════════════════════════════════════════════════════════╗
║  VISUAL SERVOING TASK DEFINITION                                 ║
║                                                                  ║
║  Goal: drive the robot end-effector to a 3-D position that is   ║
║  determined in real time from camera measurements, so that the  ║
║  arm can grasp an object whose location is not known a priori.  ║
║  The robot must regulate the Cartesian error between its TCP     ║
║  and the visually-detected target to zero in closed loop.       ║
╠══════════════════════════════════════════════════════════════════╣
║  CLASSICAL VISION PIPELINE (vision_detector.py)                  ║
║                                                                  ║
║  1. HSV color thresholding  → binary mask                       ║
║  2. Morphological open/close → noise removal                    ║
║  3. cv2.findContours        → object boundary extraction        ║
║  4. Shape filters (aspect ratio, circularity)                    ║
║  5. Image moments m00,m10,m01 → 2-D centroid (cx, cy)          ║
║  6. Aligned depth patch (7×7 median) → depth Z                  ║
║  7. Pinhole backprojection  → 3-D point in camera frame         ║
║     X = (cx − ppx)·Z / fx                                      ║
║     Y = (cy − ppy)·Z / fy                                      ║
║                                                                  ║
║  Visual feature vector: s = (X_obj, Y_obj, Z_obj) ∈ ℝ³         ║
╠══════════════════════════════════════════════════════════════════╣
║  OPTIMAL CONTROL FORMULATION                                     ║
║                                                                  ║
║  State (task-space error):                                       ║
║    x[k] = e[k] = p_ee[k] − p_des[k] ∈ ℝ³                      ║
║    p_des[k] = s[k] + δ_phase  (visually updated every step)    ║
║    δ_phase ∈ {[0,0,0.10], [0,0,0.02]} m (pre-grasp / grasp)    ║
║                                                                  ║
║  Control input:                                                  ║
║    u[k] = v[k] ∈ ℝ³   end-effector Cartesian velocity (m/s)   ║
║                                                                  ║
║  Linear discrete-time dynamics (Euler, step dt):                ║
║    x[k+1] = A x[k] + B u[k]                                    ║
║    A = I₃      (error persists without control action)          ║
║    B = dt · I₃ (velocity command integrates to position)        ║
║                                                                  ║
║  Constraints:                                                    ║
║    ‖u[k]‖ ≤ V_MAX  (Cartesian velocity safety limit)           ║
║    joint limits enforced by MoveIt2 planner                     ║
║                                                                  ║
║  Infinite-horizon LQR cost:                                      ║
║    J = Σ_{k=0}^∞ ( x[k]ᵀ Q x[k]  +  u[k]ᵀ R u[k] )          ║
║    Q = diag(q_x, q_y, q_z)  ≻ 0   position error weight       ║
║    R = diag(r_x, r_y, r_z)  ≻ 0   control effort weight       ║
║                                                                  ║
║  Optimal gain via Discrete Algebraic Riccati Equation (DARE):   ║
║    P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA                       ║
║    K = (R + BᵀPB)⁻¹ BᵀPA                                      ║
║    u*[k] = −K x[k]   (optimal LQR control law)                 ║
║                                                                  ║
║  Closed-loop error dynamics:                                     ║
║    x[k+1] = (A − BK) x[k]     → asymptotically stable         ║
╠══════════════════════════════════════════════════════════════════╣
║  PIPELINE                                                        ║
║                                                                  ║
║  RealSense D435                                                  ║
║    │  RGB + aligned depth                                        ║
║    ▼                                                             ║
║  vision_detector.py  (HSV → moments → backproject)              ║
║    │  /vision/best_object  (PoseStamped, camera frame)          ║
║    ▼                                                             ║
║  TF: camera_color_optical_frame → world                         ║
║    │  s = (X_obj, Y_obj, Z_obj)                                 ║
║    ▼                                                             ║
║  LQR controller  (this node, 10 Hz)                             ║
║    │  e = p_ee − p_des(s)                                       ║
║    │  u* = −K e                                                 ║
║    │  p_next = p_ee + u* · dt                                   ║
║    ▼                                                             ║
║  MoveIt2 (plan + execute incremental Cartesian goal)            ║
║    ▼                                                             ║
║  SO-ARM100 joints (Gazebo / physical Feetech servos)            ║
║    ▼                                                             ║
║  Camera observes new arm + object state → loop                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import csv
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Quaternion, PointStamped
from std_msgs.msg import String, Float64MultiArray

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform

from moveit.planning import MoveItPy


# ── Tuning parameters ─────────────────────────────────────────────────────────

DT      = 0.10          # control step, seconds (10 Hz loop)
Q_DIAG  = [10.0, 10.0, 15.0]   # position error weights [x, y, z]
R_DIAG  = [1.0,  1.0,  1.0]    # velocity effort weights
V_MAX   = 0.12          # maximum Cartesian velocity, m/s
THRESH_PRE  = 0.008     # pre-grasp convergence threshold, m  (8 mm)
THRESH_GRP  = 0.005     # grasp convergence threshold, m  (5 mm)
PRE_GRASP_Z = 0.10      # approach height above object centroid, m
GRASP_Z     = 0.02      # grasp height (half of 4 cm cube), m
PLACE_XYZ   = np.array([0.30, -0.15, 0.20])   # fixed place pose (world frame)


# ── Servoing phases ───────────────────────────────────────────────────────────

class Phase:
    WAIT        = "WAIT"
    PRE_GRASP   = "PRE_GRASP"
    GRASP       = "GRASP"
    CLOSE_GRIP  = "CLOSE_GRIP"
    POST_GRASP  = "POST_GRASP"
    PLACE       = "PLACE"
    OPEN_GRIP   = "OPEN_GRIP"
    DONE        = "DONE"


# ── Node ──────────────────────────────────────────────────────────────────────

class LQRVisualServoingNode(Node):
    """
    Position-Based Visual Servoing with infinite-horizon LQR.

    Subscribes  : /vision/best_object  (PoseStamped — from vision_detector.py)
    Publishes   : /vs/status   (String — current phase)
                  /vs/metrics  (Float64MultiArray — [t, ex, ey, ez, |e|,
                                                       vx, vy, vz, |v|])
    Commands    : MoveIt2 incremental Cartesian goals at DT Hz
    """

    def __init__(self):
        super().__init__("lqr_visual_servoing")

        # ── ROS parameters ─────────────────────────────────────────────────
        self.declare_parameter("ee_link",     "tcp")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("log_csv",     True)
        self.declare_parameter("auto_start",  True)   # start on first detection

        self._ee_link     = self.get_parameter("ee_link").value
        self._world_frame = self.get_parameter("world_frame").value
        self._log_csv     = self.get_parameter("log_csv").value
        self._auto_start  = self.get_parameter("auto_start").value

        # ── LQR: solve DARE once at startup ────────────────────────────────
        self._K, self._P = self._build_lqr(np.diag(Q_DIAG), np.diag(R_DIAG), DT)
        self.get_logger().info(
            f"\n[LQR] System matrices — A=I₃, B={DT}·I₃\n"
            f"[LQR] Q = diag{Q_DIAG}\n"
            f"[LQR] R = diag{R_DIAG}\n"
            f"[LQR] DARE solution P (diag) = {np.diag(self._P).round(4).tolist()}\n"
            f"[LQR] Optimal gain K (diag)  = {np.diag(self._K).round(4).tolist()}\n"
            f"[LQR] Closed-loop A-BK eigenvalues = "
            f"{np.linalg.eigvals(np.eye(3) - DT * self._K).round(4).tolist()}"
        )

        # ── Shared state (thread-safe) ──────────────────────────────────────
        self._lock         = threading.Lock()
        self._obj_world    = None          # latest object position (world frame)
        self._phase        = Phase.WAIT
        self._t0           = time.monotonic()
        self._history      = []            # rows for CSV

        # ── MoveIt2 ────────────────────────────────────────────────────────
        self._moveit  = MoveItPy(node_name="lqr_vs_moveit")
        self._arm     = self._moveit.get_planning_component("arm")
        self._gripper = self._moveit.get_planning_component("gripper")

        # ── TF ─────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Topics ─────────────────────────────────────────────────────────
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=2,
        )
        self.create_subscription(
            PoseStamped, "/vision/best_object", self._vision_cb, be
        )
        self._pub_status  = self.create_publisher(String,            "/vs/status",  10)
        self._pub_metrics = self.create_publisher(Float64MultiArray, "/vs/metrics", 10)

        # ── CSV log setup ───────────────────────────────────────────────────
        self._csv_writer = None
        self._csv_fh     = None
        if self._log_csv:
            log_dir = Path.home() / "vs_logs"
            log_dir.mkdir(exist_ok=True)
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = log_dir / f"vs_{ts}.csv"
            self._csv_fh     = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_fh)
            self._csv_writer.writerow([
                "t_s", "ex_m", "ey_m", "ez_m", "err_norm_m",
                "vx_ms", "vy_ms", "vz_ms", "v_norm_ms", "phase",
            ])
            self.get_logger().info(f"[LQR] Logging metrics to {csv_path}")

        # ── Control loop thread ─────────────────────────────────────────────
        self._ctl_thread = threading.Thread(
            target=self._control_loop, daemon=True
        )
        self._ctl_thread.start()

        self.get_logger().info(
            "[LQR] Visual Servoing Node ready — waiting for /vision/best_object …"
        )

    # ── LQR setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_lqr(Q: np.ndarray, R: np.ndarray, dt: float):
        """
        Compute discrete LQR gain K via Discrete Algebraic Riccati Equation.

        State-space model of the PBVS error dynamics:
          e[k+1] = A e[k] + B u[k]
          A = I₃           (error is constant without control)
          B = dt · I₃      (velocity u integrates into position change)

        Optimal cost-to-go satisfies DARE:
          P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA

        Optimal state-feedback gain:
          K = (R + BᵀPB)⁻¹ BᵀPA

        Optimal control law:
          u*[k] = −K e[k]
        """
        A = np.eye(3)
        B = dt * np.eye(3)
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return K, P

    # ── Vision callback ───────────────────────────────────────────────────────

    def _vision_cb(self, msg: PoseStamped):
        """
        Receive latest object pose from vision_detector.py and transform
        from camera optical frame to world frame.
        """
        try:
            pt = PointStamped()
            pt.header = msg.header
            pt.point  = msg.pose.position
            pt_world  = self._tf_buf.transform(
                pt, self._world_frame, timeout=Duration(seconds=0.05)
            )
            p = np.array([pt_world.point.x, pt_world.point.y, pt_world.point.z])
            with self._lock:
                self._obj_world = p
        except Exception as exc:
            self.get_logger().warn(
                f"[LQR] TF camera→world failed: {exc}",
                throttle_duration_sec=2.0,
            )

    # ── Control loop (runs in its own thread at DT Hz) ────────────────────────

    def _control_loop(self):
        while rclpy.ok():
            t_start = time.monotonic()
            try:
                self._step()
            except Exception as exc:
                self.get_logger().error(f"[LQR] Control step error: {exc}")
            elapsed = time.monotonic() - t_start
            time.sleep(max(0.0, DT - elapsed))

    def _step(self):
        with self._lock:
            obj  = self._obj_world.copy() if self._obj_world is not None else None
            phase = self._phase

        self._pub_status.publish(String(data=phase))

        # ── WAIT: do nothing until camera detects an object ─────────────────
        if phase == Phase.WAIT:
            if obj is not None and self._auto_start:
                self.get_logger().info(
                    f"[LQR] Object detected at {obj.round(3)} m — starting VS."
                )
                with self._lock:
                    self._phase = Phase.PRE_GRASP
                    self._t0    = time.monotonic()
            return

        if obj is None:
            self.get_logger().warn(
                "[LQR] Object lost — holding position.",
                throttle_duration_sec=2.0,
            )
            return

        # ── PRE_GRASP: LQR drives arm to 10 cm above object ─────────────────
        if phase == Phase.PRE_GRASP:
            target = obj + np.array([0.0, 0.0, PRE_GRASP_Z])
            done   = self._lqr_step(target, THRESH_PRE, phase)
            if done:
                self.get_logger().info("[LQR] Pre-grasp reached → GRASP.")
                with self._lock:
                    self._phase = Phase.GRASP

        # ── GRASP: LQR drives arm to grasp contact point ─────────────────────
        elif phase == Phase.GRASP:
            target = obj + np.array([0.0, 0.0, GRASP_Z])
            done   = self._lqr_step(target, THRESH_GRP, phase)
            if done:
                self.get_logger().info("[LQR] Grasp position reached → CLOSE_GRIP.")
                with self._lock:
                    self._phase = Phase.CLOSE_GRIP

        # ── CLOSE_GRIP ────────────────────────────────────────────────────────
        elif phase == Phase.CLOSE_GRIP:
            self._gripper_named("closed")
            time.sleep(0.6)
            with self._lock:
                self._phase = Phase.POST_GRASP

        # ── POST_GRASP: lift back to pre-grasp height (LQR with fixed object) ─
        elif phase == Phase.POST_GRASP:
            target = obj + np.array([0.0, 0.0, PRE_GRASP_Z])
            done   = self._lqr_step(target, THRESH_PRE, phase)
            if done:
                with self._lock:
                    self._phase = Phase.PLACE

        # ── PLACE: move to fixed drop pose (no vision needed) ────────────────
        elif phase == Phase.PLACE:
            done = self._lqr_step(PLACE_XYZ, THRESH_PRE, phase)
            if done:
                with self._lock:
                    self._phase = Phase.OPEN_GRIP

        # ── OPEN_GRIP: release object ─────────────────────────────────────────
        elif phase == Phase.OPEN_GRIP:
            self._gripper_named("open")
            time.sleep(0.4)
            self._flush_csv()
            self.get_logger().info(
                "[LQR] Cycle complete — resetting for next object."
            )
            with self._lock:
                self._phase     = Phase.WAIT
                self._obj_world = None   # force re-detection

    # ── LQR control step ──────────────────────────────────────────────────────

    def _lqr_step(self, target: np.ndarray, threshold: float, phase: str) -> bool:
        """
        One iteration of the LQR visual servoing loop.

        1. Read current end-effector position from TF (forward kinematics).
        2. Compute task-space error: e = p_ee − p_des.
        3. Apply LQR: u* = −K e.
        4. Clamp to safety velocity limit.
        5. Compute next EE target: p_next = p_ee + u* · dt.
        6. Send incremental Cartesian goal to MoveIt2.
        7. Log metrics.

        Returns True when ‖e‖ < threshold (task converged).
        """
        p_ee = self._get_ee_pos()
        if p_ee is None:
            return False

        # Task-space error (state vector)
        e = p_ee - target

        # Optimal LQR control: u* = −K e
        u = -self._K @ e

        # Velocity safety clamp
        u_norm = np.linalg.norm(u)
        if u_norm > V_MAX:
            u = u * (V_MAX / u_norm)

        # Next incremental Cartesian target
        p_next = p_ee + u * DT

        # Command arm via MoveIt2
        self._command_cartesian(p_next)

        # Logging
        e_norm = float(np.linalg.norm(e))
        self._log(e, u, e_norm, phase)

        return e_norm < threshold

    # ── Arm / gripper commands ────────────────────────────────────────────────

    def _get_ee_pos(self) -> np.ndarray | None:
        """Current TCP position in world frame via TF lookup."""
        try:
            tf = self._tf_buf.lookup_transform(
                self._world_frame, self._ee_link,
                rclpy.time.Time(), timeout=Duration(seconds=0.05),
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception as exc:
            self.get_logger().warn(
                f"[LQR] EE TF lookup failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _command_cartesian(self, pos: np.ndarray):
        """Send an incremental Cartesian pose goal to MoveIt2."""
        q  = _euler_to_quat(0.0, math.pi / 2, 0.0)   # gripper pointing down
        ps = PoseStamped()
        ps.header.frame_id    = self._world_frame
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = float(pos[0])
        ps.pose.position.y    = float(pos[1])
        ps.pose.position.z    = float(pos[2])
        ps.pose.orientation   = q

        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(pose_stamped_msg=ps, pose_link=self._ee_link)
        plan_params = self._arm.get_planning_parameters()
        plan_params.max_velocity_scaling_factor     = 0.6
        plan_params.max_acceleration_scaling_factor = 0.4
        result = self._arm.plan(plan_params)
        if result:
            self._moveit.execute(result.trajectory, controllers=[])

    def _gripper_named(self, name: str):
        self._gripper.set_start_state_to_current_state()
        self._gripper.set_goal_state(configuration_name=name)
        result = self._gripper.plan()
        if result:
            self._moveit.execute(result.trajectory, controllers=[])

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, e: np.ndarray, u: np.ndarray, e_norm: float, phase: str):
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

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
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
