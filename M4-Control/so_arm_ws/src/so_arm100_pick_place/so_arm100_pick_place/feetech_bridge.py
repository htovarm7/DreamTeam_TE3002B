#!/usr/bin/env python3
"""
feetech_bridge.py
─────────────────
Bridge entre ros2_control (mock_components) y los motores Feetech STS3215.

Publica posiciones REALES en /joint_states (MoveIt2 y RViz ven el robot real).
Al arrancar va a 'detections'. Al apagar (Ctrl+C) va a 'home' antes de cortar torque.
"""

import json
import math
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState

from lerobot.motors import MotorCalibration, MotorNormMode, Motor
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ── Config ────────────────────────────────────────────────────────────────────

PORT = "/dev/ttyACM0"

def _poses_file() -> Path:
    """Busca poses.json en el share del paquete (funciona con ros2 run y ros2 launch)."""
    try:
        return Path(get_package_share_directory("so_arm100_pick_place")) / "poses.json"
    except Exception:
        return Path(__file__).parent / "poses.json"

ARM_JOINTS    = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
ALL_JOINTS    = ARM_JOINTS + [GRIPPER_JOINT]

CALIBRATION = {
    "shoulder_pan":  MotorCalibration(id=1, drive_mode=0, homing_offset=-2017,
                                      range_min=2036, range_max=2053),
    "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=-1451,
                                      range_min=790,  range_max=2049),
    "elbow_flex":    MotorCalibration(id=3, drive_mode=0, homing_offset=1293,
                                      range_min=2047, range_max=3157),
    "wrist_flex":    MotorCalibration(id=4, drive_mode=0, homing_offset=-1629,
                                      range_min=791,  range_max=2805),
    "wrist_roll":    MotorCalibration(id=5, drive_mode=0, homing_offset=-402,
                                      range_min=0,    range_max=4095),
    "gripper":       MotorCalibration(id=6, drive_mode=0, homing_offset=1164,
                                      range_min=2032, range_max=3555),
}


def _rad_to_deg(r):  return math.degrees(r)
def _deg_to_rad(d):  return math.radians(d)
def _gripper_rad_to_pct(r): return max(0.0, min(100.0, (r / 1.74533) * 100.0))
def _pct_to_gripper_rad(p): return (p / 100.0) * 1.74533


class FeetechBridge(Node):

    def __init__(self):
        super().__init__("feetech_bridge")

        self.declare_parameter("port",         PORT)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("deadband_deg", 0.3)

        self._port     = self.get_parameter("port").value
        self._deadband = self.get_parameter("deadband_deg").value
        self._lock     = threading.Lock()
        self._last_cmd = {j: None for j in ALL_JOINTS}

        # ── Cargar poses ───────────────────────────────────────────────────
        self._poses: dict = {}
        poses_file = _poses_file()
        if poses_file.exists():
            self._poses = json.loads(poses_file.read_text())
            self.get_logger().info(f"[Feetech] Poses cargadas de {poses_file}: {list(self._poses.keys())}")
        else:
            self.get_logger().warn(f"[Feetech] No se encontró poses.json en {poses_file}")

        # ── Conectar bus ───────────────────────────────────────────────────
        self._bus = FeetechMotorsBus(
            port=self._port,
            motors={
                "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=CALIBRATION,
        )

        self._connected = False
        try:
            self._bus.connect()
            self._bus.write_calibration(CALIBRATION)
            for motor in self._bus.motors:
                self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self._bus.enable_torque()
            self._connected = True
            self.get_logger().info(f"[Feetech] Conectado en {self._port} — torque ON")
        except Exception as e:
            self.get_logger().error(f"[Feetech] Fallo conexión: {e}")
            return

        # ── IR A DETECTIONS al arrancar (timer de un disparo a los 2 s) ───
        self._startup_done = False
        self._startup_timer = self.create_timer(2.0, self._startup_cb)

        # ── ROS ────────────────────────────────────────────────────────────
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=2)

        self.create_subscription(JointState, "/joint_states_mock", self._cmd_cb, 10)
        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / self.get_parameter("publish_rate").value,
                          self._publish_real)

        self.get_logger().info("[Feetech] Bridge listo.")

    # ── Startup: ir a detections una sola vez ─────────────────────────────────

    def _startup_cb(self):
        if self._startup_done:
            return
        self._startup_done = True
        self._startup_timer.cancel()   # disparo único

        pose_name = "detections" if "detections" in self._poses else (
                    "home"       if "home"       in self._poses else None)

        if pose_name is None:
            self.get_logger().warn("[Feetech] No hay pose 'detections' ni 'home' en poses.json")
            return

        self.get_logger().info(f"[Feetech] → moviendo a '{pose_name}'...")
        try:
            with self._lock:
                self._bus.sync_write("Goal_Position", self._poses[pose_name])
            self.get_logger().info(f"[Feetech] ✓ En posición '{pose_name}'")
        except Exception as e:
            self.get_logger().error(f"[Feetech] Error startup: {e}")

    # ── Ejecuta comandos del mock en hardware real ────────────────────────────

    def _cmd_cb(self, msg: JointState):
        if not self._connected:
            return

        name_pos = dict(zip(msg.name, msg.position))
        goal: dict[str, float] = {}

        for j in ARM_JOINTS:
            if j not in name_pos:
                continue
            deg  = _rad_to_deg(name_pos[j])
            last = self._last_cmd[j]
            if last is None or abs(deg - last) >= self._deadband:
                goal[j] = deg
                self._last_cmd[j] = deg

        if GRIPPER_JOINT in name_pos:
            pct  = _gripper_rad_to_pct(name_pos[GRIPPER_JOINT])
            last = self._last_cmd[GRIPPER_JOINT]
            if last is None or abs(pct - last) >= self._deadband:
                goal[GRIPPER_JOINT] = pct
                self._last_cmd[GRIPPER_JOINT] = pct

        if goal:
            try:
                with self._lock:
                    self._bus.sync_write("Goal_Position", goal)
            except Exception as e:
                self.get_logger().error(f"[Feetech] write error: {e}",
                                        throttle_duration_sec=2.0)

    # ── Publica posición real en /joint_states ────────────────────────────────

    def _publish_real(self):
        if not self._connected:
            return
        try:
            with self._lock:
                pos = self._bus.sync_read("Present_Position")
        except Exception:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for j in ARM_JOINTS:
            if j in pos:
                msg.name.append(j)
                msg.position.append(_deg_to_rad(pos[j]))
                msg.velocity.append(0.0)
        if GRIPPER_JOINT in pos:
            msg.name.append(GRIPPER_JOINT)
            msg.position.append(_pct_to_gripper_rad(pos[GRIPPER_JOINT]))
            msg.velocity.append(0.0)
        if msg.name:
            self._js_pub.publish(msg)

    # ── Ir a home (bloquea hasta llegar) ─────────────────────────────────────

    def go_to_home(self):
        if not self._connected or "home" not in self._poses:
            return
        print("[Feetech] Ctrl+C → moviendo a HOME...")  # print en vez de logger (ya cerrado)
        try:
            with self._lock:
                self._bus.sync_write("Goal_Position", self._poses["home"])
            # sleep sin interrupción
            for _ in range(25):
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            print("[Feetech] ✓ En HOME.")
        except Exception as e:
            print(f"[Feetech] Error home: {e}")

    def destroy_node(self):
        if self._connected:
            try:
                self._bus.disable_torque()
                self._bus.disconnect()
                self.get_logger().info("[Feetech] Desconectado.")
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FeetechBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C → ir a home antes de apagar
        node.go_to_home()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
