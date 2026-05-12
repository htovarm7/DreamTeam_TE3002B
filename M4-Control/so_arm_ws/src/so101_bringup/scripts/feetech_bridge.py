#!/usr/bin/env python3
"""
feetech_bridge.py
─────────────────
Bridge entre ros2_control y los motores Feetech STS3215 del SO-ARM101.

Modos de operación:
  • physical (cmd_topic = joint_states_mock):
      Escucha los comandos del mock_components y los ejecuta en los motores.
      Publica las posiciones REALES en /joint_states.

  • sync    (cmd_topic = joint_states):
      Escucha /joint_states de Gazebo y espeja esas posiciones en los motores.
      Publica las posiciones reales en /physical_joint_states (diagnóstico).

Parámetros ROS:
  port          (str)   Puerto serie, ej. /dev/ttyACM0
  cmd_topic     (str)   Topic de entrada con JointState (posición objetivo)
  state_topic   (str)   Topic donde se publican los estados reales del hardware
  publish_rate  (float) Hz de lectura/publicación del hardware
  deadband_deg  (float) Mínima diferencia (grados) para enviar un nuevo comando
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from lerobot.motors import MotorCalibration, MotorNormMode, Motor
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

# ── Calibración de fábrica ─────────────────────────────────────────────────────
# Ajusta homing_offset y rangos según tu robot específico
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

ARM_JOINTS    = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
ALL_JOINTS    = ARM_JOINTS + [GRIPPER_JOINT]

# ── Conversiones ───────────────────────────────────────────────────────────────

def _rad_to_deg(r: float) -> float:
    return math.degrees(r)

def _deg_to_rad(d: float) -> float:
    return math.radians(d)

def _gripper_rad_to_pct(r: float) -> float:
    return max(0.0, min(100.0, (r / 1.74533) * 100.0))

def _pct_to_gripper_rad(p: float) -> float:
    return (p / 100.0) * 1.74533


class FeetechBridge(Node):

    def __init__(self):
        super().__init__("feetech_bridge")

        self.declare_parameter("port",         "/dev/ttyACM0")
        self.declare_parameter("cmd_topic",    "joint_states_mock")
        self.declare_parameter("state_topic",  "joint_states")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("deadband_deg", 0.3)

        self._port        = self.get_parameter("port").value
        self._cmd_topic   = self.get_parameter("cmd_topic").value
        self._state_topic = self.get_parameter("state_topic").value
        self._deadband    = self.get_parameter("deadband_deg").value
        self._lock        = threading.Lock()
        self._last_cmd    = {j: None for j in ALL_JOINTS}

        # ── Conectar al bus Feetech ────────────────────────────────────────
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
            self.get_logger().info(
                f"[FeetechBridge] Conectado en {self._port} | "
                f"cmd: /{self._cmd_topic} → state: /{self._state_topic}"
            )
        except Exception as e:
            self.get_logger().error(f"[FeetechBridge] Fallo de conexión: {e}")
            return

        # ── ROS I/O ───────────────────────────────────────────────────────
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=2)

        self.create_subscription(JointState, self._cmd_topic, self._cmd_cb, be)
        self._js_pub = self.create_publisher(JointState, self._state_topic, 10)
        self.create_timer(
            1.0 / self.get_parameter("publish_rate").value,
            self._publish_real,
        )

        self.get_logger().info("[FeetechBridge] Bridge listo.")

    # ── Recepción de comandos ─────────────────────────────────────────────────

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
                self.get_logger().error(
                    f"[FeetechBridge] write error: {e}",
                    throttle_duration_sec=2.0,
                )

    # ── Publicación de estados reales ─────────────────────────────────────────

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

    # ── Ir a home seguro antes de apagar ─────────────────────────────────────

    def go_home(self):
        if not self._connected:
            return
        home = {j: 0.0 for j in ARM_JOINTS}
        home[GRIPPER_JOINT] = 0.0
        print("[FeetechBridge] Ctrl+C → volviendo a HOME (0°)...")
        try:
            with self._lock:
                self._bus.sync_write("Goal_Position", home)
            for _ in range(30):
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            print("[FeetechBridge] ✓ En HOME.")
        except Exception as e:
            print(f"[FeetechBridge] Error volviendo a HOME: {e}")

    def destroy_node(self):
        if self._connected:
            try:
                self._bus.disable_torque()
                self._bus.disconnect()
                self.get_logger().info("[FeetechBridge] Desconectado.")
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FeetechBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.go_home()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
