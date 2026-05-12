#!/usr/bin/env python3
"""
pick_demo.py
────────────
Pick & place usando poses pre-enseñadas + confirmación visual.
No requiere IK ni MoveIt2 — usa lerobot directamente.

Poses requeridas en poses.json:
  detections  — brazo apartado, cámara ve el área de trabajo
  pre_grasp   — 10 cm arriba del objeto
  grasp       — brazo en contacto con el objeto
  place       — posición de depósito
  home        — posición de descanso al apagar

Flujo:
  home → detections → (espera objeto) → pre_grasp → grasp
  → cerrar gripper → pre_grasp → place → abrir gripper → detections

Uso:
  python3 pick_demo.py              # modo automático (auto-pick al detectar)
  python3 pick_demo.py --manual     # espera ENTER para ejecutar cada paso
"""

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String

from lerobot.motors import MotorCalibration, MotorNormMode, Motor
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ── Config ────────────────────────────────────────────────────────────────────

PORT       = "/dev/ttyACM0"
POSES_FILE = Path(__file__).parent / "poses.json"
MOVE_TIME  = 2.0   # segundos entre poses
GRIP_TIME  = 0.8   # tiempo para cerrar/abrir gripper

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


class PickDemo(Node):

    def __init__(self, manual: bool = False):
        super().__init__("pick_demo")
        self._manual    = manual
        self._detected  = False
        self._det_info  = ""

        # ── Cargar poses ───────────────────────────────────────────────────
        if not POSES_FILE.exists():
            self.get_logger().error(f"No se encontró poses.json en {POSES_FILE}")
            self.get_logger().error("Corre primero: python3 teach_poses.py")
            raise SystemExit(1)

        self._poses = json.loads(POSES_FILE.read_text())
        self.get_logger().info(f"Poses disponibles: {list(self._poses.keys())}")

        required = ["pre_grasp", "grasp", "place"]
        missing  = [p for p in required if p not in self._poses]
        if missing:
            self.get_logger().error(
                f"Faltan poses: {missing}\n"
                f"Enséñalas con: python3 teach_poses.py"
            )
            raise SystemExit(1)

        # ── Conectar bus ───────────────────────────────────────────────────
        self._bus = FeetechMotorsBus(
            port=PORT,
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
        self._bus.connect()
        self._bus.write_calibration(CALIBRATION)
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        self._bus.enable_torque()
        self.get_logger().info(f"Conectado a {PORT} — torque ON")

        # ── Vision subscription ────────────────────────────────────────────
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=2)
        self.create_subscription(String, "/vision/detections", self._det_cb, be)

        # ── Status publisher ───────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, "/pick_demo/status", 10)

        # ── Arrancar secuencia en timer (deja que el nodo inicialice) ──────
        self.create_timer(1.0, self._start_once)
        self._started = False

    # ── Vision callback ───────────────────────────────────────────────────────

    def _det_cb(self, msg: String):
        try:
            dets = json.loads(msg.data)
            if dets:
                self._detected = True
                best = dets[0]
                self._det_info = (f"{best['name']} "
                                  f"@ ({best['x']:.2f},{best['y']:.2f},{best['z']:.2f})m "
                                  f"conf={best['confidence']:.2f}")
        except Exception:
            pass

    # ── Main sequence ─────────────────────────────────────────────────────────

    def _start_once(self):
        if self._started:
            return
        self._started = True
        self.destroy_timer(list(self._timers)[0])

        try:
            self._run_pick_sequence()
        except KeyboardInterrupt:
            self.get_logger().info("Interrumpido → HOME")
            self._move("home")
        finally:
            self._bus.disable_torque()
            self._bus.disconnect()
            rclpy.shutdown()

    def _run_pick_sequence(self):
        self._pub("INIT")
        self.get_logger().info("=== PICK DEMO INICIADO ===")

        # 1. Ir a detections
        self._step("DETECTIONS", "detections")

        # 2. Esperar objeto
        self._pub("WAITING_FOR_OBJECT")
        self.get_logger().info("Esperando detección de objeto...")
        if self._manual:
            input("  [ENTER cuando haya un objeto frente a la cámara]")
        else:
            timeout = 30.0
            t0 = time.monotonic()
            while not self._detected:
                rclpy.spin_once(self, timeout_sec=0.1)
                if time.monotonic() - t0 > timeout:
                    self.get_logger().error("Timeout esperando objeto (30s)")
                    self._move("home")
                    return
        self.get_logger().info(f"  Objeto detectado: {self._det_info}")

        # 3. Pre-grasp
        self._step("PRE_GRASP", "pre_grasp")

        # 4. Grasp
        self._step("GRASP", "grasp", move_time=1.5)

        # 5. Cerrar gripper
        self._pub("CLOSE_GRIPPER")
        self.get_logger().info("→ Cerrando gripper...")
        if self._manual:
            input("  [ENTER para cerrar gripper]")
        self._gripper(90)   # 90% = casi cerrado
        time.sleep(GRIP_TIME)

        # 6. Levantar (volver a pre_grasp)
        self._step("LIFT", "pre_grasp", move_time=1.5)

        # 7. Place
        self._step("PLACE", "place")

        # 8. Abrir gripper
        self._pub("OPEN_GRIPPER")
        self.get_logger().info("→ Abriendo gripper...")
        if self._manual:
            input("  [ENTER para abrir gripper]")
        self._gripper(5)    # 5% = casi abierto
        time.sleep(GRIP_TIME)

        # 9. Volver a detections para siguiente ciclo
        self._step("DONE", "detections")
        self.get_logger().info("=== CICLO COMPLETO ===")
        self._detected = False

        if not self._manual:
            self.get_logger().info("Listo para siguiente objeto en 3s...")
            time.sleep(3.0)
            self._run_pick_sequence()   # loop automático

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _step(self, status: str, pose_name: str, move_time: float = MOVE_TIME):
        self._pub(status)
        self._move(pose_name, move_time)

    def _move(self, pose_name: str, move_time: float = MOVE_TIME):
        if pose_name not in self._poses:
            self.get_logger().warn(f"Pose '{pose_name}' no existe, ignorando.")
            return
        self.get_logger().info(f"→ {pose_name}")
        if self._manual:
            input(f"  [ENTER para mover a '{pose_name}']")
        self._bus.sync_write("Goal_Position", self._poses[pose_name])
        time.sleep(move_time)

    def _gripper(self, pct: float):
        self._bus.sync_write("Goal_Position", {"gripper": pct})

    def _pub(self, status: str):
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def destroy_node(self):
        try:
            self._bus.disable_torque()
            self._bus.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main():
    manual = "--manual" in sys.argv
    rclpy.init()
    node = PickDemo(manual=manual)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
