#!/usr/bin/env python3
"""
pick_manual.py
──────────────
Flujo pick & place manual para SO-ARM100 físico.
Sin ROS2. Sin visión. Presionas ENTER en cada paso.

Secuencia:
  HOME → PRE_GRASP → GRASP → LIFT → PLACE → OPEN_GRIPPER → PRE_HOME → HOME

El gripper va embebido en cada pose (no hay comandos de gripper separados).

Uso:
  python3 pick_manual.py
"""

import json
import sys
import time
from pathlib import Path

from lerobot.motors import MotorCalibration, MotorNormMode, Motor
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ── Config ─────────────────────────────────────────────────────────────────────

PORT      = "/dev/ttyACM0"
MOVE_TIME = 3.0
POSES_FILE = Path(__file__).parent / "poses.json"

SEQUENCE = [
    ("HOME",         "Posición inicial — gripper casi cerrado."),
    ("PRE_GRASP",    "Arriba del objeto — gripper abierto."),
    ("GRASP",        "Bajar hasta el objeto — gripper se cierra."),
    ("LIFT",         "Levantar el objeto."),
    ("PLACE",        "Ir a la posición de depósito."),
    ("OPEN_GRIPPER", "Soltar el objeto — gripper se abre."),
    ("HOME",         "Regresar a posición inicial."),
]

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


def make_bus() -> FeetechMotorsBus:
    return FeetechMotorsBus(
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


def load_poses() -> dict:
    if not POSES_FILE.exists():
        print(f"ERROR: no se encontró {POSES_FILE}")
        print("Corre primero:  python3 teach_poses.py")
        sys.exit(1)
    poses = json.loads(POSES_FILE.read_text())
    required = {name for name, _ in SEQUENCE}
    missing = [p for p in required if p not in poses]
    if missing:
        print(f"ERROR: faltan poses en poses.json: {missing}")
        print("Enséñalas con:  python3 teach_poses.py")
        sys.exit(1)
    return poses


def prompt(step: str, description: str):
    print(f"\n{'─'*54}")
    print(f"  PASO: {step}")
    print(f"  {description}")
    try:
        input("  [ENTER para ejecutar] ")
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt


def move(bus: FeetechMotorsBus, pose_name: str, poses: dict, move_time: float = MOVE_TIME):
    print(f"  → Moviendo a '{pose_name}'...")
    bus.sync_write("Goal_Position", poses[pose_name])
    time.sleep(move_time)
    print("  ✓ Listo.")


def run_pick_place(bus: FeetechMotorsBus, poses: dict):
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║         PICK & PLACE MANUAL  —  SO-ARM100             ║")
    print("║  Presiona ENTER en cada paso para continuar.           ║")
    print("║  Ctrl+C para abortar y volver a HOME.                  ║")
    print("╚════════════════════════════════════════════════════════╝")

    for pose_name, description in SEQUENCE:
        prompt(pose_name, description)
        move(bus, pose_name, poses)

    print("\n╔════════════════════════════════════════════════════════╗")
    print("║  ✓  CICLO COMPLETADO                                   ║")
    print("╚════════════════════════════════════════════════════════╝\n")


def main():
    poses = load_poses()
    print(f"Poses cargadas: {[name for name, _ in SEQUENCE]}")

    bus = make_bus()
    try:
        print(f"Conectando a {PORT}...")
        bus.connect()
        bus.write_calibration(CALIBRATION)
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.enable_torque()
        print("Torque HABILITADO.\n")

        while True:
            run_pick_place(bus, poses)
            try:
                again = input("¿Repetir ciclo? [s/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if again not in ("s", "si", "sí", "y", "yes"):
                break

    except KeyboardInterrupt:
        print("\n\nAbortado. Volviendo a HOME...")
        try:
            bus.sync_write("Goal_Position", poses["HOME"])
            time.sleep(MOVE_TIME)
        except Exception:
            pass
    finally:
        bus.disable_torque()
        bus.disconnect()
        print("Desconectado.")


if __name__ == "__main__":
    main()
