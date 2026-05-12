#!/usr/bin/env python3
"""
move_physical.py
────────────────
Mueve el brazo SO-ARM100 físico usando la calibración de lerobot.

Uso:
  python3 move_physical.py                     → va a HOME
  python3 move_physical.py home                → va a HOME
  python3 move_physical.py ready               → posición ready
  python3 move_physical.py pre_grasp           → posición pre-grasp
  python3 move_physical.py open                → abre gripper
  python3 move_physical.py closed              → cierra gripper
  python3 move_physical.py read                → lee posición actual
  python3 move_physical.py 0,0,0,0,0,50       → joints explícitos (grados):
                                                 shoulder_pan, shoulder_lift,
                                                 elbow_flex, wrist_flex,
                                                 wrist_roll, gripper(0-100)

Joints en GRADOS (con use_degrees=True de lerobot):
  shoulder_pan  : 0° = centro, ±180° rotación base
  shoulder_lift : 0° = neutro
  elbow_flex    : 0° = neutro
  wrist_flex    : 0° = neutro
  wrist_roll    : 0° = neutro
  gripper       : valor en rango 0-100 (0=abierto, 100=cerrado)
"""

import sys
import time

from lerobot.motors import MotorCalibration, MotorNormMode, Motor
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ── Configuración ─────────────────────────────────────────────────────────────

PORT      = "/dev/ttyACM0"
MOVE_TIME = 5.0   # segundos que tarda en llegar a la posición

# Valores en grados (motor body joints) y 0-100 (gripper)
NAMED_POSES = {
    "detections": {
                  "shoulder_pan": -1.71,
                  "shoulder_lift": -47.6,
                  "elbow_flex": 47.56,
                  "wrist_flex": 24.62,
                  "wrist_roll": 175.25,
                  "gripper": 1.44
               },

    "home": {
                  "shoulder_pan": 1.27,
                  "shoulder_lift": -54.29,
                  "elbow_flex": 45.8,
                  "wrist_flex": 85.01,
                  "wrist_roll": 178.42,
                  "gripper": 0.79
               },

    "HOME": {
                  "shoulder_pan": 17.63,
                  "shoulder_lift": -30.55,
                  "elbow_flex": -40.0,
                  "wrist_flex": 95.56,
                  "wrist_roll": 172.0,
                  "gripper": 6.37
               },

    "PRE_GRASP": {
                  "shoulder_pan": 19.21,
                  "shoulder_lift": 46.2,
                  "elbow_flex": -42.99,
                  "wrist_flex": 104.7,
                  "wrist_roll": 172.09,
                  "gripper": 72.42
               },

    "GRASP": {
                  "shoulder_pan": 18.68,
                  "shoulder_lift": 66.51,
                  "elbow_flex": -42.9,
                  "wrist_flex": 89.49,
                  "wrist_roll": 172.0,
                  "gripper": 18.32
               },

    "LIFT": {
                  "shoulder_pan": 12.53,
                  "shoulder_lift": 40.13,
                  "elbow_flex": -100.22,
                  "wrist_flex": 90.02,
                  "wrist_roll": 171.91,
                  "gripper": 18.32
               },

    "PLACE": {
                  "shoulder_pan": 100.0,
                  "shoulder_lift": 77.67,
                  "elbow_flex": -71.47,
                  "wrist_flex": 99.16,
                  "wrist_roll": 172.35,
                  "gripper": 18.32
               },

    "OPEN_GRIPPER": {
                  "shoulder_pan": 100.09,
                  "shoulder_lift": 78.99,
                  "elbow_flex": -71.56,
                  "wrist_flex": 99.52,
                  "wrist_roll": -176.4,
                  "gripper": 93.24
               },

    "PRE_HOME": {
                  "shoulder_pan": 99.47,
                  "shoulder_lift": 8.75,
                  "elbow_flex": -71.47,
                  "wrist_flex": 99.69,
                  "wrist_roll": -176.48,
                  "gripper": 93.24
               },

}


# ── Calibración (leída del archivo de lerobot) ────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

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


def read_position(bus: FeetechMotorsBus):
    pos = bus.sync_read("Present_Position")
    print("\nPosición actual:")
    for name, val in pos.items():
        unit = "%" if name == "gripper" else "°"
        print(f"  {name:>14}: {val:7.2f}{unit}")
    print()


def move_to(bus: FeetechMotorsBus, goal: dict):
    print("Moviendo a:")
    for name, val in goal.items():
        unit = "%" if name == "gripper" else "°"
        print(f"  {name:>14}: {val:7.2f}{unit}")

    bus.sync_write("Goal_Position", goal)
    time.sleep(MOVE_TIME)
    print("Listo.\n")


def main():
    cmd = sys.argv[1].strip() if len(sys.argv) > 1 else "home"

    # Resolver posición
    if cmd == "read":
        goal = None
    elif cmd in NAMED_POSES:
        goal = NAMED_POSES[cmd]
    else:
        # Intentar parsear como CSV: pan, lift, elbow, wrist_flex, wrist_roll, gripper
        try:
            vals = [float(v) for v in cmd.split(",")]
            if len(vals) != 6:
                print(f"ERROR: necesitas 6 valores (pan, lift, elbow, wrist_flex, wrist_roll, gripper)")
                print(f"       ejemplo: 0,0,0,0,0,50")
                sys.exit(1)
            goal = dict(zip(
                ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll", "gripper"],
                vals
            ))
        except ValueError:
            print(f"ERROR: pose desconocida '{cmd}'")
            print(f"Poses disponibles: {', '.join(NAMED_POSES.keys())}")
            print(f"O pasa 6 valores CSV en grados: pan,lift,elbow,wrist_flex,wrist_roll,gripper(0-100)")
            sys.exit(1)

    # Conectar y ejecutar
    bus = make_bus()
    try:
        print(f"Conectando a {PORT}...")
        bus.connect()
        bus.write_calibration(CALIBRATION)

        # Habilitar torque y modo posición
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        bus.enable_torque()
        print("Torque habilitado.\n")

        if goal is None:
            read_position(bus)
        else:
            read_position(bus)
            move_to(bus, goal)
            read_position(bus)

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        bus.disable_torque()
        bus.disconnect()
        print("Desconectado.")


if __name__ == "__main__":
    main()
