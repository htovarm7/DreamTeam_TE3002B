#!/usr/bin/env python3
"""
teach_poses.py  —  Modo enseñanza interactivo para SO-ARM100
─────────────────────────────────────────────────────────────
1. Deshabilita el torque → puedes mover el brazo a mano libremente
2. Presionas ENTER para leer la posición actual
3. Le das un nombre (home, detection, pre_grasp, place, etc.)
4. Al terminar, guarda todas las poses en poses.json
   y actualiza move_physical.py automáticamente.

Uso:
  python3 teach_poses.py
"""

import json
import os
import re
import time
from pathlib import Path

from lerobot.motors import MotorCalibration, MotorNormMode, Motor
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

# ── Config ────────────────────────────────────────────────────────────────────

PORT = "/dev/ttyACM0"

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

SCRIPT_DIR   = Path(__file__).parent
POSES_FILE   = SCRIPT_DIR / "poses.json"
MOVE_SCRIPT  = SCRIPT_DIR / "move_physical.py"


# ── Bus factory ───────────────────────────────────────────────────────────────

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


# ── Display ───────────────────────────────────────────────────────────────────

def print_pos(pos: dict, label: str = ""):
    header = f"  [{label}]" if label else ""
    print(f"\nPosición actual:{header}")
    for name, val in pos.items():
        unit = "%" if name == "gripper" else "°"
        print(f"    {name:>14}: {val:7.2f}{unit}")
    print()


def read_pos(bus: FeetechMotorsBus) -> dict:
    raw = bus.sync_read("Present_Position")
    return {k: round(v, 2) for k, v in raw.items()}


# ── Patch move_physical.py ────────────────────────────────────────────────────

def patch_move_script(poses: dict):
    """Rewrite the NAMED_POSES dict inside move_physical.py with the new values."""
    if not MOVE_SCRIPT.exists():
        print(f"  (move_physical.py no encontrado en {MOVE_SCRIPT}, solo se guardó poses.json)")
        return

    text = MOVE_SCRIPT.read_text()

    # Build new NAMED_POSES block
    lines = ["NAMED_POSES = {\n"]
    for pose_name, joints in poses.items():
        lines.append(f'    "{pose_name}": {{\n')
        items = list(joints.items())
        for i, (jname, val) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'                  "{jname}": {val}{comma}\n')
        lines.append("               },\n\n")
    lines.append("}\n")
    new_block = "".join(lines)

    # Replace existing NAMED_POSES block
    new_text = re.sub(
        r"NAMED_POSES\s*=\s*\{.*?\n\}\n",
        new_block,
        text,
        flags=re.DOTALL,
    )
    MOVE_SCRIPT.write_text(new_text)
    print(f"  move_physical.py actualizado con {len(poses)} poses.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load existing poses
    poses: dict = {}
    if POSES_FILE.exists():
        poses = json.loads(POSES_FILE.read_text())
        print(f"Poses existentes cargadas: {', '.join(poses.keys())}")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         MODO ENSEÑANZA  —  SO-ARM100                ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  El torque estará DESHABILITADO.                     ║")
    print("║  Mueve el brazo a mano a la posición deseada.       ║")
    print("║  Presiona ENTER para capturar.                       ║")
    print("║  Comandos: [nombre]  guardar pose                    ║")
    print("║            read      ver posición sin guardar        ║")
    print("║            list      listar poses guardadas          ║")
    print("║            test <n>  probar una pose guardada        ║")
    print("║            delete <n> borrar una pose                ║")
    print("║            done / quit  terminar y guardar           ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    bus = make_bus()
    try:
        print(f"Conectando a {PORT}...")
        bus.connect()
        bus.write_calibration(CALIBRATION)

        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        bus.disable_torque()
        print("Torque DESHABILITADO — puedes mover el brazo libremente.\n")

        while True:
            try:
                cmd = input("► ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                # ENTER sin texto → leer y pedir nombre
                pos = read_pos(bus)
                print_pos(pos)
                name = input("  Nombre para esta pose (ENTER para descartar): ").strip()
                if name:
                    poses[name] = pos
                    print(f"  ✓ Pose '{name}' guardada.")

            elif cmd == "read":
                pos = read_pos(bus)
                print_pos(pos)

            elif cmd == "list":
                if not poses:
                    print("  (sin poses guardadas)")
                else:
                    for pname, pvals in poses.items():
                        parts = ", ".join(
                            f"{k}={v:.1f}{'%' if k=='gripper' else '°'}"
                            for k, v in pvals.items()
                        )
                        print(f"  {pname:>12}: {parts}")

            elif cmd.startswith("test "):
                pname = cmd[5:].strip()
                if pname not in poses:
                    print(f"  Pose '{pname}' no existe.")
                else:
                    print(f"  Habilitando torque y moviendo a '{pname}'...")
                    bus.enable_torque()
                    bus.sync_write("Goal_Position", poses[pname])
                    time.sleep(2.5)
                    pos = read_pos(bus)
                    print_pos(pos, label=pname)
                    bus.disable_torque()
                    print("  Torque deshabilitado de nuevo.\n")

            elif cmd.startswith("delete "):
                pname = cmd[7:].strip()
                if pname in poses:
                    del poses[pname]
                    print(f"  Pose '{pname}' eliminada.")
                else:
                    print(f"  Pose '{pname}' no existe.")

            elif cmd in ("done", "quit", "exit", "q"):
                break

            else:
                # Tratar el texto como nombre de pose directamente
                pos = read_pos(bus)
                print_pos(pos, label=cmd)
                poses[cmd] = pos
                print(f"  ✓ Pose '{cmd}' guardada.")

    finally:
        bus.disable_torque()
        bus.disconnect()

    # Guardar poses.json
    POSES_FILE.write_text(json.dumps(poses, indent=4))
    print(f"\n✓ {len(poses)} poses guardadas en {POSES_FILE}")

    # Actualizar move_physical.py
    if poses:
        patch_move_script(poses)

    print("\nPoses finales:")
    for pname, pvals in poses.items():
        parts = ", ".join(
            f"{k}={v:.1f}{'%' if k=='gripper' else '°'}"
            for k, v in pvals.items()
        )
        print(f"  {pname:>12}: {parts}")


if __name__ == "__main__":
    main()
