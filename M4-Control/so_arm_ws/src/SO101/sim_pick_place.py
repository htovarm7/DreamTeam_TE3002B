#!/usr/bin/env python3
"""
sim_pick_place.py
─────────────────
Simulación MuJoCo del SO101 con detección y pick & place físico.

Flujo:
  home → detections (detecta cubo) → pre_grasp → grasp
  → cerrar gripper (agarra cubo) → levantar → place → soltar → detections

Controles en la ventana del viewer:
  SPACE  — avanza al siguiente paso (modo manual)
  R      — reinicia simulación
  A      — alterna modo auto / manual
  Q/ESC  — salir

Uso:
  python3 sim_pick_place.py            # modo manual
  python3 sim_pick_place.py --auto     # ciclo automático
"""

import sys
import math
import argparse
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# ── Rutas ─────────────────────────────────────────────────────────────────────

SCENE_XML = Path(__file__).parent / "pick_place_scene.xml"

# ── Joints y límites ──────────────────────────────────────────────────────────

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper",
]

JOINT_LIMITS = {
    "shoulder_pan":  (-1.9199, 1.9199),
    "shoulder_lift": (-1.7453, 1.7453),
    "elbow_flex":    (-1.6900, 1.6900),
    "wrist_flex":    (-1.6581, 1.6581),
    "wrist_roll":    (-2.7438, 2.8412),
    "gripper":       (-0.1745, 1.7453),
}


def _clamp(v, name):
    lo, hi = JOINT_LIMITS[name]
    return max(lo, min(hi, v))

def _d2r(deg):  return deg * math.pi / 180.0
def _g2r(pct):  return (pct / 100.0) * 1.7   # 0%=cerrado→0 rad, 100%=abierto→1.7 rad

def _pose(pan, lift, elbow, wflx, wroll, grip):
    raw = [_d2r(pan), _d2r(lift), _d2r(elbow), _d2r(wflx), _d2r(wroll), _g2r(grip)]
    return np.array([_clamp(v, n) for v, n in zip(raw, JOINT_NAMES)])


POSES = {
    #                    pan     lift   elbow  w_flex w_roll grip%
    "home":       _pose(  1.27, -54.29,  45.80,  85.01, 90.0,   2.0),
    "detections": _pose( -1.71, -47.60,  47.56,  24.62, 90.0,   2.0),
    "pre_grasp":  _pose(  1.89,  55.60, -79.74,  95.00, 90.0,  95.0),
    "grasp":      _pose( 14.55,  87.25, -73.23,  95.00, 90.0,  95.0),
    "close_grip": _pose( 14.55,  87.25, -73.23,  95.00, 90.0,   8.0),
    "lift":       _pose(  1.89,  55.60, -79.74,  95.00, 90.0,   8.0),
    "place":      _pose( 75.12,  85.67, -73.14,  95.00, 90.0,   8.0),
    "open_grip":  _pose( 75.12,  85.67, -73.14,  95.00, 90.0,  95.0),
}

# Secuencia: (pose_key, descripción, callback_al_terminar)
SEQUENCE = [
    ("home",       "HOME — posición de descanso",               None),
    ("detections", "DETECTIONS — buscando objeto...",           "detect"),
    ("pre_grasp",  "PRE-GRASP — posicionando sobre el cubo",    None),
    ("grasp",      "GRASP — bajando al cubo",                   None),
    ("close_grip", "CLOSE GRIP — cerrando gripper",             "attach"),
    ("lift",       "LIFT — levantando el objeto",               None),
    ("place",      "PLACE — moviéndose a zona de depósito",     None),
    ("open_grip",  "OPEN GRIP — soltando el objeto",            "detach"),
    ("detections", "Volviendo a posición de observación",       None),
]

MOVE_DURATION   = 2.5   # segundos por movimiento
SIM_STEPS_PER_S = 500   # pasos de simulación por segundo de movimiento


# ── Utilidades de cuaterniones (convenio MuJoCo: [w, x, y, z]) ───────────────

def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ── Clase principal ───────────────────────────────────────────────────────────

class PickPlaceSim:

    def __init__(self, auto: bool = False):
        if not SCENE_XML.exists():
            raise FileNotFoundError(f"No se encontró: {SCENE_XML}")

        self._model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self._data  = mujoco.MjData(self._model)
        self._auto  = auto

        # IDs de actuadores
        self._act_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in JOINT_NAMES
        ]

        # IDs de cuerpos y sitios
        self._gripper_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._cube_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._grip_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._cam_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "cam_site")

        # Dirección freejoint del cubo en qpos/qvel
        _jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_adr = self._model.jnt_qposadr[_jid]
        self._cube_qvel_adr = self._model.jnt_dofadr[_jid]

        # Estado del agarre
        self._holding       = False
        self._rel_pos_body  = None   # posición relativa en frame del gripper
        self._rel_quat      = None   # cuaternión relativo

        # Control de flujo
        self._step_idx   = 0
        self._advance    = threading.Event()
        self._reset_flag = False
        self._quit_flag  = False
        self._auto_flag  = auto

        # Colocar cubo justo donde llegará el gripper en grasp
        print("Calculando posición de agarre...")
        self._auto_place_cube()

        # Inicializar en HOME
        self._set_ctrl(POSES["home"])
        mujoco.mj_forward(self._model, self._data)

    # ── Posicionamiento automático del cubo ───────────────────────────────────

    def _auto_place_cube(self):
        """Corre la pose grasp en silencio y pone el cubo donde llega el gripper."""
        ctrl_bak  = self._data.ctrl.copy()
        qpos_bak  = self._data.qpos.copy()
        qvel_bak  = self._data.qvel.copy()

        for i, aid in enumerate(self._act_ids):
            self._data.ctrl[aid] = POSES["grasp"][i]
        for _ in range(1500):
            mujoco.mj_step(self._model, self._data)

        grip_pos = self._data.site_xpos[self._grip_site_id].copy()

        # Restaurar estado
        self._data.ctrl[:] = ctrl_bak
        np.copyto(self._data.qpos, qpos_bak)
        np.copyto(self._data.qvel, qvel_bak)

        # Cubo en (x, y) del gripper, apoyado en el suelo
        cx, cy, cz = grip_pos[0], grip_pos[1], 0.025
        adr = self._cube_qpos_adr
        self._data.qpos[adr:adr+3] = [cx, cy, cz]
        self._data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self._model, self._data)

        print(f"  Cubo colocado en ({cx:.3f}, {cy:.3f}, {cz:.3f}) m")

    # ── Control de actuadores ─────────────────────────────────────────────────

    def _set_ctrl(self, target: np.ndarray):
        for i, aid in enumerate(self._act_ids):
            self._data.ctrl[aid] = target[i]

    def _current_qpos(self) -> np.ndarray:
        vals = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            vals.append(self._data.qpos[self._model.jnt_qposadr[jid]])
        return np.array(vals)

    # ── Detección simulada ────────────────────────────────────────────────────

    def _detect_cube(self) -> bool:
        """Simula un sensor de cámara-profundidad desde virtual_cam."""
        cam_pos  = self._data.site_xpos[self._cam_site_id]
        cube_pos = self._data.xpos[self._cube_body_id]

        diff     = cube_pos - cam_pos
        distance = float(np.linalg.norm(diff))

        # Añadir ruido de sensor (σ = 5 mm)
        rng  = np.random.default_rng()
        noise = rng.normal(0, 0.005, 3)
        meas  = cube_pos + noise

        if distance < 0.90:
            conf = max(0.50, 1.0 - distance * 0.8)
            print(f"\n  ┌── [DETECCIÓN] Objeto encontrado ─────────────────")
            print(f"  │  Clase      : cubo")
            print(f"  │  Posición   : ({meas[0]:.3f}, {meas[1]:.3f}, {meas[2]:.3f}) m")
            print(f"  │  Distancia  : {distance:.3f} m")
            print(f"  │  Confianza  : {conf:.2f}")
            print(f"  └──────────────────────────────────────────────────\n")
            return True
        else:
            print("\n  [DETECCIÓN] No se encontró ningún objeto en el workspace.\n")
            return False

    # ── Agarre cinético ───────────────────────────────────────────────────────

    def _attach_cube(self):
        """Registra la pose relativa cubo–gripper y activa el agarre."""
        grip_pos  = self._data.xpos[self._gripper_body_id]
        cube_pos  = self._data.xpos[self._cube_body_id]
        R_grip    = self._data.xmat[self._gripper_body_id].reshape(3, 3)

        dist = float(np.linalg.norm(cube_pos - grip_pos))

        # Posición relativa en el frame del gripper
        self._rel_pos_body = R_grip.T @ (cube_pos - grip_pos)

        # Cuaternión relativo: q_rel = q_grip^-1 * q_cube
        q_grip = self._data.xquat[self._gripper_body_id]
        q_cube = self._data.xquat[self._cube_body_id]
        self._rel_quat = _qmul(_qconj(q_grip), q_cube)

        self._holding = True
        print(f"  [GRASP] ¡Cubo agarrado! (dist gripper: {dist:.3f} m)")

    def _detach_cube(self):
        """Libera el cubo para que caiga libremente."""
        self._holding = False
        self._rel_pos_body = None
        self._rel_quat     = None
        print("  [RELEASE] Cubo liberado.")

    def _update_hold(self):
        """Llamar en cada paso de simulación mientras se sostiene el cubo."""
        if not self._holding:
            return
        R_grip   = self._data.xmat[self._gripper_body_id].reshape(3, 3)
        grip_pos = self._data.xpos[self._gripper_body_id]
        q_grip   = self._data.xquat[self._gripper_body_id]

        new_pos  = grip_pos + R_grip @ self._rel_pos_body
        new_quat = _qmul(q_grip, self._rel_quat)

        adr = self._cube_qpos_adr
        self._data.qpos[adr:adr+3] = new_pos
        self._data.qpos[adr+3:adr+7] = new_quat

        vadr = self._cube_qvel_adr
        self._data.qvel[vadr:vadr+6] = 0.0

    # ── Movimiento suave ──────────────────────────────────────────────────────

    def _move_to(self, target: np.ndarray, viewer, duration: float = MOVE_DURATION):
        start       = self._current_qpos()
        total_steps = max(1, int(duration * SIM_STEPS_PER_S))

        for k in range(total_steps):
            if self._reset_flag or self._quit_flag:
                return
            alpha = k / total_steps
            t     = alpha * alpha * (3.0 - 2.0 * alpha)   # ease-in-out cúbico
            self._set_ctrl(start + t * (target - start))
            mujoco.mj_step(self._model, self._data)
            self._update_hold()
            if viewer.is_running():
                viewer.sync()
            else:
                self._quit_flag = True
                return

    def _idle(self, viewer, seconds: float = 0.4):
        steps = int(seconds * SIM_STEPS_PER_S)
        for _ in range(steps):
            if self._reset_flag or self._quit_flag:
                return
            mujoco.mj_step(self._model, self._data)
            self._update_hold()
            if viewer.is_running():
                viewer.sync()
            else:
                self._quit_flag = True
                return

    # ── Secuencia principal ───────────────────────────────────────────────────

    def _run_step(self, viewer):
        if self._step_idx >= len(SEQUENCE):
            return

        pose_key, description, callback = SEQUENCE[self._step_idx]
        dur = 1.5 if "grip" in pose_key else MOVE_DURATION

        print(f"\n[{self._step_idx+1}/{len(SEQUENCE)}] {description}")

        if not self._auto_flag:
            print("  → SPACE para ejecutar...")
            self._advance.clear()
            while not self._advance.wait(timeout=0.05):
                if self._reset_flag or self._quit_flag or not viewer.is_running():
                    return
                mujoco.mj_step(self._model, self._data)
                self._update_hold()
                viewer.sync()

        self._move_to(POSES[pose_key], viewer, duration=dur)
        self._idle(viewer, 0.3)

        if callback == "detect":
            self._detect_cube()
            if not self._auto_flag:
                input("  [ENTER para continuar con el pick & place]")

        elif callback == "attach":
            self._attach_cube()

        elif callback == "detach":
            self._detach_cube()

        self._step_idx += 1

    # ── Bucle principal ───────────────────────────────────────────────────────

    def run(self):
        with mujoco.viewer.launch_passive(
            self._model, self._data, key_callback=self._key_cb
        ) as viewer:

            viewer.cam.distance  = 0.85
            viewer.cam.azimuth   = 120
            viewer.cam.elevation = -22
            viewer.cam.lookat[:] = [0.15, -0.10, 0.15]

            _banner(self._auto_flag)

            while viewer.is_running() and not self._quit_flag:

                if self._reset_flag:
                    self._do_reset(viewer)
                    continue

                if self._step_idx >= len(SEQUENCE):
                    if self._auto_flag:
                        self._step_idx = 0
                        print("\n" + "═"*50)
                        print("  NUEVO CICLO")
                        print("═"*50)
                    else:
                        print("\nCiclo completo. Pulsa R para reiniciar o Q para salir.")
                        while (viewer.is_running()
                               and not self._reset_flag
                               and not self._quit_flag):
                            mujoco.mj_step(self._model, self._data)
                            self._update_hold()
                            viewer.sync()
                    continue

                self._run_step(viewer)

        print("\nSimulación terminada.")

    def _do_reset(self, viewer):
        print("\n[RESET]\n")
        mujoco.mj_resetData(self._model, self._data)
        self._holding   = False
        self._step_idx  = 0
        self._reset_flag = False
        self._auto_place_cube()
        self._set_ctrl(POSES["home"])
        mujoco.mj_forward(self._model, self._data)
        viewer.sync()

    # ── Teclado ───────────────────────────────────────────────────────────────

    def _key_cb(self, keycode: int):
        if keycode == 32:           # SPACE
            self._advance.set()
        elif keycode == ord('R'):
            self._reset_flag = True
            self._advance.set()
        elif keycode == ord('A'):
            self._auto_flag = not self._auto_flag
            print(f"Modo: {'AUTOMÁTICO' if self._auto_flag else 'MANUAL'}")
        elif keycode in (ord('Q'), 27):   # Q o ESC
            self._quit_flag = True
            self._advance.set()


# ── Helpers de presentación ───────────────────────────────────────────────────

def _banner(auto: bool):
    print("\n" + "═"*52)
    print("   SO101 — MuJoCo Pick & Place Simulation")
    print("═"*52)
    print(f"   Modo   : {'AUTOMÁTICO' if auto else 'MANUAL (SPACE para avanzar)'}")
    print("   R      : reiniciar")
    print("   A      : alternar auto/manual")
    print("   Q/ESC  : salir")
    print("═"*52 + "\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SO101 MuJoCo Pick & Place")
    parser.add_argument("--auto", action="store_true",
                        help="Ejecutar ciclo automático continuo")
    args = parser.parse_args()

    try:
        PickPlaceSim(auto=args.auto).run()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
