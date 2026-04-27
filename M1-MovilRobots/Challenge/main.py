"""
================================================================================
    Mobile Robots Challenge — TE3002B M1
    Husky A200 + ANYmal + xArm6 Lite + PuzzleBots
================================================================================

Misión integrada:
        1. Husky A200 (skid-steer) empuja cajas bloqueando la pista hacia la salida
        2. ANYmal (4 patas, cuadrúpedo) lo sigue y pasa detrás del Husky
        3. ANYmal se posiciona en estación de trabajo (mesa con PuzzleBots)
        4. Brazo xArm6 Lite (adherido al ANYmal) organiza/apila PuzzleBots en mesa
        5. Objetivo final: Husky en salida + ANYmal organizando puzzles en mesa

Pipeline ML:
        Fase 1 — EMPUJE (Husky):
                - Random Forest  clasifica acción del Husky (APPROACH/PUSH/NAVIGATE)
                - Ridge Regression predice (v, ω) del Husky para empuje
                - Política Híbrida combina ambas
        
        Fase 2 — NAVEGACIÓN (ANYmal):
                - RF clasifica: FOLLOW_HUSKY → MOVE_TO_MESA → AWAIT
                - Ridge predice velocidad de seguimiento y aproximación
        
        Fase 3 — MANIPULACIÓN (xArm6 Lite):
                - RF clasifica: GRAB_PUZZLE → ORIENT → PLACE → RELEASE
                - Ridge predice posiciones (x,y,z) del end-effector

Robots:
        - Husky A200        : skid-steer, radio 0.1651m, 50kg, max 1.0 m/s
        - ANYmal            : 4 patas, altura 0.52m, 30kg, max 0.8 m/s
        - xArm6 Lite        : brazo 6-DOF sobre ANYmal, reach 0.65m, payload 1kg
        - PuzzleBots (×3)   : cubos 0.15×0.15×0.15 m en mesa de 1.5×1.0 m

Parámetros principales:
        N_BOXES        = 3      ← cajas en pista (Husky empuja)
        N_PUZZLES      = 3      ← cubos en mesa (xArm6 organiza)
        CHALLENGE_MODE = True   ← habilita obstáculos y timeouts

Arquitectura modular:
        Cada robot implementa: step(cmd, dt) / get_pose() / reset()
        Arena maneja cajas, mesa y colisiones
        StateExtractor unifica características de todos los robots
        HeuristicMultiAgent genera datos para los 3 puntos de control

================================================================================
"""

import subprocess, sys
for pkg in ['numpy', 'matplotlib', 'scikit-learn', 'pandas', 'seaborn']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (classification_report, confusion_matrix,
                                      mean_squared_error, r2_score,
                                      accuracy_score)

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#161b22',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#c9d1d9',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'text.color'       : '#c9d1d9',
    'grid.color'       : '#21262d',
    'grid.alpha'       : 0.6,
    'legend.facecolor' : '#161b22',
    'legend.edgecolor' : '#30363d',
    'font.size'        : 10,
})

N_BOXES = 3

BOX_COLORS = ['#ffd700', '#ff6b6b', '#4ecdc4', '#a8e6cf', '#ff8b94', '#b4a7d6']

print(f'✅ Configuración: N_BOXES={N_BOXES} | Seed={SEED}')


class HuskyA200:
    """
    Husky A200 de Clearpath Robotics — skid-steer 4 ruedas.

    El giro se logra variando velocidad entre lados (como un tanque),
    lo que implica deslizamiento lateral durante el giro.

    Atributo ROBOT_TYPE: permite al simulador identificar el robot.
    Otros robots (Jackal, Ridgeback) heredarán de RobotBase con
    la misma interfaz: step / get_pose / reset.
    """
    ROBOT_TYPE      = 'husky_a200'
    ROBOT_FOOTPRINT = (0.99, 0.67)  
    ROBOT_COLOR     = '#f0883e'     

    def __init__(self, r=0.1651, B=0.555, mass=50.0):
        self.r    = r    
        self.B    = B   
        self.mass = mass  
        self.payload_max = 75.0

        self.max_wheel_speed = 8.0  
        self.max_v           = 1.0    
        self.max_omega       = 2.0   

        self.x = self.y = self.theta = 0.0
        self.v = self.omega = 0.0

        self.terrain = 'asphalt'
        self._slip = {
            'asphalt': 1.00,
            'grass'  : 0.85,
            'gravel' : 0.78,
            'sand'   : 0.65,
            'mud'    : 0.50,
        }

    def forward_kinematics(self, wR1, wR2, wL1, wL2):
        """
        Cinemática directa: (wR1, wR2, wL1, wL2) [rad/s] → (v [m/s], ω [rad/s])

        wR1, wR2 : ruedas derechas (frontal, trasera)
        wL1, wL2 : ruedas izquierdas (frontal, trasera)
        Promedia cada lado (idealización: ruedas del mismo lado giran igual).
        Aplica factor de deslizamiento del terreno sobre v.
        """
        avg_R = (wR1 + wR2) / 2.0
        avg_L = (wL1 + wL2) / 2.0
        slip  = self._slip.get(self.terrain, 0.8)
        v     = self.r / 2.0 * (avg_R + avg_L) * slip
        omega = self.r / self.B * (avg_R - avg_L)
        return v, omega

    def inverse_kinematics(self, v, omega):
        """
        Cinemática inversa: (v, ω) → (wR1, wR2, wL1, wL2) saturados a límites hw.
        Se asume que las ruedas de cada lado giran a la misma velocidad.
        """
        wR = (2.0 * v + omega * self.B) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.B) / (2.0 * self.r)
        wR = np.clip(wR, -self.max_wheel_speed, self.max_wheel_speed)
        wL = np.clip(wL, -self.max_wheel_speed, self.max_wheel_speed)
        return wR, wR, wL, wL   # (wR1, wR2, wL1, wL2)

    def update_pose(self, v, omega, dt):
        """
        Integra la pose usando el método del punto medio.
        Más preciso que Euler puro para trayectorias curvas.
        """
        theta_mid   = self.theta + omega * dt / 2.0
        self.x     += v * np.cos(theta_mid) * dt
        self.y     += v * np.sin(theta_mid) * dt
        self.theta += omega * dt
        self.theta  = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.v, self.omega = v, omega

    def step(self, v_cmd, omega_cmd, dt=0.05):
        """Aplica comando y actualiza pose. Retorna (v_real, omega_real)."""
        v_cmd     = np.clip(v_cmd,     -self.max_v,     self.max_v)
        omega_cmd = np.clip(omega_cmd, -self.max_omega, self.max_omega)
        w = self.inverse_kinematics(v_cmd, omega_cmd)
        v, omega = self.forward_kinematics(*w)
        self.update_pose(v, omega, dt)
        return v, omega

    def get_pose(self):
        return (self.x, self.y, self.theta)

    def set_terrain(self, t):
        self.terrain = t

    def reset(self, x=0., y=0., theta=0.):
        self.x, self.y, self.theta = x, y, theta
        self.v = self.omega = 0.0


h = HuskyA200()
v, w = h.forward_kinematics(3., 3., 3., 3.)
print(f'✅ HuskyA200 | v={v:.4f} m/s  ω={w:.4f} rad/s (recto)')
wr1, wr2, wl1, wl2 = h.inverse_kinematics(0.5, 0.8)
print(f'   IK (v=0.5, ω=0.8) → wR={wr1:.2f}  wL={wl1:.2f} rad/s')


class Box:
    """
    Caja física que puede ser empujada por el robot.

    Física simplificada:
        F_net = F_push - F_friccion
        Si F_push < F_friccion → la caja no se mueve (estática).
        La velocidad de la caja se amortigua en cada paso (coef. 0.85).
    """
    SIZE = 0.50  
    MASS = 10.0   
    MU   = 0.35   
    G    = 9.81   

    def __init__(self, x, y, box_id):
        self.x0          = x       
        self.y0          = y
        self.x           = x       
        self.y           = y
        self.id          = box_id
        self.pushed      = False   
        self.vx          = 0.0     
        self.vy          = 0.0
        self.push_count  = 0       

    def apply_push(self, fx, fy, dt):
        """
        Aplica fuerza de empuje (fx, fy) [N] durante dt [s].
        Modelo: F_net = F_push − F_friccion, con amortiguación.
        """
        f_frict = self.MU * self.MASS * self.G
        f_mag   = np.hypot(fx, fy)
        if f_mag < f_frict:           
            return
        ax = (fx - f_frict * fx / (f_mag + 1e-9)) / self.MASS
        ay = (fy - f_frict * fy / (f_mag + 1e-9)) / self.MASS
        self.vx += ax * dt
        self.vy += ay * dt
        self.vx *= 0.85               
        self.vy *= 0.85
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.push_count += 1

    def reset(self):
        self.x, self.y  = self.x0, self.y0
        self.vx = self.vy = 0.0
        self.pushed     = False
        self.push_count = 0

    @property
    def center(self):
        return np.array([self.x, self.y])

    @property
    def corners(self):
        h = self.SIZE / 2
        return [(self.x-h, self.y-h), (self.x+h, self.y-h),
                (self.x+h, self.y+h), (self.x-h, self.y+h)]


class Arena:
    """
    Pista rectangular con zona de paso bloqueada por cajas.

    Layout (metros):

        y=10 ┌──────────[SALIDA]──────────┐
             │                            │
        y=7  │   ░░░░ ZONA BLOQUEADA ░░░░ │  ← cajas aquí
        y=5  │                            │
             │                            │
        y=0  └────────────────────────────┘
             x=0                        x=8

    - Robot empieza en (4, 1) mirando norte (θ = π/2).
    - Salida: franja superior (y > 9).
    - Cajas colocadas aleatoriamente en y ∈ [5.5, 7.5], x ∈ [1.5, 6.5].
    """
    WIDTH       = 8.0
    HEIGHT      = 10.0
    BLOCK_Y_MIN = 4.5
    BLOCK_Y_MAX = 6.0
    BLOCK_X_MIN = 1.5
    BLOCK_X_MAX = 6.5
    CLEAR_Y     = 3.5  

    def __init__(self, n_boxes=N_BOXES, seed=SEED):
        self.n_boxes = n_boxes
        self.rng     = np.random.default_rng(seed)
        self.boxes   = self._place_boxes()

        self.start  = (self.WIDTH / 2, 1.0, np.pi / 2)   
        self.goal   = (self.WIDTH / 2, self.HEIGHT - 0.5)
        self.goal_r = 0.7   

    def _place_boxes(self):
        """Coloca n_boxes cajas centradas en el medio de la pista, igualmente espaciadas."""
        if self.n_boxes == 0:
            return []
        xs = np.linspace(
            self.BLOCK_X_MIN + 0.5,
            self.BLOCK_X_MAX - 0.5,
            self.n_boxes
        )
        y_center = self.HEIGHT / 2   
        boxes = []
        for i, bx in enumerate(xs):
            by = y_center + self.rng.uniform(-0.3, 0.3)
            boxes.append(Box(bx, by, i))
        return boxes

    def reset(self, new_seed=None):
        if new_seed is not None:
            self.rng   = np.random.default_rng(new_seed)
            self.boxes = self._place_boxes()
        else:
            for b in self.boxes:
                b.reset()

    def robot_box_contact(self, rx, ry, theta, box, push_dist=0.65):
        """
        Detecta si el frente del robot toca la caja.
        Devuelve (en_contacto, fx, fy) donde (fx, fy) es la fuerza de empuje [N].
        """
        fx_pt = rx + push_dist * np.cos(theta)
        fy_pt = ry + push_dist * np.sin(theta)
        h = box.SIZE / 2 + 0.1          
        in_contact = (abs(fx_pt - box.x) < h and abs(fy_pt - box.y) < h)
        if not in_contact:
            return False, 0., 0.
        F_MAX = 80.0                      
        return True, F_MAX * np.cos(theta), F_MAX * np.sin(theta)

    def is_box_cleared(self, box):
        """True si la caja salió de la zona bloqueada."""
        return (box.x < self.BLOCK_X_MIN - 0.3 or
                box.x > self.BLOCK_X_MAX + 0.3 or
                box.y < self.CLEAR_Y            or
                box.y > self.BLOCK_Y_MAX + 1.0)

    def all_boxes_cleared(self):
        return all(self.is_box_cleared(b) for b in self.boxes)

    def reached_goal(self, x, y):
        return np.hypot(x - self.goal[0], y - self.goal[1]) < self.goal_r

    def out_of_bounds(self, x, y):
        return (x < 0.3 or x > self.WIDTH  - 0.3 or
                y < 0.3 or y > self.HEIGHT - 0.3)

class StateExtractor:
    """
    Extrae el vector de estado para los modelos ML.

    Dimensión: 10 + 5 × N_BOXES  (fija independientemente del nº de cajas)

    Features fijas [0..9]:
        x, y, sinθ, cosθ, v, ω, dist_goal, angle_goal, frac_cleared, phase

    Features por caja [10+5i .. 14+5i]:
        dist_box, angle_box, cleared, dx_push_optimo, dy_push_optimo
    """

    def __init__(self, arena: Arena, husky: HuskyA200):
        self.arena = arena
        self.husky = husky
        self.n     = arena.n_boxes
        self.dim   = 10 + 5 * self.n

    def extract(self, phase=0):
        """Retorna vector de estado como np.float32."""
        a  = self.arena
        h  = self.husky
        rx, ry, rt = h.get_pose()
        gx, gy     = a.goal

        dist_goal  = np.hypot(rx - gx, ry - gy)
        angle_goal = np.arctan2(gy - ry, gx - rx) - rt
        angle_goal = np.arctan2(np.sin(angle_goal), np.cos(angle_goal))
        n_cleared  = sum(a.is_box_cleared(b) for b in a.boxes)

        fixed = [
            rx / a.WIDTH,
            ry / a.HEIGHT,
            np.sin(rt),
            np.cos(rt),
            h.v     / h.max_v,
            h.omega / h.max_omega,
            dist_goal / np.hypot(a.WIDTH, a.HEIGHT),
            angle_goal / np.pi,
            n_cleared / self.n,
            float(phase),
        ]

        per_box = []
        for box in a.boxes:
            dist_box  = np.hypot(rx - box.x, ry - box.y)
            angle_box = np.arctan2(box.y - ry, box.x - rx) - rt
            angle_box = np.arctan2(np.sin(angle_box), np.cos(angle_box))
            cleared   = float(a.is_box_cleared(box))
            cx = (box.x - a.WIDTH / 2) / a.WIDTH
            cy = -0.5
            per_box += [
                dist_box  / np.hypot(a.WIDTH, a.HEIGHT),
                angle_box / np.pi,
                cleared,
                cx,
                cy,
            ]

        return np.array(fixed + per_box, dtype=np.float32)


class HeuristicBoxPusher:
    """
    Controlador heurístico en 3 fases para el Husky.
    Genera tuplas (state, action_label, v_cmd, omega_cmd) para entrenar RF y Ridge.

    Fases:
        approach → alinearse al sur de la caja objetivo
        align    → orientar el robot hacia el norte (θ ≈ π/2)
        push     → avanzar sobre la caja hasta despejarla
        navigate → todas las cajas despejadas → ir a la meta

    Etiquetas de acción (para Random Forest):
        0  APPROACH_BOX   — acercarse a la caja objetivo
        1  ALIGN_PUSH     — alinearse para el eje de empuje
        2  PUSH_BOX       — empujar la caja
        3  NAVIGATE_GOAL  — avanzar hacia la meta
        4  ROTATE         — girar en sitio (corrección angular)
        5  RETREAT        — retroceder (anti-bloqueo)
    """

    ACTION_NAMES = {
        0: 'APPROACH_BOX',
        1: 'ALIGN_PUSH',
        2: 'PUSH_BOX',
        3: 'NAVIGATE_GOAL',
        4: 'ROTATE',
        5: 'RETREAT',
    }

    def __init__(self, arena, husky, state_extractor):
        self.arena = arena
        self.husky = husky
        self.se    = state_extractor
        self.reset()

    def reset(self):
        self.phase         = 'push'
        self.target_box    = None
        self.push_dir      = None
        self.sub_phase     = 'approach'
        self.stuck_counter = 0
        self.prev_pos      = None

    def _select_target_box(self):
        """Caja no despejada más cercana al robot."""
        rx, ry, _ = self.husky.get_pose()
        candidates = [b for b in self.arena.boxes
                      if not self.arena.is_box_cleared(b)]
        if not candidates:
            return None
        return min(candidates, key=lambda b: np.hypot(rx - b.x, ry - b.y))

    def _go_to_point(self, tx, ty, speed=0.55):
        """
        Genera (v, ω) para mover el robot hacia (tx, ty).
        Control proporcional puro con reducción de velocidad por error angular.
        """
        rx, ry, rt = self.husky.get_pose()
        dist  = np.hypot(tx - rx, ty - ry)
        angle = np.arctan2(ty - ry, tx - rx) - rt
        angle = np.arctan2(np.sin(angle), np.cos(angle))

        omega = np.clip(1.8 * angle, -1.8, 1.8)

        if abs(angle) > 0.6:         
            v = 0.05
        elif dist < 0.3:            
            v = 0.0
        else:
            v = speed * min(1.0, dist / 1.5)

        return v, omega

    def step(self, dt=0.05, noise=0.03):
        """
        Decide la acción del ciclo actual y actualiza la simulación.

        Retorna:
            action_id : int    (etiqueta para RF)
            v_cmd     : float  (objetivo para Ridge)
            omega_cmd : float
            state     : np.ndarray
            phase_int : int    (0=empujar, 1=navegar)
        """
        rx, ry, rt = self.husky.get_pose()
        arena      = self.arena
        phase_int  = 0 if self.phase == 'push' else 1
        state      = self.se.extract(phase=phase_int)

        if self.prev_pos is not None:
            moved = np.hypot(rx - self.prev_pos[0], ry - self.prev_pos[1])
            self.stuck_counter = self.stuck_counter + 1 if moved < 0.015 else 0
        self.prev_pos = (rx, ry)

        if self.stuck_counter > 40:
            self.stuck_counter = 0
            v_cmd, omega_cmd   = -0.3, 1.4 * (1 if np.random.rand() > 0.5 else -1)
            action_id          = 5  
            self.husky.step(v_cmd, omega_cmd, dt)
            return action_id, v_cmd, omega_cmd, state, phase_int

        if self.phase == 'push' and arena.all_boxes_cleared():
            self.phase = 'navigate'

        if self.phase == 'push':
            if self.target_box is None or arena.is_box_cleared(self.target_box):
                self.target_box = self._select_target_box()
                self.sub_phase  = 'approach'
                if self.target_box is not None:
                    self.push_dir = -1 if self.target_box.x > arena.WIDTH / 2 else 1

            if self.target_box is not None:
                box = self.target_box

                if self.sub_phase == 'approach':
                    tx, ty = box.x, box.y - 1.2
                    if np.hypot(rx - tx, ry - ty) < 0.35:
                        self.sub_phase = 'align'
                    v_cmd, omega_cmd = self._go_to_point(tx, ty, speed=0.5)
                    action_id = 0  

                elif self.sub_phase == 'align':
                    angle_err = np.arctan2(np.sin(np.pi/2 - rt),
                                            np.cos(np.pi/2 - rt))
                    if abs(angle_err) < 0.12:
                        self.sub_phase = 'push'
                    v_cmd     = 0.0
                    omega_cmd = np.clip(2.0 * angle_err, -1.5, 1.5)
                    action_id = 1 

                else: 
                    v_cmd, omega_cmd = self._go_to_point(
                        box.x, box.y + 0.1, speed=0.65)
                    action_id = 2  
                    if arena.is_box_cleared(box):
                        self.target_box = None
                        self.sub_phase  = 'approach'
            else:
                v_cmd, omega_cmd, action_id = 0., 0., 4   
        else:
            gx, gy = arena.goal
            v_cmd, omega_cmd = self._go_to_point(gx, gy, speed=0.7)
            action_id = 3 

        v_cmd     += np.random.normal(0, noise)
        omega_cmd += np.random.normal(0, noise * 1.5)

        for box in arena.boxes:
            contact, fx, fy = arena.robot_box_contact(rx, ry, rt, box)
            if contact and not arena.is_box_cleared(box):
                box.apply_push(fx, fy, dt)

        self.husky.step(v_cmd, omega_cmd, dt)
        return action_id, v_cmd, omega_cmd, state, phase_int


def run_episode(arena, husky, max_steps=3000, dt=0.05, noise=0.03, seed=None):
    """
    Ejecuta un episodio completo con el controlador heurístico.

    Retorna:
        states    : (N, dim)  float32
        actions   : (N,)      int
        targets   : (N, 2)    float32  — [v_cmd, omega_cmd]
        traj      : lista de (x, y, theta)
        box_trajs : dict {box_id: lista de (x, y)}
        success   : bool
        info      : dict con métricas del episodio
    """
    arena.reset(new_seed=seed)
    sx, sy, st = arena.start
    husky.reset(sx, sy, st)

    se   = StateExtractor(arena, husky)
    ctrl = HeuristicBoxPusher(arena, husky, se)

    states, actions, targets = [], [], []
    traj       = []
    box_trajs  = {b.id: [] for b in arena.boxes}
    success    = False
    boxes_cleared_order = []

    for step in range(max_steps):
        rx, ry, rt = husky.get_pose()
        traj.append((rx, ry, rt))
        for b in arena.boxes:
            box_trajs[b.id].append((b.x, b.y))

        if arena.all_boxes_cleared() and arena.reached_goal(rx, ry):
            success = True
            break

        for b in arena.boxes:
            if arena.is_box_cleared(b) and b.id not in boxes_cleared_order:
                boxes_cleared_order.append(b.id)

        if arena.out_of_bounds(rx, ry):
            break

        action_id, v_cmd, omega_cmd, state, _ = ctrl.step(dt, noise)
        states.append(state)
        actions.append(action_id)
        targets.append([v_cmd, omega_cmd])

    info = {
        'steps'               : step,
        'boxes_cleared'       : len(boxes_cleared_order),
        'boxes_cleared_order' : boxes_cleared_order,
        'push_counts'         : {b.id: b.push_count for b in arena.boxes},
    }
    return (np.array(states,  dtype=np.float32),
            np.array(actions, dtype=int),
            np.array(targets, dtype=np.float32),
            traj, box_trajs, success, info)


N_EPISODES = 80
husky = HuskyA200()
arena = Arena(n_boxes=N_BOXES)

all_states, all_actions, all_targets = [], [], []
successes = 0
best_traj = best_box_trajs = best_info = None
best_steps = np.inf

print(f'\nRecolectando {N_EPISODES} episodios (N_BOXES={N_BOXES})...')
for ep in range(N_EPISODES):
    ep_noise = 0.01 + 0.06 * np.random.rand()
    S, A, T, traj, btrajs, suc, info = run_episode(
        arena, husky, noise=ep_noise, seed=SEED + ep)

    if len(S) > 5:
        all_states.append(S)
        all_actions.append(A)
        all_targets.append(T)

    if suc:
        successes += 1
        if info['steps'] < best_steps:
            best_steps     = info['steps']
            best_traj      = traj
            best_box_trajs = btrajs
            best_info      = info

    if (ep + 1) % 20 == 0:
        print(f'  Ep {ep+1:3d}/{N_EPISODES} | Éxitos acumulados: {successes}')

X     = np.vstack(all_states)
y_cls = np.concatenate(all_actions)
y_reg = np.vstack(all_targets)

print(f'\n✅ Dataset recolectado')
print(f'   Total muestras : {X.shape[0]:,}')
print(f'   Features (dim) : {X.shape[1]}')
print(f'   Éxito          : {successes}/{N_EPISODES} ({successes/N_EPISODES:.1%})')
print(f'\n   Distribución de acciones:')
u, c = np.unique(y_cls, return_counts=True)
for ui, ci in zip(u, c):
    bar = '█' * int(30 * ci / len(y_cls))
    print(f'   {HeuristicBoxPusher.ACTION_NAMES[ui]:15s} {bar} {ci:5d} ({ci/len(y_cls):.1%})')


def draw_arena(ax, arena_ref, boxes_snapshot=None):
    """
    Dibuja la arena en un eje matplotlib.
    boxes_snapshot = {box_id: (x_final, y_final)} para mostrar posición final.
    """
    ax.set_facecolor('#0d1117')
    ax.add_patch(plt.Rectangle((0, 0), arena_ref.WIDTH, arena_ref.HEIGHT,
                                fill=False, edgecolor='#58a6ff', lw=2))
    ax.add_patch(plt.Rectangle(
        (arena_ref.BLOCK_X_MIN, arena_ref.BLOCK_Y_MIN),
        arena_ref.BLOCK_X_MAX - arena_ref.BLOCK_X_MIN,
        arena_ref.BLOCK_Y_MAX - arena_ref.BLOCK_Y_MIN,
        alpha=0.08, facecolor='#f85149', edgecolor='#f85149', ls='--', lw=1))

    for i, box in enumerate(arena_ref.boxes):
        bx, by = (boxes_snapshot[box.id]
                  if boxes_snapshot else (box.x0, box.y0))
        h2 = box.SIZE / 2
        ax.add_patch(plt.Rectangle(
            (bx - h2, by - h2), box.SIZE, box.SIZE,
            facecolor=BOX_COLORS[i % len(BOX_COLORS)],
            edgecolor='white', lw=1.2, alpha=0.75, zorder=4))
        ax.text(bx, by, f'C{i+1}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='#0d1117', zorder=5)

    gx, gy = arena_ref.goal
    ax.add_patch(plt.Circle((gx, gy), arena_ref.goal_r, color='#3fb950', alpha=0.2))
    ax.plot(gx, gy, '*', color='#3fb950', ms=14, zorder=6)
    ax.set_xlim(-0.3, arena_ref.WIDTH  + 0.3)
    ax.set_ylim(-0.3, arena_ref.HEIGHT + 0.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.1)


arena_viz = Arena(n_boxes=N_BOXES, seed=SEED)
fig, ax = plt.subplots(figsize=(7, 9))
fig.patch.set_facecolor('#0d1117')
draw_arena(ax, arena_viz)
sx, sy, st = arena_viz.start
robot_patch = plt.Circle((sx, sy), 0.45, color='#f0883e', alpha=0.85, zorder=7)
ax.add_patch(robot_patch)
ax.annotate('', xy=(sx + 0.7*np.cos(st), sy + 0.7*np.sin(st)),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle='->', color='white', lw=2.0))
ax.text(sx - 0.8, sy, 'Husky\nA200', color='#f0883e', fontsize=8.5)
ax.axhline(arena_viz.HEIGHT - 0.05, color='#3fb950', lw=3, alpha=0.6)
ax.text(arena_viz.WIDTH/2, arena_viz.HEIGHT + 0.2, '▼ SALIDA ▼',
        ha='center', color='#3fb950', fontsize=11, fontweight='bold')
ax.set_title(f'🤖 Pista Husky A200 — {N_BOXES} caja(s)', fontsize=13,
             color='white', pad=12)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
plt.tight_layout()
plt.savefig('arena_inicial.png', dpi=130, bbox_inches='tight')
plt.show()

if best_traj:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')

    ax = axes[0]
    draw_arena(ax, arena_viz)
    xs = [p[0] for p in best_traj]
    ys = [p[1] for p in best_traj]
    ts = [p[2] for p in best_traj]
    n  = len(xs)
    cmap_t = LinearSegmentedColormap.from_list('traj', ['#4ecdc4', '#f0883e', '#f85149'])
    for i in range(0, n-1, max(1, n//300)):
        ax.plot(xs[i:i+2], ys[i:i+2], color=cmap_t(i/n), lw=1.8, alpha=0.7)
    for i in range(0, n, max(1, n//15)):
        ax.annotate('', xy=(xs[i]+0.4*np.cos(ts[i]), ys[i]+0.4*np.sin(ts[i])),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.2, alpha=0.6))
    ax.plot(xs[0],  ys[0],  'o', color='#3fb950', ms=10, zorder=8, label='Inicio')
    ax.plot(xs[-1], ys[-1], 's', color='#f85149', ms=10, zorder=8, label='Fin')
    ax.set_title(f'Trayectoria Husky (heurístico)\n'
                 f'Pasos: {best_steps}  |  Cajas: {best_info["boxes_cleared"]}/{N_BOXES}',
                 color='white', fontsize=11)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    draw_arena(ax2, arena_viz)
    for i, box in enumerate(arena_viz.boxes):
        bxs = [p[0] for p in best_box_trajs[box.id]]
        bys = [p[1] for p in best_box_trajs[box.id]]
        c   = BOX_COLORS[i % len(BOX_COLORS)]
        ax2.plot(bxs, bys, '-', color=c, lw=2.5, alpha=0.8, label=f'Caja C{i+1}')
        ax2.plot(bxs[0],  bys[0],  'o', color=c, ms=9, zorder=7)
        ax2.plot(bxs[-1], bys[-1], 'X', color=c, ms=11, zorder=7)
        if len(bxs) > 5:
            ax2.annotate('', xy=(bxs[-1], bys[-1]), xytext=(bxs[0], bys[0]),
                         arrowprops=dict(arrowstyle='->', color=c, lw=1.5,
                                          connectionstyle='arc3,rad=0.2'))
    ax2.set_title(f'Trayectorias de las Cajas\n'
                  f'Orden de despeje: {best_info["boxes_cleared_order"]}',
                  color='white', fontsize=11)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    ax2.legend(fontsize=9, loc='upper right')

    plt.suptitle('🤖 Mejor demostración heurística — Husky A200',
                 fontsize=13, color='white', y=1.01)
    plt.tight_layout()
    plt.savefig('demo_heuristico.png', dpi=130, bbox_inches='tight')
    plt.show()


X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_cls, y_reg, test_size=0.20, random_state=SEED, stratify=y_cls)

print(f'\nTrain: {X_tr.shape[0]:,}  |  Test: {X_te.shape[0]:,}  |  Features: {X.shape[1]}')

rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        max_features='sqrt',
        n_jobs=-1,
        random_state=SEED
    ))
])

rf.fit(X_tr, yc_tr)
yc_pred = rf.predict(X_te)
acc     = accuracy_score(yc_te, yc_pred)
cv_acc  = cross_val_score(rf, X, y_cls, cv=5, scoring='accuracy', n_jobs=-1)

print(f'\n📊 Random Forest Classifier')
print(f'   Accuracy test   : {acc:.4f}  ({acc*100:.2f}%)')
print(f'   CV Accuracy 5-k : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
action_labels = [HeuristicBoxPusher.ACTION_NAMES[i] for i in
                 range(len(HeuristicBoxPusher.ACTION_NAMES))]
print(f'\n{classification_report(yc_te, yc_pred, target_names=action_labels, zero_division=0)}')

feature_names = (
    ['x', 'y', 'sinθ', 'cosθ', 'v', 'ω', 'd_goal', 'α_goal',
     'cleared_frac', 'phase'] +
    [f'C{i//5+1}_{["dist","angle","cleared","dx","dy"][i%5]}'
     for i in range(5 * N_BOXES)]
)

importances = rf.named_steps['clf'].feature_importances_
sort_idx    = np.argsort(importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0d1117')

ax = axes[0]
ax.set_facecolor('#161b22')
top_n   = min(20, len(importances))
idx_top = sort_idx[:top_n]

def feat_color(name):
    if name.startswith('C'):        return '#ffd700'
    if 'goal' in name or 'cleared' in name: return '#3fb950'
    return '#58a6ff'

colors = [feat_color(feature_names[i]) for i in idx_top]
ax.barh(range(top_n), importances[idx_top], color=colors, alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in idx_top], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Importancia Gini', color='#c9d1d9')
ax.set_title('🌲 RF — Top features', color='white', fontsize=12)
ax.legend(handles=[
    mpatches.Patch(color='#ffd700', label='Info cajas'),
    mpatches.Patch(color='#3fb950', label='Meta / progreso'),
    mpatches.Patch(color='#58a6ff', label='Estado robot'),
], fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')

ax = axes[1]
ax.set_facecolor('#161b22')
present_labels = sorted(np.unique(np.concatenate([yc_te, yc_pred])))
present_names  = [HeuristicBoxPusher.ACTION_NAMES[i] for i in present_labels]
cm      = confusion_matrix(yc_te, yc_pred, labels=present_labels)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
n_lab = len(present_labels)
ax.set_xticks(range(n_lab)); ax.set_yticks(range(n_lab))
ax.set_xticklabels(present_names, rotation=35, ha='right', fontsize=8)
ax.set_yticklabels(present_names, fontsize=8)
for i in range(n_lab):
    for j in range(n_lab):
        ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='white' if cm_norm[i,j] > 0.55 else '#666')
ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
ax.set_title(f'🎯 Matriz de Confusión  |  Acc={acc:.3f}', color='white', fontsize=12)

plt.suptitle('Random Forest — Clasificación de Acciones de Empuje',
             fontsize=13, color='white')
plt.tight_layout()
plt.savefig('rf_boxpush.png', dpi=130, bbox_inches='tight')
plt.show()

ridge_v = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.8))])
ridge_w = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.8))])

ridge_v.fit(X_tr, yr_tr[:, 0])
ridge_w.fit(X_tr, yr_tr[:, 1])

v_pred = ridge_v.predict(X_te)
w_pred = ridge_w.predict(X_te)

r2_v  = r2_score(yr_te[:, 0], v_pred)
r2_w  = r2_score(yr_te[:, 1], w_pred)
mse_v = mean_squared_error(yr_te[:, 0], v_pred)
mse_w = mean_squared_error(yr_te[:, 1], w_pred)

cv_v = cross_val_score(ridge_v, X, y_reg[:, 0], cv=5, scoring='r2')
cv_w = cross_val_score(ridge_w, X, y_reg[:, 1], cv=5, scoring='r2')

print(f'\n📊 Ridge Regression')
print(f'   v     | MSE={mse_v:.5f}  R²={r2_v:.4f}  '
      f'CV-R²={cv_v.mean():.4f}±{cv_v.std():.4f}')
print(f'   omega | MSE={mse_w:.5f}  R²={r2_w:.4f}  '
      f'CV-R²={cv_w.mean():.4f}±{cv_w.std():.4f}')

coef_v = np.abs(ridge_v.named_steps['ridge'].coef_)
coef_w = np.abs(ridge_w.named_steps['ridge'].coef_)
print('\n   Top-5 features para v:')
for i in np.argsort(coef_v)[::-1][:5]:
    print(f'     {feature_names[i]:20s}  coef={coef_v[i]:.4f}')
print('   Top-5 features para ω:')
for i in np.argsort(coef_w)[::-1][:5]:
    print(f'     {feature_names[i]:20s}  coef={coef_w[i]:.4f}')

N_VIZ = min(500, len(yr_te))
idx   = np.random.choice(len(yr_te), N_VIZ, replace=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor('#0d1117')

for ax, true, pred, label, unit, r2, color in [
    (axes[0], yr_te[idx, 0], v_pred[idx], 'v',   'm/s',   r2_v, '#58a6ff'),
    (axes[1], yr_te[idx, 1], w_pred[idx], 'ω', 'rad/s', r2_w, '#3fb950'),
]:
    ax.set_facecolor('#161b22')
    err = np.abs(true - pred)
    sc  = ax.scatter(true, pred, c=err, cmap='YlOrRd', alpha=0.55, s=15,
                     vmin=0, vmax=np.percentile(err, 90))
    mn, mx = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], '--', color=color, lw=1.8, label='Ideal')
    plt.colorbar(sc, ax=ax, label='|error|', fraction=0.046)
    ax.set_xlabel(f'{label} real [{unit}]')
    ax.set_ylabel(f'{label} pred [{unit}]')
    ax.set_title(f'Ridge — {label}   R²={r2:.4f}', color='white', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

ax = axes[2]
ax.set_facecolor('#161b22')
top_n_coef = 12
all_top = list(set(
    list(np.argsort(coef_v)[::-1][:top_n_coef]) +
    list(np.argsort(coef_w)[::-1][:top_n_coef])
))[:top_n_coef]
y_pos = np.arange(len(all_top))
ax.barh(y_pos - 0.2, coef_v[all_top], 0.35, color='#58a6ff', alpha=0.8, label='coef v')
ax.barh(y_pos + 0.2, coef_w[all_top], 0.35, color='#3fb950', alpha=0.8, label='coef ω')
ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in all_top], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('|Coeficiente|')
ax.set_title('Ridge — Coeficientes', color='white', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='x')

plt.suptitle('Ridge Regression — Control Continuo (v, ω) para Empuje de Cajas',
             fontsize=13, color='white')
plt.tight_layout()
plt.savefig('ridge_boxpush.png', dpi=130, bbox_inches='tight')
plt.show()

def run_ml_episode(arena, husky, policy_fn, max_steps=3000, dt=0.05, seed=SEED):
    """
    Episodio completo usando un modelo ML como controlador.

    policy_fn(state: np.ndarray) → (v_cmd: float, omega_cmd: float)

    Retorna: traj, box_trajs, success, steps, boxes_cleared
    """
    arena.reset(new_seed=seed)
    sx, sy, st = arena.start
    husky.reset(sx, sy, st)
    se = StateExtractor(arena, husky)

    traj, box_trajs = [], {b.id: [] for b in arena.boxes}
    success = False
    phase   = 0

    for step in range(max_steps):
        rx, ry, rt = husky.get_pose()
        traj.append((rx, ry, rt))
        for b in arena.boxes:
            box_trajs[b.id].append((b.x, b.y))

        if arena.all_boxes_cleared():
            phase = 1
        if arena.all_boxes_cleared() and arena.reached_goal(rx, ry):
            success = True
            break
        if arena.out_of_bounds(rx, ry):
            break

        state = se.extract(phase=phase)
        v_cmd, omega_cmd = policy_fn(state)

        for box in arena.boxes:
            contact, fx, fy = arena.robot_box_contact(rx, ry, rt, box)
            if contact and not arena.is_box_cleared(box):
                box.apply_push(fx, fy, dt)

        husky.step(v_cmd, omega_cmd, dt)

    cleared = sum(arena.is_box_cleared(b) for b in arena.boxes)
    return traj, box_trajs, success, step, cleared


ACTION_VW = {
    0: ( 0.50,  0.0),   
    1: ( 0.00,  1.2),   
    2: ( 0.65,  0.0),   
    3: ( 0.70,  0.0),   
    4: ( 0.00,  1.2),   
    5: (-0.30,  1.2),   
}

def rf_policy(state):
    """Random Forest: clasifica la acción y usa velocidades nominales."""
    action_id = int(rf.predict([state])[0])
    return ACTION_VW[action_id]

def ridge_policy(state):
    """Ridge Regression: predice (v, ω) directamente."""
    return (float(ridge_v.predict([state])[0]),
            float(ridge_w.predict([state])[0]))

def hybrid_policy(state):
    """
    Política híbrida RF + Ridge:
        - RF clasifica la acción (qué hacer)
        - Ridge refina las velocidades (cómo hacerlo)
        - Combina 60% nominal + 40% Ridge
    """
    action_id        = int(rf.predict([state])[0])
    v_ridge          = float(ridge_v.predict([state])[0])
    w_ridge          = float(ridge_w.predict([state])[0])
    v_nom, w_nom     = ACTION_VW[action_id]
    return (0.6 * v_nom + 0.4 * v_ridge,
            0.6 * w_nom + 0.4 * w_ridge)


husky   = HuskyA200()
results = {}
print('\nEvaluando modelos...')
for name, policy in [('RF',             rf_policy),
                      ('Ridge',          ridge_policy),
                      ('Híbrido RF+Ridge', hybrid_policy)]:
    traj, btrajs, suc, steps, cleared = run_ml_episode(
        Arena(n_boxes=N_BOXES), husky, policy, seed=SEED)
    results[name] = dict(traj=traj, btrajs=btrajs, success=suc,
                         steps=steps, cleared=cleared)
    status = '✅' if suc else '❌'
    print(f'  {status} {name:20s} | Pasos: {steps:4d} | Cajas: {cleared}/{N_BOXES}')


policies_plot = ['RF', 'Ridge', 'Híbrido RF+Ridge']
traj_colors   = {'RF': '#58a6ff', 'Ridge': '#3fb950', 'Híbrido RF+Ridge': '#ffd700'}
arena_ref     = Arena(n_boxes=N_BOXES, seed=SEED)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor('#0d1117')

for ax, name in zip(axes, policies_plot):
    r   = results[name]
    col = traj_colors[name]
    draw_arena(ax, arena_ref)

    xs = [p[0] for p in r['traj']]
    ys = [p[1] for p in r['traj']]
    ts = [p[2] for p in r['traj']]
    n  = len(xs)
    if n > 1:
        cmap2 = LinearSegmentedColormap.from_list('t', [col + '55', col])
        for i in range(0, n-1, max(1, n//250)):
            ax.plot(xs[i:i+2], ys[i:i+2], color=cmap2(i/n), lw=1.8, alpha=0.75)
        for i in range(0, n, max(1, n//10)):
            ax.annotate('', xy=(xs[i]+0.35*np.cos(ts[i]),
                                ys[i]+0.35*np.sin(ts[i])),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=1.0, alpha=0.5))

    for i, box in enumerate(arena_ref.boxes):
        bxs = [p[0] for p in r['btrajs'][box.id]]
        bys = [p[1] for p in r['btrajs'][box.id]]
        c2  = BOX_COLORS[i % len(BOX_COLORS)]
        ax.plot(bxs, bys, '-', color=c2, lw=2, alpha=0.6)
        ax.plot(bxs[-1], bys[-1], 'X', color=c2, ms=9, zorder=7)

    if n > 0:
        ax.plot(xs[0],  ys[0],  'o', color='#3fb950', ms=10, zorder=8)
        ax.plot(xs[-1], ys[-1], 's', color=col,       ms=10, zorder=8)

    status = '✅ ÉXITO' if r['success'] else '❌ FALLO'
    ax.set_title(f'{name}\n{status}  |  Pasos: {r["steps"]}  |  '
                 f'Cajas: {r["cleared"]}/{N_BOXES}',
                 color='white', fontsize=10, pad=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

plt.suptitle('🤖 Husky A200 — Empuje de Cajas: RF vs Ridge vs Híbrido',
             fontsize=14, color='white', y=1.02)
plt.tight_layout()
plt.savefig('comparacion_modelos.png', dpi=130, bbox_inches='tight')
plt.show()

summary = pd.DataFrame({
    'Modelo'          : ['Heurístico (ref.)', 'Random Forest',
                         'Ridge Regression', 'Híbrido RF+Ridge'],
    'Tipo'            : ['Regla', 'Clasificación',
                         'Regresión', 'Clasificación+Regresión'],
    'Tarea ML'        : ['N/A', '¿Qué acción?',
                         '¿Qué (v,ω)?', '¿Qué acción? + refinar (v,ω)'],
    'Éxito'           : [
        '✅' if best_traj else '❌',
        '✅' if results['RF']['success'] else '❌',
        '✅' if results['Ridge']['success'] else '❌',
        '✅' if results['Híbrido RF+Ridge']['success'] else '❌',
    ],
    'Cajas despejadas': [
        f'{best_info["boxes_cleared"]}/{N_BOXES}' if best_traj else f'?/{N_BOXES}',
        f'{results["RF"]["cleared"]}/{N_BOXES}',
        f'{results["Ridge"]["cleared"]}/{N_BOXES}',
        f'{results["Híbrido RF+Ridge"]["cleared"]}/{N_BOXES}',
    ],
    'Pasos'           : [
        best_steps if best_traj else 'N/A',
        results['RF']['steps'],
        results['Ridge']['steps'],
        results['Híbrido RF+Ridge']['steps'],
    ],
})

print('\n' + '='*75)
print('  RESUMEN COMPARATIVO — Husky A200 Box Pushing')
print('='*75)
print(summary.to_string(index=False))
print('='*75)

print(f'\n📊 Métricas ML:')
print(f'   Random Forest  — Accuracy : {acc:.4f}  |  '
      f'CV: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
print(f'   Ridge (v)      — R²: {r2_v:.4f}  |  MSE: {mse_v:.5f}')
print(f'   Ridge (omega)  — R²: {r2_w:.4f}  |  MSE: {mse_w:.5f}')

print(f'\n🔧 Configuración:')
print(f'   Robot          : {HuskyA200.ROBOT_TYPE}  {HuskyA200.ROBOT_FOOTPRINT} m')
print(f'   Número de cajas: {N_BOXES}  ← modificar N_BOXES al inicio del archivo')
print(f'   Dataset        : {X.shape[0]:,} muestras × {X.shape[1]} features')
print(f'   Episodios      : {N_EPISODES}  ({successes} exitosos)')

modelos     = ['Heurístico', 'RF', 'Ridge', 'Híbrido']
colores_bar = ['#8b949e', '#58a6ff', '#3fb950', '#ffd700']

pasos = [
    best_steps if best_traj else 3000,
    results['RF']['steps'],
    results['Ridge']['steps'],
    results['Híbrido RF+Ridge']['steps'],
]
cajas = [
    best_info['boxes_cleared'] if best_traj else 0,
    results['RF']['cleared'],
    results['Ridge']['cleared'],
    results['Híbrido RF+Ridge']['cleared'],
]
exitos = [
    1 if best_traj else 0,
    1 if results['RF']['success'] else 0,
    1 if results['Ridge']['success'] else 0,
    1 if results['Híbrido RF+Ridge']['success'] else 0,
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0d1117')

for ax, vals, title, ylabel in [
    (axes[0], pasos,  'Pasos hasta completar (↓ mejor)',        'Pasos'),
    (axes[1], cajas,  f'Cajas despejadas / {N_BOXES} (↑ mejor)', 'Cajas'),
    (axes[2], exitos, 'Misión completada',                       '1=Sí / 0=No'),
]:
    ax.set_facecolor('#161b22')
    bars = ax.bar(modelos, vals, color=colores_bar, alpha=0.85, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.02,
                str(v), ha='center', fontsize=12,
                fontweight='bold', color='white')
    ax.set_title(title, color='white', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(vals) * 1.2 + 1)

plt.suptitle('📊 Comparación Final — Husky A200 Box Pushing',
             fontsize=13, color='white')
plt.tight_layout()
plt.savefig('resumen_final.png', dpi=130, bbox_inches='tight')
plt.show()

print('\n✅ Ejecución completada. Archivos guardados:')
for f in ['arena_inicial.png', 'demo_heuristico.png', 'rf_boxpush.png',
          'ridge_boxpush.png', 'comparacion_modelos.png', 'resumen_final.png']:
    print(f'   📁 {f}')