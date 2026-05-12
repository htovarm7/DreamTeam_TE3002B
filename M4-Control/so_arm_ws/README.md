# SO-ARM100 — LQR Visual Servoing Pick & Place
### ROS2 Humble · MoveIt2 · RealSense D435 · Classical CV + Optimal Control

> **Course project — TE3002B Intelligent Robotics**
> Complete visual servoing pipeline: classical vision → task-space representation → LQR optimal control → closed-loop actuation on physical Feetech STS3215 servos.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Package Structure](#2-package-structure)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Build](#4-build)
5. [Quick Start — Physical Robot](#5-quick-start--physical-robot)
6. [Testing the Pick](#6-testing-the-pick)
7. [Teaching Poses](#7-teaching-poses)
8. [ROS2 Topics & Actions](#8-ros2-topics--actions)
9. [Optimal Control Formulation](#9-optimal-control-formulation)
10. [Classical Vision Pipeline](#10-classical-vision-pipeline)
11. [Analyzing Results](#11-analyzing-results)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

Pick-and-place with a 6-DOF SO-ARM100 guided by visual feedback from a RealSense D435 depth camera. The controller is an infinite-horizon **Linear Quadratic Regulator (LQR)** with the optimal gain computed offline via the Discrete Algebraic Riccati Equation (DARE), running at 10 Hz in closed loop.

```
Camera → Classical CV → 3-D Feature → LQR Controller → IK → Feetech Motors → Camera (loop)
```

---

## 2. Package Structure

```
so_arm_ws/src/
├── so_arm100_description/          # URDF/XACRO robot model + meshes STL
│   ├── urdf/so_arm100.urdf.xacro
│   ├── urdf/so_arm100.ros2_control.xacro  # joint interfaces (mock_components)
│   └── config/ros2_controllers.yaml
│
├── so_arm100_moveit_config/        # MoveIt2: SRDF, OMPL, kinematics, controllers
│   ├── config/kinematics.yaml
│   ├── config/ompl_planning.yaml   # RRTConnect planner
│   ├── config/so_arm100.srdf
│   └── launch/move_group.launch.py
│
└── so_arm100_pick_place/           # Pipeline principal
    ├── so_arm100_pick_place/
    │   ├── feetech_bridge.py       # ★ Bridge ros2_control ↔ motores Feetech reales
    │   ├── realsense_node.py       # RealSense D435 driver (RGB + aligned depth)
    │   ├── vision_detector.py      # CV clásica: HSV + contornos + momentos + backproject
    │   ├── lqr_visual_servoing.py  # ★ Controlador LQR-PBVS (DARE, closed loop)
    │   ├── pick_place_node.py      # State machine pick & place (IK + trayectoria)
    │   ├── joint_commander.py      # Interfaz directa de joints por nombre
    │   ├── move_physical.py        # Control directo de servos Feetech (sin ROS2)
    │   ├── teach_poses.py          # Herramienta de enseñanza de poses
    │   ├── plot_vs_results.py      # Análisis offline: error + Lyapunov
    │   └── poses.json              # Poses guardadas (home, detections, etc.)
    └── launch/
        ├── physical.launch.py      # ★ Launch completo para robot físico
        ├── vision_pick.launch.py   # Solo visión + RViz (debug / calibración)
        └── test_joints.launch.py   # Test de joints por nombre
```

---

## 3. Pipeline Architecture

```
Intel RealSense D435
  RGB 640×480 @ 30 Hz   →  /camera/color/image_raw
  Depth 640×480 @ 30 Hz →  /camera/aligned_depth/image_raw
            │
            ▼
  vision_detector.py
    HSV threshold → morph open/close → contours → shape filter
    → image moments → centroid (cx, cy) → depth patch → backproject
    → /vision/best_object  (PoseStamped, camera_color_optical_frame)
            │   TF: camera → world
            ▼
  lqr_visual_servoing.py   (10 Hz, hilo dedicado)
    e[k] = p_ee[k] − p_des[k]
    u*[k] = −K e[k]       K ← DARE
    → /compute_ik (MoveIt2 KinematicsService)
    → /arm_controller/follow_joint_trajectory
            │
            ▼
  feetech_bridge.py
    /joint_states_mock → grados → lerobot → motores STS3215
    motores reales → radianes → /joint_states  (MoveIt2 / RViz ven posición real)
            │
            ▼
  6× Feetech STS3215 @ /dev/ttyACM0
```

### Estado del LQR (fase state machine)

```
WAIT → PRE_GRASP → GRASP → CLOSE_GRIP → POST_GRASP → PLACE → OPEN_GRIP → WAIT
        (LQR)       (LQR)                  (LQR)        (LQR)
```

---

## 4. Build

```bash
cd ~/Desktop/DreamTeam_TE3002B/M4-Control/so_arm_ws

# Primera vez: instalar dependencias Python
pip install "numpy<2" scipy matplotlib pyrealsense2 opencv-python

# Construir
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

> **IMPORTANTE:** `numpy<2` es requerido — cv_bridge y pyrealsense2 no son compatibles con NumPy 2.x.

---

## 5. Quick Start — Physical Robot

### Prerequisitos
- Brazo SO-ARM100 conectado en `/dev/ttyACM0`
- RealSense D435 conectada por USB
- Workspace construido y sourceado

### Lanzar todo

```bash
source ~/Desktop/DreamTeam_TE3002B/M4-Control/so_arm_ws/install/setup.bash
ros2 launch so_arm100_pick_place physical.launch.py
```

Al arrancar, el sistema:
1. Lanza `robot_state_publisher` + `ros2_control` + controladores
2. Conecta a motores Feetech → mueve a pose `detections`
3. Lanza `move_group` (MoveIt2)
4. Lanza `realsense_node` + `vision_detector`
5. Lanza `lqr_visual_servoing` (modo WAIT, auto-disparo)
6. Abre RViz con robot real + markers de detección

### Sin cámara (solo brazo)

```bash
ros2 launch so_arm100_pick_place physical.launch.py realsense:=false
```

### Apagar correctamente

`Ctrl+C` → el brazo va automáticamente a pose `home` antes de cortar torque.

---

## 6. Testing the Pick

### Paso 1 — Verificar que el sistema esté listo

```bash
# Estado del LQR
ros2 topic echo /vs/status

# Qué objetos detecta la cámara
ros2 topic echo /vision/detections
```

### Paso 2 — Poner un objeto frente a la cámara

El detector reconoce por color HSV:

| Color | Objeto |
|-------|--------|
| Amarillo | Banana |
| Verde | Manzana verde |
| Rojo | Manzana roja |
| Naranja | Naranja |
| Rectangular grande | Caja de cereal |

Cuando el detector ve el objeto, publica en `/vision/best_object`.

### Paso 3 — El LQR se dispara solo (`auto_start: True`)

```
WAIT  →  (objeto detectado)  →  PRE_GRASP
       el brazo se mueve 10 cm arriba del objeto
→  GRASP
       baja al objeto
→  CLOSE_GRIP
       cierra gripper
→  POST_GRASP
       sube
→  PLACE
       va a la posición de depósito fija
→  OPEN_GRIP
       suelta
→  WAIT  (listo para el siguiente)
```

### Monitorear en tiempo real

```bash
# Estado de la fase
ros2 topic echo /vs/status

# Error de posición e[x,y,z] + velocidad u[x,y,z] (se actualiza a 10 Hz)
ros2 topic echo /vs/metrics

# Posición real de los motores
ros2 topic echo /joint_states

# Ver imagen de detección con contornos
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

### Ver imagen de cámara

```bash
# En terminal nueva (con workspace sourceado):
ros2 run rqt_image_view rqt_image_view /camera/color/image_raw &
ros2 run rqt_image_view rqt_image_view /vision/debug_image &
```

---

## 7. Teaching Poses

Herramienta interactiva para guardar posiciones del brazo. El launch debe estar **detenido** (los motores libres).

```bash
python3 ~/Desktop/DreamTeam_TE3002B/M4-Control/so_arm_ws/src/so_arm100_pick_place/so_arm100_pick_place/teach_poses.py
```

**Comandos disponibles:**

| Comando | Acción |
|---------|--------|
| `ENTER` | Lee posición actual → pide nombre para guardar |
| `read` | Lee posición sin guardar |
| `list` | Lista todas las poses guardadas |
| `test <nombre>` | Prueba moviéndose a esa pose |
| `delete <nombre>` | Borra una pose |
| `done` / `quit` | Guarda `poses.json` y actualiza `move_physical.py` |

**Poses recomendadas:**

| Nombre | Para qué |
|--------|---------|
| `detections` | Posición inicial donde la cámara ve la zona de trabajo |
| `home` | Posición segura al apagar (brazo plegado, no cae) |
| `pre_grasp` | Referencia manual si se necesita (LQR lo calcula automático) |
| `place` | Zona de depósito (si se usa `pick_place_node` manual) |

Las poses se cargan automáticamente en `feetech_bridge.py`:
- Al arrancar → va a `detections`
- Al apagar (Ctrl+C) → va a `home`

### Mover a una pose sin ROS2

```bash
python3 move_physical.py detections
python3 move_physical.py home
python3 move_physical.py read      # solo leer posición actual
```

---

## 8. ROS2 Topics & Actions

### Cámara

| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | BGR8 640×480 @ 30 Hz |
| `/camera/aligned_depth/image_raw` | `sensor_msgs/Image` | Depth 32FC1 alineado a color |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Intrínsecos: fx, fy, ppx, ppy |

### Visión

| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/vision/best_object` | `geometry_msgs/PoseStamped` | Objeto con mayor confianza |
| `/vision/detections` | `std_msgs/String` | JSON: nombre, x, y, z, confianza |
| `/vision/markers` | `visualization_msgs/MarkerArray` | Esferas en RViz |
| `/vision/debug_image` | `sensor_msgs/Image` | Imagen con contornos dibujados |

### LQR

| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/vs/status` | `std_msgs/String` | Fase actual: WAIT / PRE_GRASP / GRASP… |
| `/vs/metrics` | `std_msgs/Float64MultiArray` | `[t, ex, ey, ez, ‖e‖, vx, vy, vz, ‖v‖]` |

### Joints

| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Posición REAL de los motores (publica feetech_bridge) |
| `/joint_states_mock` | `sensor_msgs/JointState` | Posición virtual del mock (ros2_control interno) |
| `/joint_commander/target` | `std_msgs/String` | Nombre de pose o valores CSV en rad |

### Actions

| Action | Tipo | Descripción |
|--------|------|-------------|
| `/arm_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Trayectoria de 5 joints del brazo |
| `/gripper_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Gripper abierto/cerrado |

---

## 9. Optimal Control Formulation

**Estado:** `e[k] = p_ee[k] − p_des[k]` (error Cartesiano ℝ³)

**Dinámica:** `e[k+1] = A e[k] + B u[k]`  con  `A = I₃`, `B = I₃`

**Costo:** `J = Σ eᵀQe + uᵀRu`  con  `Q = diag(10,10,15)`, `R = diag(1,1,1)`

**DARE:** `P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA`

**Ley óptima:** `u*[k] = −K e[k]`  →  K (diag) ≈ [0.916, 0.916, 0.941]

**Eigenvalores en lazo cerrado:** ≈ [0.084, 0.084, 0.059] — todos dentro del círculo unitario → estable.

---

## 10. Classical Vision Pipeline

Sin redes neuronales. Todo CV clásico (`vision_detector.py`):

```
RGB frame → cvtColor HSV → inRange (por perfil de objeto) → morphologyEx open/close
→ findContours → filtro (área mínima + aspect ratio + circularidad)
→ moments → centroid (cx, cy)
→ depth patch 7×7 median → Z
→ backproject: X=(cx−ppx)Z/fx, Y=(cy−ppy)Z/fy
→ /vision/best_object (PoseStamped en camera_color_optical_frame)
```

**Confianza:** `min(1, area/20000) × (0.5 + 0.5 × circularity)`

---

## 11. Analyzing Results

Cada ciclo de pick genera un CSV en `~/vs_logs/vs_YYYYMMDD_HHMMSS.csv`.

```bash
# Analizar el CSV más reciente
python3 src/so_arm100_pick_place/so_arm100_pick_place/plot_vs_results.py

# Analizar un archivo específico
python3 src/so_arm100_pick_place/so_arm100_pick_place/plot_vs_results.py ~/vs_logs/vs_20260511_121020.csv
```

Genera: error por eje, norma del error, velocidad LQR, función de Lyapunov `V(e) = eᵀPe`, y línea de tiempo de fases.

---

## 12. Dependencies

| Dependencia | Versión | Uso |
|-------------|---------|-----|
| ROS2 Humble | Ubuntu 22.04 | Middleware |
| MoveIt2 | Humble | Planificación + IK (`/compute_ik`) |
| ros2_control | Humble | Controladores de trayectoria |
| OpenCV | ≥ 4.5 | Pipeline de visión |
| NumPy | < 2.0 | LQR / DARE (compatibilidad cv_bridge) |
| SciPy | latest | `solve_discrete_are` para DARE |
| pyrealsense2 | ≥ 2.57 | Driver RealSense D435 |
| cv_bridge | Humble | Conversión ROS ↔ OpenCV |
| tf2_ros | Humble | Transformadas de coordenadas |
| lerobot | latest | Control servos Feetech STS3215 |
| matplotlib | latest | Análisis offline de resultados |

```bash
# Instalar Python deps
pip install "numpy<2" scipy matplotlib pyrealsense2 opencv-python

# Instalar deps ROS2
rosdep install --from-paths src --ignore-src -r -y
```
