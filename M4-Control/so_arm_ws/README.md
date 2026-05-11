# SO-ARM100 — LQR Visual Servoing Pick & Place
### ROS2 Humble · MoveIt2 · RealSense D435 · Classical CV + Optimal Control

> **Course project — TE3002B Intelligent Robotics**
> Complete visual servoing pipeline: classical vision → task-space representation → LQR optimal control → closed-loop actuation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Optimal Control Formulation](#2-optimal-control-formulation)
3. [Classical Vision Pipeline](#3-classical-vision-pipeline)
4. [Package Structure](#4-package-structure)
5. [Full Pipeline Architecture](#5-full-pipeline-architecture)
6. [ROS2 Topics & Actions](#6-ros2-topics--actions)
7. [Build](#7-build)
8. [Running — Simulation](#8-running--simulation)
9. [Running — Real Hardware](#9-running--real-hardware)
10. [Analyzing Results](#10-analyzing-results)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

The system performs **pick-and-place manipulation** using a 6-DOF SO-ARM100 robot arm guided entirely by visual feedback from an Intel RealSense D435 depth camera.

The pipeline follows the visual servoing paradigm:

```
Camera  →  Classical CV  →  3-D Feature  →  LQR Controller  →  Arm Actuation  →  Camera (loop)
```

The controller is **not heuristic or reactive**. It is derived from an infinite-horizon **Linear Quadratic Regulator (LQR)** formulated on the task-space position error, with the optimal gain computed offline via the Discrete Algebraic Riccati Equation (DARE). The visual measurements are continuously fed back into the control law at 10 Hz, making the system a true closed-loop visual servo.

---

## 2. Optimal Control Formulation

### 2.1 Visual Features

The perception stage (described in §3) extracts the 3-D centroid of the target object in world coordinates:

```
s[k] = (X_obj[k], Y_obj[k], Z_obj[k])  ∈ ℝ³
```

This is the **visual feature vector**, updated every camera frame.

### 2.2 State, Control Input, and Task Variable

| Symbol | Definition |
|--------|-----------|
| `p_ee[k]` | End-effector (TCP) position in world frame — from forward kinematics via TF |
| `p_des[k]` | Desired grasp position = `s[k] + δ_phase` (visually determined, updated every step) |
| `e[k]` | Task-space error = `p_ee[k] − p_des[k]` ∈ ℝ³ |
| `u[k]` | Cartesian velocity command = `[vx, vy, vz]` ∈ ℝ³ (m/s) |

The phase offset `δ_phase` switches between:
- **PRE_GRASP**: `[0, 0, +0.10]` m — approach 10 cm above the object
- **GRASP**: `[0, 0, +0.02]` m — contact at half the object height

### 2.3 System Dynamics

The end-effector position is controlled via velocity commands. With Euler discretisation at step `dt`:

```
p_ee[k+1] = p_ee[k] + dt · u[k]
```

The error dynamics are therefore:

```
e[k+1] = e[k] + dt · u[k]
```

Written in state-space form:

```
x[k+1] = A x[k] + B u[k]

A = I₃          (identity — error persists without control)
B = dt · I₃     (velocity command integrates to position change)
```

### 2.4 Cost Function

The infinite-horizon LQR cost penalises position error and control effort:

```
J = Σ_{k=0}^∞  ( x[k]ᵀ Q x[k]  +  u[k]ᵀ R u[k] )

Q = diag(10, 10, 15)    [position error weights — z weighted higher for vertical approach]
R = diag(1,  1,  1)     [velocity effort weights]
```

### 2.5 Constraints

```
‖u[k]‖ ≤ V_MAX = 0.12 m/s     (Cartesian velocity safety limit)
joint limits enforced by MoveIt2 planner at each incremental step
```

### 2.6 Optimal Gain via DARE

The optimal state-feedback gain is computed **once at startup** by solving the Discrete Algebraic Riccati Equation:

```
P = Q + AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA

K = (R + BᵀPB)⁻¹ BᵀPA
```

**Optimal control law:**

```
u*[k] = −K e[k]
```

The closed-loop error dynamics `e[k+1] = (A − BK) e[k]` are asymptotically stable (all eigenvalues of `A − BK` strictly inside the unit circle), which is confirmed by printing eigenvalues at node startup.

### 2.7 Lyapunov Stability Certificate

The value function `V(e) = eᵀ P e` satisfies `V(e[k+1]) − V(e[k]) < 0` along all non-zero trajectories, providing a formal Lyapunov stability proof that is also visualised in the results plot.

---

## 3. Classical Vision Pipeline

Implemented in `vision_detector.py`. **No neural networks or deep learning are used.**

```
RealSense D435  ──►  RGB frame (640×480)  ──►  HSV conversion
                     Depth frame (aligned)
                            │
               ┌────────────▼────────────┐
               │  HSV thresholding       │  cv2.inRange() per object profile
               │  Morphological open/close│  7×7 elliptical kernel (noise removal)
               │  Contour extraction     │  cv2.findContours(RETR_EXTERNAL)
               │  Shape filters          │  aspect ratio + circularity ∈ profile range
               │  Image moments          │  m00, m10, m01 → centroid (cx, cy)
               │  Depth patch (7×7 median)│  robust Z estimate at centroid
               │  Pinhole backprojection │  X=(cx−ppx)Z/fx, Y=(cy−ppy)Z/fy
               └────────────┬────────────┘
                            │
               /vision/best_object  (PoseStamped, camera frame)
               /vision/markers      (RViz spheres + labels)
               /vision/debug_image  (annotated BGR frame)
               /vision/detections   (JSON list of all detections)
```

**Objects detected** (by colour profile): banana, green apple, red apple, orange, cereal box.

**Confidence metric**: `min(1, area/20000) × (0.5 + 0.5 × circularity)` — used to select the best pick target.

---

## 4. Package Structure

```
so_arm_ws/src/
├── so_arm100_description/          # URDF/XACRO robot model + RViz config
│   ├── urdf/so_arm100.urdf.xacro
│   └── launch/display.launch.py
│
├── so_arm100_moveit_config/        # MoveIt2: SRDF, OMPL, kinematics, controllers
│   ├── config/kinematics.yaml      # IK solver config
│   ├── config/ompl_planning.yaml   # RRTConnect planner
│   └── launch/move_group.launch.py
│
├── so_arm100_gazebo/               # Gazebo simulation world + ros2_control
│   ├── config/ros2_controllers.yaml
│   ├── worlds/pick_place.world
│   └── launch/sim.launch.py
│
└── so_arm100_pick_place/           # Vision + LQR visual servoing (main package)
    ├── so_arm100_pick_place/
    │   ├── realsense_node.py       # RealSense D435 driver (RGB + aligned depth)
    │   ├── vision_detector.py      # Classical CV: HSV + contours + moments + backproject
    │   ├── vision_node.py          # ZED2 simulation variant (red-cube only)
    │   ├── lqr_visual_servoing.py  # ★ LQR-PBVS controller (DARE, closed loop)
    │   ├── plot_vs_results.py      # Offline analysis: error plots + Lyapunov
    │   ├── pick_place_node.py      # Legacy open-loop state machine (reference)
    │   ├── joint_commander.py      # Direct joint trajectory interface
    │   ├── move_physical.py        # Feetech STS3215 servo control (lerobot)
    │   └── teach_poses.py          # Interactive pose teaching tool
    └── launch/
        ├── vs_launch.launch.py     # ★ Complete LQR visual servoing pipeline
        ├── vision_pick.launch.py   # Vision-only + RViz (debug / calibration)
        ├── pick_place.launch.py    # Legacy open-loop pipeline (simulation)
        └── test_joints.launch.py   # Joint commander test
```

---

## 5. Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENSOR LAYER                                       │
│                                                                              │
│   Intel RealSense D435                                                       │
│     RGB  640×480 @ 30 Hz   →  /camera/color/image_raw                      │
│     Depth 640×480 @ 30 Hz  →  /camera/aligned_depth/image_raw              │
│     Camera intrinsics      →  /camera/color/camera_info                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                    CLASSICAL VISION PIPELINE  (vision_detector.py)           │
│                                                                              │
│   HSV colour thresholding  →  morphological open/close                      │
│   →  contour detection     →  shape filtering (aspect ratio, circularity)   │
│   →  image moments (m00, m10, m01)  →  centroid (cx, cy)                   │
│   →  aligned-depth patch (7×7 median)  →  Z                                │
│   →  pinhole backprojection  →  (X, Y, Z) in camera frame                  │
│                                                                              │
│   Output: /vision/best_object  (PoseStamped, camera_color_optical_frame)    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │  TF: camera → world
┌────────────────────────────────▼────────────────────────────────────────────┐
│                    LQR VISUAL SERVOING  (lqr_visual_servoing.py)            │
│                                                                              │
│   Visual feature   s[k] = (X_obj, Y_obj, Z_obj)  in world frame            │
│   Desired pose     p_des[k] = s[k] + δ_phase                               │
│   Error            e[k]  = p_ee[k] − p_des[k]                              │
│                                                                              │
│   Optimal control  u*[k] = −K e[k]                                         │
│     K  ← DARE: P = Q + AᵀPA − AᵀPB(R+BᵀPB)⁻¹BᵀPA                        │
│                                                                              │
│   Next EE target   p_next = p_ee + u* · dt                                 │
│                                                                              │
│   Runs at 10 Hz in a dedicated thread — closed loop every DT = 0.1 s       │
│   Publishes: /vs/status  /vs/metrics  →  CSV log in ~/vs_logs/             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │  incremental Cartesian pose goal
┌────────────────────────────────▼────────────────────────────────────────────┐
│                    MOTION PLANNING  (MoveIt2 move_group)                    │
│                                                                              │
│   Planner: OMPL RRTConnect                                                  │
│   IK solver: configured in kinematics.yaml                                  │
│   Velocity scaling: 0.6×  |  Acceleration scaling: 0.4×                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │  JointTrajectory actions
┌────────────────────────────────▼────────────────────────────────────────────┐
│                    JOINT CONTROLLERS  (ros2_control @ 100 Hz)               │
│                                                                              │
│   arm_controller     — 5 joints: shoulder_pan, shoulder_lift,              │
│                                  elbow, wrist_flex, wrist_roll              │
│   gripper_controller — 2 joints: left_finger, right_finger                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                    ACTUATION                                                  │
│                                                                              │
│   Simulation : Gazebo (pick_place.world)                                    │
│   Hardware   : 6× Feetech STS3215 servos via serial (lerobot SDK)          │
└─────────────────────────────────────────────────────────────────────────────┘
         │  (robot moves → camera sees new state → loop repeats)
         └──────────────────────────────► back to SENSOR LAYER
```

### Servoing Phase State Machine

```
WAIT  →  PRE_GRASP  →  GRASP  →  CLOSE_GRIP  →  POST_GRASP  →  PLACE  →  OPEN_GRIP  →  WAIT
         (LQR)          (LQR)                     (LQR)          (LQR)
```

The LQR controller runs in **PRE_GRASP**, **GRASP**, **POST_GRASP**, and **PLACE** phases, continuously re-reading the camera and correcting the trajectory at every step.

---

## 6. ROS2 Topics & Actions

### Sensor Topics

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | `realsense_node` | BGR8 RGB stream (640×480 @ 30 Hz) |
| `/camera/aligned_depth/image_raw` | `sensor_msgs/Image` | `realsense_node` | 32FC1 depth aligned to colour (metres) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | `realsense_node` | Intrinsics: fx, fy, ppx, ppy, distortion |

### Vision Topics

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/vision/best_object` | `geometry_msgs/PoseStamped` | `vision_detector` | Highest-confidence object in camera frame |
| `/vision/detections` | `std_msgs/String` | `vision_detector` | JSON list of all detected objects |
| `/vision/markers` | `visualization_msgs/MarkerArray` | `vision_detector` | RViz spheres + text labels + approach arrows |
| `/vision/debug_image` | `sensor_msgs/Image` | `vision_detector` | Annotated BGR frame with contours and labels |

### LQR Visual Servoing Topics

| Topic | Type | Publisher | Description |
|-------|------|-----------|-------------|
| `/vs/status` | `std_msgs/String` | `lqr_visual_servoing` | Current phase (WAIT / PRE_GRASP / GRASP …) |
| `/vs/metrics` | `std_msgs/Float64MultiArray` | `lqr_visual_servoing` | `[t, ex, ey, ez, ‖e‖, vx, vy, vz, ‖v‖]` at each step |

### Control Actions

| Action Server | Type | Description |
|---------------|------|-------------|
| `/arm_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | 5-DOF arm trajectory execution |
| `/gripper_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Gripper open / close |

---

## 7. Build

```bash
cd so_arm_ws

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Build all packages
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

To rebuild only the pick-and-place package after code changes:

```bash
colcon build --packages-select so_arm100_pick_place
source install/setup.bash
```

---

## 8. Running — Simulation

Requires Gazebo Ignition and MoveIt2. Run each command in a separate terminal (always `source install/setup.bash` first).

### Terminal 1 — Gazebo simulation + MoveIt2 + RViz

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch so_arm100_gazebo sim.launch.py
```

Wait ~8 seconds for Gazebo and MoveIt2 to fully initialise.

### Terminal 2 — LQR Visual Servoing pipeline

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch so_arm100_pick_place vs_launch.launch.py sim:=true
```

This starts the vision detector and the LQR controller. Place a coloured object in the simulated scene. The arm will:
1. Detect the object via HSV + depth backprojection
2. Compute the optimal velocity `u* = −K e` at each step
3. Servo the end-effector to the pre-grasp position above the object
4. Descend to the grasp contact point
5. Close the gripper and lift the object
6. Place it at the fixed drop pose and reset

### Optional — URDF viewer only

```bash
ros2 launch so_arm100_description display.launch.py
```

---

## 9. Running — Real Hardware

Requires a RealSense D435 plugged in and the SO-ARM100 arm connected over USB serial.

### Terminal 1 — MoveIt2 (no Gazebo)

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch so_arm100_moveit_config move_group.launch.py
```

### Terminal 2 — RealSense + Vision + LQR controller

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch so_arm100_pick_place vs_launch.launch.py port:=/dev/ttyUSB0
```

> **Note:** Extrinsic (eye-to-hand) calibration between the camera and the arm base must be completed first. Use `easy_handeye2` to generate the static TF. Until calibrated, use the simulation.

### Terminal 3 (optional) — Monitor LQR metrics live

```bash
ros2 topic echo /vs/metrics
ros2 topic echo /vs/status
```

---

## 10. Analyzing Results

After any pick-and-place cycle completes, a CSV is written to `~/vs_logs/vs_YYYYMMDD_HHMMSS.csv`.

Run the analysis script to generate plots:

```bash
# Uses the most recent CSV automatically
python3 src/so_arm100_pick_place/so_arm100_pick_place/plot_vs_results.py

# Or pass a specific file
python3 src/so_arm100_pick_place/so_arm100_pick_place/plot_vs_results.py ~/vs_logs/vs_20240510_143022.csv
```

**Plots produced:**

| Plot | What it shows |
|------|--------------|
| Error components ex, ey, ez vs time | Convergence of each spatial axis |
| Error norm ‖e‖ vs time | Overall convergence with phase thresholds |
| Control velocity ‖v‖ vs time | LQR optimal effort — highest when far, tapers at convergence |
| Lyapunov function V(e) = eᵀPe vs time | Monotone decrease proves closed-loop stability |
| Phase timeline | PRE_GRASP / GRASP / POST_GRASP / PLACE durations |

The script also prints a summary: total duration, final error, cumulative LQR cost `J`, gain `K`, and closed-loop eigenvalues.

---

## 11. Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| ROS2 Humble | 22.04 LTS | Middleware |
| MoveIt2 | Humble | Motion planning + Cartesian control |
| ros2_control | Humble | Joint trajectory controllers |
| OpenCV | ≥ 4.5 | Classical computer vision pipeline |
| NumPy / SciPy | latest | LQR / DARE computation |
| pyrealsense2 | ≥ 2.50 | RealSense D435 SDK |
| cv_bridge | Humble | ROS ↔ OpenCV image conversion |
| tf2_ros | Humble | Coordinate frame transforms |
| lerobot | latest | Feetech STS3215 servo control (hardware only) |
| matplotlib | latest | Offline results analysis |

Install Python dependencies:

```bash
pip install numpy scipy matplotlib pyrealsense2 opencv-python
```

Install ROS2 dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y
```
