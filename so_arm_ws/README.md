# SO-ARM100 Pick & Place — ROS2 Humble + MoveIt2 + ZED2

## Arquitectura

```
so_arm_ws/src/
├── so_arm100_description/   # URDF/XACRO del brazo + ZED2 montada
├── so_arm100_moveit_config/ # SRDF, OMPL, cinemática, controladores MoveIt
├── so_arm100_gazebo/        # Mundo Gazebo, controladores ros2_control, launch
└── so_arm100_pick_place/    # Nodo visión (ZED2) + nodo pick & place
```

## Flujo completo

```
Gazebo (física)
  └─ ZED2 plugin → /zed2/rgb/image_rect_color
                 → /zed2/depth/depth_registered
                          │
                    vision_node  (HSV + depth back-projection)
                          │
                /vision/object_position (PointStamped, camera frame)
                          │ TF world←camera
                    pick_place_node  (máquina de estados)
                          │
                    MoveIt2 move_group  (OMPL RRTConnect)
                          │
                  arm_controller / gripper_controller  (ros2_control)
                          │
                    Gazebo joints
```

## Build

```bash
cd so_arm_ws
./build.sh
source install/setup.bash
```

## Ejecución

### Terminal 1 — Simulación completa (Gazebo + MoveIt + RViz)
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch so_arm100_gazebo sim.launch.py
```

### Terminal 2 — Visión + Pick & Place (esperar ~8 s a que Gazebo cargue)
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch so_arm100_pick_place pick_place.launch.py
```

### Sólo visualización del URDF en RViz
```bash
ros2 launch so_arm100_description display.launch.py
```

## Topics principales

| Topic | Tipo | Descripción |
|-------|------|-------------|
| `/zed2/rgb/image_rect_color` | `sensor_msgs/Image` | RGB del ZED2 simulado |
| `/zed2/depth/depth_registered` | `sensor_msgs/Image` | Profundidad 32FC1 |
| `/vision/object_position` | `geometry_msgs/PointStamped` | Posición del cubo (cam frame) |
| `/vision/debug_image` | `sensor_msgs/Image` | RGB con detección superpuesta |
| `/pick_place/status` | `std_msgs/String` | Estado actual de la máquina |
| `/arm_controller/follow_joint_trajectory` | Action | Trayectorias del brazo |
| `/gripper_controller/follow_joint_trajectory` | Action | Trayectorias del gripper |

## Próximos pasos

- [ ] Reemplazar meshes box por STL reales del SO-ARM100 (disponibles en HuggingFace)
- [ ] Calibración extrínsecos cámara↔brazo (eye-to-hand) con `easy_handeye2`
- [ ] Conectar el brazo físico vía `lerobot` / Feetech SDK
- [ ] Agregar pipeline YOLO para detección multi-objeto
