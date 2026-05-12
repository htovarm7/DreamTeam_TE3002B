#!/usr/bin/env bash
# build.sh — Compilar el workspace SO-ARM101
# Uso: ./build.sh
set -e

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash

cd "$WS_DIR"
colcon build --symlink-install \
             --packages-select so101_description so101_bringup \
             --cmake-args -DCMAKE_BUILD_TYPE=Release

echo ""
echo "✓ Build completado. Ejecuta:"
echo "  source $WS_DIR/install/setup.bash"
echo ""
echo "Launches disponibles:"
echo "  ros2 launch so101_bringup gazebo.launch.py          # Simulación Gazebo"
echo "  ros2 launch so101_bringup physical.launch.py        # Robot físico"
echo "  ros2 launch so101_bringup sync.launch.py            # Gazebo + físico en sync"
echo "  ros2 launch so101_description display.launch.py     # Solo URDF en RViz2"
