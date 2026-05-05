#!/usr/bin/env bash
# Build the SO-ARM100 workspace
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Sourcing ROS2 Humble ==="
source /opt/ros/humble/setup.bash

echo "=== Installing Python deps ==="
pip3 install -q opencv-python numpy

echo "=== Building workspace ==="
colcon build --symlink-install \
             --cmake-args -DCMAKE_BUILD_TYPE=Release

echo ""
echo "=== Build complete! ==="
echo "Run:  source install/setup.bash"
echo "Then: ros2 launch so_arm100_gazebo sim.launch.py"
echo "Then (new terminal): ros2 launch so_arm100_pick_place pick_place.launch.py"
