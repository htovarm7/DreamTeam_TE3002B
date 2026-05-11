cmake_minimum_required(VERSION 3.10)
project(so_arm100_pick_place)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  so_arm100_pick_place/pick_place_node.py
  so_arm100_pick_place/joint_commander.py
  so_arm100_pick_place/realsense_node.py
  so_arm100_pick_place/vision_detector.py
  so_arm100_pick_place/move_physical.py
  so_arm100_pick_place/teach_poses.py
  so_arm100_pick_place/lqr_visual_servoing.py
  so_arm100_pick_place/plot_vs_results.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
