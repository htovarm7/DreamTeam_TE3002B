cmake_minimum_required(VERSION 3.10)
project(so101_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/feetech_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
