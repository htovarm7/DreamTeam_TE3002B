cmake_minimum_required(VERSION 3.10)
project(so101_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf launch rviz config
  DESTINATION share/${PROJECT_NAME}
)

# Meshes live in SO101/assets at the workspace root (two levels up from src/so101_description)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../SO101/assets/
  DESTINATION share/${PROJECT_NAME}/meshes
  FILES_MATCHING PATTERN "*.stl"
)

ament_package()
