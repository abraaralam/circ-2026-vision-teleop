cmake_minimum_required(VERSION 3.8)
project(arm_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)

add_library(arm_control SHARED
  src/mini_arm_hardware.cpp
)

target_include_directories(arm_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(arm_control
  rclcpp
  rclcpp_lifecycle
  hardware_interface
  pluginlib
  std_msgs
)

pluginlib_export_plugin_description_file(hardware_interface plugin.xml)

install(
  TARGETS arm_control
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install python teleop scripts as executables
install(PROGRAMS
  arm_control/keyboard_teleop.py
  arm_control/keyboard_cartesian_ik_teleop.py
  arm_control/ps5_arm_teleop.py
  arm_control/ps5_cartesian_ik_teleop.py
  arm_control/logitech_joystick_ik.py
  arm_control/joint_state_offset.py
  arm_control/mediapipe_ik_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install models (for hand track)
install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()
