cmake_minimum_required(VERSION 3.12)
project(skeleton_detection)

if(NOT WIN32)
  find_program(SYSTEM_PYTHON_EXECUTABLE NAMES python3 PATHS /usr/bin NO_DEFAULT_PATH)
  if(SYSTEM_PYTHON_EXECUTABLE)
    # Keep ROS code generation on the system interpreter even if a project venv is active.
    set(Python3_EXECUTABLE "${SYSTEM_PYTHON_EXECUTABLE}" CACHE FILEPATH "System Python for ROS" FORCE)
    set(PYTHON_EXECUTABLE "${SYSTEM_PYTHON_EXECUTABLE}" CACHE FILEPATH "System Python for ROS" FORCE)
  endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonSkeleton.msg"
  "msg/SkeletonFrame.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_get_python_install_dir(python_install_dir)

install(
  DIRECTORY skeleton_detection/
  DESTINATION ${python_install_dir}/${PROJECT_NAME}
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

set(generated_script_dir "${CMAKE_CURRENT_BINARY_DIR}/generated_scripts")
file(MAKE_DIRECTORY "${generated_script_dir}")

file(WRITE "${generated_script_dir}/camera_reader_node" [=[
#!/usr/bin/env python3
from skeleton_detection.camera_reader import main


if __name__ == "__main__":
    main()
]=])

file(WRITE "${generated_script_dir}/skeleton_detection_node" [=[
#!/usr/bin/env python3
from skeleton_detection.skeleton_detection import main


if __name__ == "__main__":
    main()
]=])

if(NOT WIN32)
  file(CHMOD
    "${generated_script_dir}/camera_reader_node"
    "${generated_script_dir}/skeleton_detection_node"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endif()

install(
  PROGRAMS
    "${generated_script_dir}/camera_reader_node"
    "${generated_script_dir}/skeleton_detection_node"
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
