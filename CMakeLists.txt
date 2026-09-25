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

# Also registers the PYTHONPATH environment hook, without which the installed
# package is not importable (previously registered by rosidl_generate_interfaces).
ament_python_install_package(${PROJECT_NAME})

set(generated_script_dir "${CMAKE_CURRENT_BINARY_DIR}/generated_scripts")
file(MAKE_DIRECTORY "${generated_script_dir}")

file(WRITE "${generated_script_dir}/image_publisher" [=[
#!/usr/bin/env python3
from skeleton_detection.input.image_publisher import main


if __name__ == "__main__":
    main()
]=])

file(WRITE "${generated_script_dir}/iot_node" [=[
#!/usr/bin/env python3
from skeleton_detection.iot_node import main


if __name__ == "__main__":
    main()
]=])

if(NOT WIN32)
  file(CHMOD
    "${generated_script_dir}/image_publisher"
    "${generated_script_dir}/iot_node"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endif()

install(
  PROGRAMS
    "${generated_script_dir}/image_publisher"
    "${generated_script_dir}/iot_node"
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
