FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
        python3-pip \
        ros-humble-cv-bridge \
        ros-humble-vision-opencv \
        ros-humble-sensor-msgs \
        ros-humble-geometry-msgs \
        ros-humble-realsense2-camera \
        ros-humble-rqt-image-view \
        ros-humble-rqt-gui \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
