FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# ROS 2 and system dependencies
RUN apt-get update && apt-get install -y \
        python3-pip \
        python3-dev \
        build-essential \
        ca-certificates \
        curl \
        gnupg \
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

# CUDA development tools
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm -f /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y \
        cuda-nvcc-13-0=13.0.88-1 \
        cuda-cudart-dev-13-0=13.0.96-1 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN python3 -m pip install --no-cache-dir \
        "pip==26.2.1" \
        "setuptools==79.0.1" \
        "wheel==0.48.0"

# Shared Python package constraints
RUN printf '%s\n' \
        'numpy==1.23.5' \
        'opencv-python==4.8.1.78' \
        'scipy==1.15.3' \
        'matplotlib==3.10.9' \
        'setuptools==79.0.1' \
        'Cython==0.29.37' \
        'torch==2.14.0+cu130' \
        'torchvision==0.29.0+cu130' \
        'mmengine==0.10.7' \
        'mmcv-lite==2.1.0' \
        'mmdet==3.3.0' \
        'mmpose==1.3.2' \
        'chumpy==0.70' \
        'xtcocotools==1.14.3' \
        'pycocotools==2.0.11' \
        'shapely==2.1.2' \
        'terminaltables==3.1.10' \
        'json-tricks==3.17.3' \
        'munkres==1.1.4' \
        > /etc/pip-constraints.txt

# PyTorch CUDA 13
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.14.0+cu130" \
        "torchvision==0.29.0+cu130"

RUN python3 -m pip install --no-cache-dir \
        -c /etc/pip-constraints.txt \
        "numpy==1.23.5" \
        "opencv-python==4.8.1.78" \
        "scipy==1.15.3" \
        "matplotlib==3.10.9" \
        "Cython==0.29.37"

RUN python3 -m pip install --no-cache-dir \
        --no-build-isolation \
        -c /etc/pip-constraints.txt \
        "chumpy==0.70" \
        "xtcocotools==1.14.3"

# OpenMMLab
RUN python3 -m pip install --no-cache-dir \
        --no-build-isolation \
        -c /etc/pip-constraints.txt \
        "mmengine==0.10.7" \
        "mmcv-lite==2.1.0" \
        "mmdet==3.3.0" \
        "mmpose==1.3.2"

# RTMO does not use mmcv native ops, but MMPose imports them at module load time.
# This stub allows those imports and intentionally fails if an op is actually called.
COPY docker/mmcv_ext_stub.py \
    /usr/local/lib/python3.10/dist-packages/mmcv/_ext.py

# Catch dependency/import regressions during docker build
RUN python3 -c "\
import torch, torchvision, mmengine, mmcv, mmdet, mmpose, xtcocotools, chumpy, numpy, cv2; \
from mmpose.apis import init_model, inference_bottomup; \
from mmpose.models.heads.hybrid_heads.rtmo_head import RTMOHead; \
print('RTMO import chain OK')"

WORKDIR /ros2_ws