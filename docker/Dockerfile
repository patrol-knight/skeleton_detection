FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# ROS 2 and system dependen     ies
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

# RealSense Python bindings for direct in-process capture.
# Version pinned to match the ros-humble-librealsense2 2.58.3 runtime already
# installed above; PyPI ships a manylinux2014_aarch64 cp310 wheel, so nothing
# is built from source. Needed because the RTMO node opens the D456 itself
# instead of consuming a realsense2_camera image topic.
RUN python3 -m pip install --no-cache-dir \
        -c /etc/pip-constraints.txt \
        "pyrealsense2==2.58.3.10794"

# BoT-SORT + ReID tracking stack. Pinned to 19.0.0 deliberately: it is the
# NEWEST BoxMOT release that leaves numpy unconstrained. BoxMOT >= 20.0.0
# requires numpy>=2.2, which would force out the numpy 1.23.5 that chumpy,
# opencv-python 4.8 and the compiled mmpose/xtcocotools wheels depend on.
# The constraints file is load-bearing here, not decorative: it keeps pip from
# pulling numpy 2.x in through a transitive dependency, and makes a future
# incompatible resolution fail loudly at build time instead of silently
# breaking the RTMO stack at runtime.
RUN python3 -m pip install --no-cache-dir \
        -c /etc/pip-constraints.txt \
        "boxmot==19.0.0"

# RTMO does not use mmcv native ops, but MMPose imports them at module load time.
# This stub allows those imports and intentionally fails if an op is actually called.
COPY docker/mmcv_ext_stub.py \
    /usr/local/lib/python3.10/dist-packages/mmcv/_ext.py

# Bake the runtime model assets into the image at /opt/models so a pulled
# image is self-contained. Previously these lived in the source tree's
# data/checkpoints/, which is both .gitignore'd and .dockerignore'd, so
# `git clone` + `docker pull` produced an environment with no weights.
#
# The RTMO config is version-controlled in this repo (models/rtmo/) and copied
# in here, so the config the image runs is the config you can read in git.
# It is two files: rtmo-m.py plus the default_runtime.py it inherits from.
COPY models/rtmo/ /opt/models/rtmo/

# The .pth/.pt weights are NOT in git (too large); they are downloaded here at
# build time from their official sources and sha256-verified, so a corrupted or
# substituted file fails the build instead of silently shipping. Because they
# live in an image layer, destroying and recreating a container never loses
# them.
COPY docker/fetch_models.py /tmp/fetch_models.py
RUN python3 /tmp/fetch_models.py && rm -f /tmp/fetch_models.py

ENV RTMO_MODEL_CONFIG=/opt/models/rtmo/rtmo-m.py \
    RTMO_CHECKPOINT=/opt/models/rtmo/rtmo-m.pth \
    REID_CHECKPOINT=/opt/models/reid/osnet_x0_25_msmt17.pt

# Catch dependency/import regressions and missing assets during docker build.
# These assertions exist so a future dependency resolution that displaces the
# working RTMO stack fails LOUDLY here instead of breaking at runtime.
COPY docker/verify_image.py /tmp/verify_image.py
RUN python3 /tmp/verify_image.py && rm -f /tmp/verify_image.py

WORKDIR /ros2_ws

# Entrypoint sources ROS 2 Humble and /ros2_ws/install before running the
# container command, so both this CMD and a Compose `command:` override start
# with the ROS environment already set up.
COPY docker/skeleton_detection_entrypoint.sh /skeleton_detection_entrypoint.sh
RUN chmod +x /skeleton_detection_entrypoint.sh

ENTRYPOINT ["/skeleton_detection_entrypoint.sh"]
CMD ["ros2", "launch", "skeleton_detection", "skeleton_detection_bringup.launch.py", "enable_tracking:=true", "with_reid:=true", "occlusion_aware_tracking:=true", "publish_visualization_image:=true", "draw_person_xyz:=true"]