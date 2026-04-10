# OpenPifPaf Setup Guide
This guide provides a step-by-step walkthrough for setting up OpenPifPaf on Ubuntu 22.04

## Prerequisites
```
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev ffmpeg libsm6 libxext6
```

## Environment Setup
```
cd patrolknight
python3 -m venv pifpaf_env
source pifpaf_env/bin/activate
```

## Installation
OpenPifPaf requires specific versions of its dependencies to run without "Undefined Symbol" or "NumPy Mismatch" errors.
```
pip install --upgrade pip setuptools wheel

pip install "setuptools<82.0.0"

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

pip install "numpy<2.0"

pip install openpifpaf --no-build-isolation
```
In run_f1tenth.sh, you can choose the environment map and the modules you want to run.

## User Commands
```
python3 -m openpifpaf.predict path/to/image.jpg --image-output
```