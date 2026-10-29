"""Model asset paths: ``$SKELETON_MODELS_DIR`` (default /opt/models), per-file env overrides."""

import os

MODELS_DIR = os.environ.get("SKELETON_MODELS_DIR", "/opt/models")

DEFAULT_MODEL_CONFIG = os.environ.get(
    "RTMO_MODEL_CONFIG", os.path.join(MODELS_DIR, "rtmo", "rtmo-m.py")
)
DEFAULT_CHECKPOINT = os.environ.get(
    "RTMO_CHECKPOINT", os.path.join(MODELS_DIR, "rtmo", "rtmo-m.pth")
)
DEFAULT_REID_CHECKPOINT = os.environ.get(
    "REID_CHECKPOINT", os.path.join(MODELS_DIR, "reid", "osnet_x0_25_msmt17.pt")
)
