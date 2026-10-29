"""Default model asset paths.

Every asset lives under one models directory: ``/opt/models`` in the Docker
image (see docker/fetch_models.py), anything else natively via
``SKELETON_MODELS_DIR``. The per-file variables still override one asset.
"""

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
