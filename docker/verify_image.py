"""Build-time gate: fail the image build if the verified stack is displaced.

Every check here corresponds to something that has actually broken (or was at
risk of breaking) during this project's dependency work:

  * numpy 2.x displacing 1.23.5 breaks chumpy, opencv-python 4.8 and the
    compiled mmpose/xtcocotools wheels. BoxMOT >= 20 requires numpy >= 2.2,
    which is exactly why boxmot is pinned to 19.0.0.
  * a torch/torchvision swap would invalidate the CUDA 13 / GB10 build.
  * mmcv-lite + the mmcv._ext stub must still satisfy the mmpose import chain.
  * the baked model assets must exist, or a pulled image is not self-contained.
"""

import sys
from pathlib import Path

EXPECTED = {
    "numpy": "1.23",
    "torch": "2.14.0+cu130",
    "torchvision": "0.29.0+cu130",
    "boxmot_pypi_version": "19.0.0",
}

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print("== core numeric/ML stack ==")
import numpy

check(f"numpy {numpy.__version__}", numpy.__version__.startswith(EXPECTED["numpy"]),
      f"expected {EXPECTED['numpy']}.x")

import torch
import torchvision

check(f"torch {torch.__version__}", torch.__version__ == EXPECTED["torch"],
      f"expected {EXPECTED['torch']}")
check(f"torchvision {torchvision.__version__}",
      torchvision.__version__ == EXPECTED["torchvision"],
      f"expected {EXPECTED['torchvision']}")

import cv2

check(f"opencv {cv2.__version__}", cv2.__version__.startswith("4.8"))

print("== OpenMMLab / RTMO import chain ==")
import mmcv
import mmdet
import mmengine
import mmpose

check(f"mmengine {mmengine.__version__}", mmengine.__version__ == "0.10.7")
check(f"mmcv {mmcv.__version__}", mmcv.__version__ == "2.1.0")
check(f"mmdet {mmdet.__version__}", mmdet.__version__ == "3.3.0")
check(f"mmpose {mmpose.__version__}", mmpose.__version__ == "1.3.2")

try:
    from mmpose.apis import inference_bottomup, init_model  # noqa: F401
    from mmpose.models.heads.hybrid_heads.rtmo_head import RTMOHead  # noqa: F401

    check("RTMO import chain (mmcv._ext stub satisfies mmpose)", True)
except Exception as exc:  # noqa: BLE001
    check("RTMO import chain", False, str(exc))

print("== RealSense ==")
try:
    import pyrealsense2 as rs

    check(f"pyrealsense2 {rs.__version__}", rs.__version__.startswith("2.58"))
except Exception as exc:  # noqa: BLE001
    check("pyrealsense2 import", False, str(exc))

print("== BoxMOT (BoT-SORT + PyTorch ReID; not yet wired into the node) ==")
try:
    from importlib.metadata import version

    installed = version("boxmot")
    check(f"boxmot {installed}", installed == EXPECTED["boxmot_pypi_version"],
          f"expected {EXPECTED['boxmot_pypi_version']}")
except Exception as exc:  # noqa: BLE001
    check("boxmot version", False, str(exc))

try:
    from boxmot.trackers.botsort.botsort import BotSort  # noqa: F401

    check("BoT-SORT tracker import", True)
except Exception as exc:  # noqa: BLE001
    check("BoT-SORT tracker import", False, str(exc))

try:
    from boxmot.reid import ReID  # noqa: F401
    from boxmot.reid.backends.pytorch_backend import PyTorchBackend  # noqa: F401

    check("BoxMOT ReID (PyTorch backend) import", True)
except Exception as exc:  # noqa: BLE001
    check("BoxMOT ReID import", False, str(exc))

print("== baked model assets ==")
for label, path, min_mb in (
    ("RTMO config", "/opt/models/rtmo/rtmo-m.py", 0.01),
    # rtmo-m.py's _base_; the model cannot be built without it.
    ("RTMO config _base_", "/opt/models/rtmo/default_runtime.py", 0.0005),
    ("RTMO checkpoint", "/opt/models/rtmo/rtmo-m.pth", 50.0),
    ("ReID checkpoint", "/opt/models/reid/osnet_x0_25_msmt17.pt", 1.0),
):
    file_path = Path(path)
    exists = file_path.is_file()
    size_mb = file_path.stat().st_size / 1e6 if exists else 0.0
    check(f"{label} {path}", exists and size_mb >= min_mb, f"{size_mb:.2f} MB")

if failures:
    print(f"\nIMAGE VERIFICATION FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("\nIMAGE VERIFICATION PASSED")
