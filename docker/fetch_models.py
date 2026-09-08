"""Bake runtime model assets into the Docker image under /opt/models.

Why this exists
---------------
The model files used to live in ``data/checkpoints/`` inside the source tree,
which is both ``.gitignore``d and ``.dockerignore``d.  Anyone who did
``git clone`` + ``docker pull`` therefore got a repository and an image with no
weights, and the node failed at start-up.  Baking the assets into the image at
a stable path makes a pulled image self-contained.

Layout produced::

    /opt/models/
    ├── rtmo/
    │   ├── rtmo-m.py     standalone (base-resolved) MMPose config
    │   └── rtmo-m.pth    official RTMO-M Body7 checkpoint
    └── reid/
        └── osnet_x0_25_msmt17.pt   ReID weights for the future BoT-SORT work

Sources are official only:
  * RTMO checkpoint -> download.openmmlab.com (the URL published in the MMPose
    RTMO project), verified against the sha256 that MMPose encodes in the
    filename.
  * RTMO config     -> the config shipped inside the installed ``mmpose`` wheel
    (``mmpose/.mim/configs/...``).  It is re-emitted through mmengine's own
    ``Config.dump()`` because the original declares a RELATIVE
    ``_base_ = ['../../../_base_/default_runtime.py']`` and would not load from
    a flat directory.  The dump inlines the merged result, so the copy is
    self-contained; it was verified to produce bit-identical inference output.
  * ReID weights    -> BoxMOT's own registry URL, fetched with BoxMOT's own
    downloader, then checksum-verified.  No third-party mirror is used.

Every download is verified by sha256, so a corrupted or substituted file fails
the image build instead of silently shipping.
"""

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_ROOT = Path("/opt/models")
RTMO_DIR = MODELS_ROOT / "rtmo"
REID_DIR = MODELS_ROOT / "reid"

# Official MMPose RTMO-M (Body7, 640x640). The sha256 prefix is part of the
# published filename, which is how MMPose distributes the digest.
RTMO_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/"
    "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth"
)
RTMO_CHECKPOINT_SHA256 = (
    "39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3"
)
RTMO_SOURCE_CONFIG = (
    "/usr/local/lib/python3.10/dist-packages/mmpose/.mim/configs/"
    "body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py"
)

REID_MODEL_NAME = "osnet_x0_25_msmt17.pt"
REID_SHA256 = "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str, label: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"FATAL: {label} sha256 mismatch\n"
            f"  expected {expected}\n  got      {actual}\n"
            "Refusing to bake an unverified model file into the image."
        )
    print(f"  sha256 OK ({actual[:16]}...)")


def fetch_rtmo_checkpoint() -> None:
    dest = RTMO_DIR / "rtmo-m.pth"
    print(f"[rtmo] downloading checkpoint -> {dest}")
    with urllib.request.urlopen(RTMO_CHECKPOINT_URL, timeout=120) as response:
        with dest.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")
    verify(dest, RTMO_CHECKPOINT_SHA256, "RTMO checkpoint")


def emit_rtmo_config() -> None:
    """Write a standalone, base-resolved copy of the official RTMO-M config."""
    from mmengine import Config

    dest = RTMO_DIR / "rtmo-m.py"
    source = Path(RTMO_SOURCE_CONFIG)
    if not source.is_file():
        raise SystemExit(f"FATAL: MMPose RTMO config not found at {source}")
    print(f"[rtmo] resolving config {source.name} -> {dest}")
    Config.fromfile(str(source)).dump(str(dest))
    # Reload the emitted file on its own to prove it no longer depends on the
    # relative _base_ path it was written from.
    reloaded = Config.fromfile(str(dest))
    assert "model" in reloaded and "test_dataloader" in reloaded, (
        "resolved RTMO config is missing required sections"
    )
    print(f"  {dest.stat().st_size / 1e3:.1f} kB, reloads standalone OK")


def fetch_reid_checkpoint() -> None:
    """Use BoxMOT's own registry URL and downloader (no third-party mirror)."""
    from boxmot.reid.core.registry import TRAINED_URLS
    from boxmot.utils.download import download_file

    dest = REID_DIR / REID_MODEL_NAME
    url = TRAINED_URLS[REID_MODEL_NAME]
    print(f"[reid] downloading {REID_MODEL_NAME} -> {dest}")
    print(f"  official BoxMOT source: {url}")
    download_file(url, dest)
    print(f"  {dest.stat().st_size / 1e6:.2f} MB")
    verify(dest, REID_SHA256, "ReID checkpoint")


def main() -> None:
    RTMO_DIR.mkdir(parents=True, exist_ok=True)
    REID_DIR.mkdir(parents=True, exist_ok=True)

    fetch_rtmo_checkpoint()
    emit_rtmo_config()
    fetch_reid_checkpoint()

    print("\nBaked model assets:")
    for path in sorted(MODELS_ROOT.rglob("*")):
        if path.is_file():
            print(f"  {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
