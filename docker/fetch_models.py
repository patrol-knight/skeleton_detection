"""Bake runtime model assets into the Docker image under /opt/models.

Why this exists
---------------
The model files used to live in ``data/checkpoints/`` inside the source tree,
which is both ``.gitignore``d and ``.dockerignore``d.  Anyone who did
``git clone`` + ``docker pull`` therefore got a repository and an image with no
weights, and the node failed at start-up.  Baking the assets into the image at
a stable path makes a pulled image self-contained.

This script downloads the WEIGHTS only.  The RTMO config is version-controlled
in the repository under ``models/rtmo/`` and copied into the image by the
Dockerfile ``COPY`` immediately above the line that runs this script -- so the
config the image runs is the config a reviewer can read in git.  This script
then asserts that copy is present and loadable.

Layout produced::

    /opt/models/
    ├── rtmo/
    │   ├── rtmo-m.py           MMPose config      (COPY'd from models/rtmo/)
    │   ├── default_runtime.py  its _base_         (COPY'd from models/rtmo/)
    │   └── rtmo-m.pth          official RTMO-M Body7 checkpoint (downloaded)
    └── reid/
        └── osnet_x0_25_msmt17.pt   ReID weights   (downloaded)

Sources are official only:
  * RTMO checkpoint -> download.openmmlab.com (the URL published in the MMPose
    RTMO project), verified against the sha256 that MMPose encodes in the
    filename.
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


def check_rtmo_config() -> None:
    """Assert the repo-vendored RTMO config was COPY'd in and actually loads.

    The config is NOT generated here: the Dockerfile copies models/rtmo/ from
    the repository.  This function is the build-time gate that catches a
    missing COPY, a renamed file, or a broken _base_ chain -- and it catches it
    during `docker build` rather than at node start-up on the robot.
    """
    from mmengine import Config

    config = RTMO_DIR / "rtmo-m.py"
    base = RTMO_DIR / "default_runtime.py"
    for path, why in (
        (config, "COPY models/rtmo/ /opt/models/rtmo/ in the Dockerfile"),
        (base, "rtmo-m.py declares it as its _base_"),
    ):
        if not path.is_file():
            raise SystemExit(
                f"FATAL: expected {path} to exist -- {why}.\n"
                "The RTMO config is version-controlled in models/rtmo/; it is "
                "not downloaded or generated."
            )

    print(f"[rtmo] validating vendored config {config}")
    # Loading resolves the _base_ chain, so a broken sibling path fails here.
    loaded = Config.fromfile(str(config))
    missing = [
        key
        for key in ("model", "test_dataloader", "default_scope")
        if key not in loaded
    ]
    if missing:
        raise SystemExit(
            f"FATAL: {config} is missing required section(s): {missing}. "
            "'default_scope' comes from default_runtime.py; without it "
            "init_model cannot resolve the mmpose registry."
        )
    print(f"  {config.stat().st_size / 1e3:.1f} kB, _base_ resolves OK")


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
    # RTMO_DIR normally already exists (the Dockerfile COPY creates it);
    # mkdir keeps the script runnable standalone for debugging.
    RTMO_DIR.mkdir(parents=True, exist_ok=True)
    REID_DIR.mkdir(parents=True, exist_ok=True)

    check_rtmo_config()
    fetch_rtmo_checkpoint()
    fetch_reid_checkpoint()

    print("\nBaked model assets:")
    for path in sorted(MODELS_ROOT.rglob("*")):
        if path.is_file():
            print(f"  {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
