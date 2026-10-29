"""Download and sha256-verify the model weights into a models directory.

    python3 docker/fetch_models.py --models-dir ~/models   # default /opt/models

The RTMO config is tracked in models/rtmo/ and copied alongside.
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_ROOT = RTMO_DIR = REID_DIR = None

# Absent in Docker, where the Dockerfile COPYs the config instead.
REPO_RTMO_DIR = Path(__file__).resolve().parents[1] / "models" / "rtmo"

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
            "Refusing to keep an unverified model file."
        )
    print(f"  sha256 OK ({actual[:16]}...)")


def is_verified(path: Path, expected: str) -> bool:
    if path.is_file() and sha256_of(path) == expected:
        print(f"  {path} already present, sha256 OK; skipping download")
        return True
    return False


def copy_rtmo_config() -> None:
    if not REPO_RTMO_DIR.is_dir() or REPO_RTMO_DIR == RTMO_DIR.resolve():
        return
    print(f"[rtmo] copying config {REPO_RTMO_DIR} -> {RTMO_DIR}")
    for path in REPO_RTMO_DIR.glob("*.py"):
        shutil.copy2(path, RTMO_DIR / path.name)


def fetch_rtmo_checkpoint() -> None:
    dest = RTMO_DIR / "rtmo-m.pth"
    if is_verified(dest, RTMO_CHECKPOINT_SHA256):
        return
    print(f"[rtmo] downloading checkpoint -> {dest}")
    with urllib.request.urlopen(RTMO_CHECKPOINT_URL, timeout=120) as response:
        with dest.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")
    verify(dest, RTMO_CHECKPOINT_SHA256, "RTMO checkpoint")


def check_rtmo_config() -> None:
    """Fail early if the RTMO config is missing or its _base_ chain is broken."""
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
    if is_verified(dest, REID_SHA256):
        return
    url = TRAINED_URLS[REID_MODEL_NAME]
    print(f"[reid] downloading {REID_MODEL_NAME} -> {dest}")
    print(f"  official BoxMOT source: {url}")
    download_file(url, dest)
    print(f"  {dest.stat().st_size / 1e6:.2f} MB")
    verify(dest, REID_SHA256, "ReID checkpoint")


def main() -> None:
    global MODELS_ROOT, RTMO_DIR, REID_DIR

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("SKELETON_MODELS_DIR", "/opt/models")),
        help="destination directory (default: $SKELETON_MODELS_DIR, else /opt/models)",
    )
    MODELS_ROOT = parser.parse_args().models_dir.expanduser().resolve()
    RTMO_DIR = MODELS_ROOT / "rtmo"
    REID_DIR = MODELS_ROOT / "reid"

    RTMO_DIR.mkdir(parents=True, exist_ok=True)
    REID_DIR.mkdir(parents=True, exist_ok=True)

    copy_rtmo_config()
    check_rtmo_config()
    fetch_rtmo_checkpoint()
    fetch_reid_checkpoint()

    print(f"\nModel assets in {MODELS_ROOT}:")
    for path in sorted(MODELS_ROOT.rglob("*")):
        if path.is_file():
            print(f"  {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
