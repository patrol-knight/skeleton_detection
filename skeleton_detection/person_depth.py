"""Per-person depth estimation from an aligned depth image.

Pure numpy: this module knows nothing about ROS, about RealSense or about
:class:`~skeleton_detection.rtmo_inference.PersonDetection`.  It takes a depth
image plus one person's COCO keypoints and returns a single scalar depth in
meters, which makes it trivially testable on synthetic arrays (see
``test/test_person_depth.py``).

Input contract
--------------
``depth_image`` must be a ``(H, W)`` array in the **same pixel coordinate
system as the keypoints**.  The keypoints come from RTMO, which runs on the
COLOUR frame, so the depth image must be ALIGNED TO COLOUR before it gets
here.  That alignment is done once, in
:mod:`skeleton_detection.realsense_capture` (``rs.align(rs.stream.color)`` in
the capture loop) -- never here, and never by indexing a raw depth frame with
colour coordinates.

Units are decided by the caller through ``DepthParams.depth_scale``:

* raw RealSense Z16 (``uint16``)  -> ``depth_scale`` = the device's depth scale
  (meters per unit, ~0.001 on a D4xx), read from the depth sensor;
* already in meters (float array) -> ``depth_scale = 1.0`` (do NOT rescale).

Everything this module returns is in METERS.

Algorithm (deliberately a TWO-STAGE MEDIAN, not a mean)
-------------------------------------------------------
::

    3x3 depth neighbourhood around each visible keypoint
            |  median of the valid values
            v
        joint_depth

    all joint_depth values of that person
            |  median
            v
        person depth

The median is used at both stages because a depth image around a person is
bimodal: a pixel is either on the person or on the background behind them.  A
mean would blend the two and place the person somewhere in empty space; a
median picks an actually-observed surface.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# A person with no usable depth is reported as NaN, never as 0.0: zero is a
# legal-looking distance and would silently be consumed as "right at the
# camera". NaN forces every consumer to handle "unknown" explicitly.
NO_DEPTH = float("nan")


@dataclass(frozen=True)
class DepthParams:
    """Configuration for :func:`compute_person_depth`.

    Attributes:
        depth_scale: multiplied into every raw depth sample to obtain meters.
            ``1.0`` means the image is already in meters.
        keypoint_score_threshold: a keypoint is used only when its RTMO
            per-joint confidence is at or above this value.
        window: side length of the odd square neighbourhood sampled around
            each keypoint. ``3`` is the specified 3x3 patch.
        min_depth_m / max_depth_m: optional sanity range in meters. ``<= 0``
            disables that bound. Zero, NaN and infinite samples are ALWAYS
            rejected regardless of these.
    """

    depth_scale: float = 1.0
    keypoint_score_threshold: float = 0.3
    window: int = 3
    min_depth_m: float = 0.0
    max_depth_m: float = 0.0


DEFAULT_DEPTH_PARAMS = DepthParams()


def is_valid_depth(value: float) -> bool:
    """True when ``value`` is a usable depth (finite and strictly positive)."""
    return bool(np.isfinite(value)) and float(value) > 0.0


def format_depth(depth: float, decimals: int = 2) -> str:
    """Human-readable depth label; the single owner of the "N/A" convention."""
    if not is_valid_depth(depth):
        return "N/A"
    return f"{float(depth):.{decimals}f} m"


def compute_joint_depth(
    depth_image: np.ndarray,
    u: float,
    v: float,
    params: DepthParams = DEFAULT_DEPTH_PARAMS,
) -> float:
    """Median depth (meters) of the ``window x window`` patch around (u, v).

    ``u`` is the column and ``v`` the row, in depth-image pixels.  Returns
    :data:`NO_DEPTH` when the keypoint falls outside the image or when no
    sample in the patch is valid.  Patches are CLIPPED at the image border, so
    a keypoint on the edge simply contributes fewer samples rather than
    wrapping around or raising.
    """
    height, width = depth_image.shape[:2]

    if not (np.isfinite(u) and np.isfinite(v)):
        return NO_DEPTH

    column = int(round(float(u)))
    row = int(round(float(v)))
    if not (0 <= column < width and 0 <= row < height):
        return NO_DEPTH

    half = max(0, int(params.window) // 2)
    row_start, row_end = max(0, row - half), min(height, row + half + 1)
    column_start, column_end = max(0, column - half), min(width, column + half + 1)

    patch = np.asarray(
        depth_image[row_start:row_end, column_start:column_end], dtype=np.float64
    )
    # Scale to meters BEFORE filtering so the range bounds are always in
    # meters, whatever units the incoming image used.
    patch = patch.ravel() * float(params.depth_scale)

    # Reject the three universally invalid cases: zero (the RealSense "no
    # measurement" code), NaN and +/-inf.
    valid = np.isfinite(patch) & (patch > 0.0)
    if params.min_depth_m > 0.0:
        valid &= patch >= float(params.min_depth_m)
    if params.max_depth_m > 0.0:
        valid &= patch <= float(params.max_depth_m)

    samples = patch[valid]
    if samples.size == 0:
        return NO_DEPTH

    # STAGE 1 of the two-stage median: one 3x3 neighbourhood -> one joint depth.
    return float(np.median(samples))


def compute_person_depth(
    depth_image: Optional[np.ndarray],
    keypoints_xy: Optional[Sequence],
    keypoint_scores: Optional[Sequence],
    params: DepthParams = DEFAULT_DEPTH_PARAMS,
) -> float:
    """Estimate one person's depth in meters, or :data:`NO_DEPTH`.

    Args:
        depth_image: ``(H, W)`` depth image ALIGNED TO THE COLOUR FRAME, in
            raw units (with ``params.depth_scale``) or in meters (scale 1.0).
            ``None`` -- no depth stream -- yields :data:`NO_DEPTH`.
        keypoints_xy: ``(K, 2)`` keypoint pixel coordinates ``[u, v]``, in the
            same coordinate system as ``depth_image``.
        keypoint_scores: ``(K,)`` per-keypoint confidence; joints below
            ``params.keypoint_score_threshold`` are ignored, which is the same
            visibility rule the rest of the package uses.

    Returns:
        The median of the valid per-joint depths, or :data:`NO_DEPTH` when the
        person has no visible keypoint with a usable local depth.
    """
    if depth_image is None or keypoints_xy is None or keypoint_scores is None:
        return NO_DEPTH

    image = np.asarray(depth_image)
    if image.ndim != 2 or image.size == 0:
        return NO_DEPTH

    points = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(keypoint_scores, dtype=np.float64).ravel()
    count = min(points.shape[0], scores.shape[0])

    joint_depths = []
    for index in range(count):
        # Same visibility rule as the tracker/visualisation: an invisible or
        # hallucinated joint must not vote on where the person is.
        if not np.isfinite(scores[index]):
            continue
        if scores[index] < float(params.keypoint_score_threshold):
            continue

        joint_depth = compute_joint_depth(
            image, points[index, 0], points[index, 1], params
        )
        # Keypoints with no valid local depth are DROPPED here, so they cannot
        # bias the person-level median.
        if is_valid_depth(joint_depth):
            joint_depths.append(joint_depth)

    if not joint_depths:
        return NO_DEPTH

    # STAGE 2 of the two-stage median: all surviving joint depths -> person depth.
    return float(np.median(joint_depths))
