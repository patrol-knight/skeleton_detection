"""Per-person depth and 3D camera-frame position from an aligned depth image.

Pure numpy: this module knows nothing about ROS, about pyrealsense2 or about
:class:`~skeleton_detection.inference.rtmo_inference.PersonDetection`.  It
takes a depth
image, one person's COCO keypoints and bounding box, and the colour camera
intrinsics as plain numbers, which makes it trivially testable on synthetic
arrays (see ``test/test_person_depth.py`` and ``test/test_person_position.py``).

Input contract
--------------
``depth_image`` must be a ``(H, W)`` array in the **same pixel coordinate
system as the keypoints**.  The keypoints come from RTMO, which runs on the
COLOUR frame, so the depth image must be ALIGNED TO COLOUR before it gets
here.  That alignment is done once, in
:mod:`skeleton_detection.input.realsense_capture` (``rs.align(rs.stream.color)``
in
the capture loop) -- never here, and never by indexing a raw depth frame with
colour coordinates.

For the same reason the intrinsics passed in must be the COLOUR stream's
intrinsics (:class:`CameraIntrinsics`), not the native depth sensor's: after
alignment both the pixel grid and the depth values live in the colour camera.

Units are decided by the caller through ``DepthParams.depth_scale``:

* raw RealSense Z16 (``uint16``)  -> ``depth_scale`` = the device's depth scale
  (meters per unit, ~0.001 on a D4xx), read from the depth sensor;
* already in meters (float array) -> ``depth_scale = 1.0`` (do NOT rescale).

Everything this module returns is in METERS.

Z-depth vs. Euclidean distance -- they are NOT the same
-------------------------------------------------------
::

    RealSense depth image value  = optical-axis Z coordinate
                                   (distance to the plane through the camera
                                   origin perpendicular to the optical axis)

    published PersonSkeleton.depth
                                 = Euclidean camera-to-person distance
                                 = sqrt(X^2 + Y^2 + Z^2)

The two agree only near the image centre; towards the image edge the Euclidean
distance is noticeably larger than Z.  In this module anything named ``z`` is
the optical-axis coordinate, and :attr:`CameraPoint.distance` is the Euclidean
distance.

Algorithm
---------
Person Z is a deliberately robust TWO-STAGE MEDIAN, not a mean::

    3x3 depth neighbourhood around each visible keypoint
            |  median of the valid values
            v
        joint Z

    all joint Z values of that person
            |  median
            v
        person Z

The median is used at both stages because a depth image around a person is
bimodal: a pixel is either on the person or on the background behind them.  A
mean would blend the two and place the person somewhere in empty space; a
median picks an actually-observed surface.

The person is then placed laterally at the centre ``(u, v)`` of its bounding
box (clipped to the image), and the pinhole model turns ``(u, v, Z)`` into a
point in the colour camera OPTICAL frame (x right, y down, z forward)::

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# A person with no usable depth is reported as NaN, never as 0.0: zero is a
# legal-looking distance and would silently be consumed as "right at the
# camera". NaN forces every consumer to handle "unknown" explicitly.
NO_DEPTH = float("nan")


@dataclass(frozen=True)
class DepthParams:
    """Configuration for :func:`compute_person_z`.

    Attributes:
        depth_scale: multiplied into every raw depth sample to obtain meters.
            ``1.0`` means the image is already in meters.
        keypoint_score_threshold: a keypoint is used only when its RTMO
            per-joint confidence is at or above this value.
        window: side length of the odd square neighbourhood sampled around
            each keypoint. ``3`` is the specified 3x3 patch.
        min_depth_m / max_depth_m: optional sanity range in meters, applied to
            the Z samples. ``<= 0`` disables that bound. Zero, NaN and
            infinite samples are ALWAYS rejected regardless of these.
    """

    depth_scale: float = 1.0
    keypoint_score_threshold: float = 0.3
    window: int = 3
    min_depth_m: float = 0.0
    max_depth_m: float = 0.0


DEFAULT_DEPTH_PARAMS = DepthParams()


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of the COLOUR stream, in colour-image pixels.

    Plain numbers so this module never imports pyrealsense2; build it from
    ``rs.intrinsics`` with ``cx = ppx`` and ``cy = ppy``.  ``width``/``height``
    are the colour image size the intrinsics were calibrated for.

    Lens distortion is ignored (pure pinhole). The D4xx colour stream is
    delivered with its distortion model and coefficients logged at startup by
    the node, so this assumption can be checked on the actual device.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def is_valid(self) -> bool:
        return (
            all(math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy))
            and self.fx > 0.0
            and self.fy > 0.0
            and self.width > 0
            and self.height > 0
        )


@dataclass(frozen=True)
class CameraPoint:
    """A 3D point in the colour camera OPTICAL frame, in meters.

    Axes (REP-103 optical convention): ``x`` toward image right, ``y`` toward
    image down, ``z`` forward along the optical axis. All three are NaN when
    the point could not be computed -- never zero.
    """

    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")

    def is_valid(self) -> bool:
        return all(math.isfinite(value) for value in (self.x, self.y, self.z))

    @property
    def distance(self) -> float:
        """Euclidean distance from the optical origin, ``sqrt(x^2+y^2+z^2)``.

        This is what ``PersonSkeleton.depth`` publishes. It is NOT the
        RealSense depth value (that is ``z``). NaN for an invalid point.
        """
        if not self.is_valid():
            return NO_DEPTH
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


# "No 3D position": NaN on every axis, and therefore a NaN distance.
NO_POSITION = CameraPoint()


def is_valid_depth(value: float) -> bool:
    """True when ``value`` is a usable depth (finite and strictly positive)."""
    return bool(np.isfinite(value)) and float(value) > 0.0


def format_depth(depth: float, decimals: int = 2) -> str:
    """Human-readable depth label; the single owner of the "N/A" convention."""
    if not is_valid_depth(depth):
        return "N/A"
    return f"{float(depth):.{decimals}f} m"


def compute_joint_z(
    depth_image: np.ndarray,
    u: float,
    v: float,
    params: DepthParams = DEFAULT_DEPTH_PARAMS,
) -> float:
    """Median optical-axis Z (meters) of the ``window x window`` patch at (u, v).

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

    # STAGE 1 of the two-stage median: one 3x3 neighbourhood -> one joint Z.
    return float(np.median(samples))


def compute_person_z(
    depth_image: Optional[np.ndarray],
    keypoints_xy: Optional[Sequence],
    keypoint_scores: Optional[Sequence],
    params: DepthParams = DEFAULT_DEPTH_PARAMS,
) -> float:
    """Estimate one person's optical-axis Z in meters, or :data:`NO_DEPTH`.

    This is the camera-frame Z coordinate, NOT the published Euclidean
    ``depth``; see :func:`compute_person_position` for that.

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
        The median of the valid per-joint Z values, or :data:`NO_DEPTH` when
        the person has no visible keypoint with a usable local depth.
    """
    if depth_image is None or keypoints_xy is None or keypoint_scores is None:
        return NO_DEPTH

    image = np.asarray(depth_image)
    if image.ndim != 2 or image.size == 0:
        return NO_DEPTH

    points = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(keypoint_scores, dtype=np.float64).ravel()
    count = min(points.shape[0], scores.shape[0])

    joint_zs = []
    for index in range(count):
        # Same visibility rule as the tracker/visualisation: an invisible or
        # hallucinated joint must not vote on where the person is.
        if not np.isfinite(scores[index]):
            continue
        if scores[index] < float(params.keypoint_score_threshold):
            continue

        joint_z = compute_joint_z(image, points[index, 0], points[index, 1], params)
        # Keypoints with no valid local depth are DROPPED here, so they cannot
        # bias the person-level median.
        if is_valid_depth(joint_z):
            joint_zs.append(joint_z)

    if not joint_zs:
        return NO_DEPTH

    # STAGE 2 of the two-stage median: all surviving joint Z values -> person Z.
    return float(np.median(joint_zs))


def bbox_center_pixel(
    bbox_xyxy: Optional[Sequence], width: int, height: int
) -> Optional[Tuple[float, float]]:
    """Representative colour pixel ``(u, v)`` of a person: its bbox centre.

    RTMO boxes are NOT clipped and may extend past the frame, so the box is
    first clipped to ``[0, width] x [0, height]`` and the centre of the
    VISIBLE part is used -- the same part the keypoints sampled Z from.
    Returns ``None`` for a non-finite box or one entirely outside the image.
    """
    if bbox_xyxy is None:
        return None
    box = np.asarray(bbox_xyxy, dtype=np.float64).ravel()
    if box.size != 4 or not np.all(np.isfinite(box)):
        return None

    x1 = min(max(box[0], 0.0), float(width))
    x2 = min(max(box[2], 0.0), float(width))
    y1 = min(max(box[1], 0.0), float(height))
    y2 = min(max(box[3], 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def deproject_pixel_to_point(
    u: float, v: float, z: float, intrinsics: Optional[CameraIntrinsics]
) -> CameraPoint:
    """Pinhole deprojection of colour pixel ``(u, v)`` at optical-axis ``z``.

    ``X = (u - cx) * Z / fx``, ``Y = (v - cy) * Z / fy``, ``Z = z``, all in
    meters, in the colour camera optical frame. Returns :data:`NO_POSITION`
    when ``z`` is not a usable depth or the intrinsics are missing/invalid.
    """
    if intrinsics is None or not intrinsics.is_valid():
        return NO_POSITION
    if not (is_valid_depth(z) and math.isfinite(u) and math.isfinite(v)):
        return NO_POSITION

    z = float(z)
    x = (float(u) - intrinsics.cx) * z / intrinsics.fx
    y = (float(v) - intrinsics.cy) * z / intrinsics.fy
    return CameraPoint(x=x, y=y, z=z)


def compute_person_position(
    depth_image: Optional[np.ndarray],
    keypoints_xy: Optional[Sequence],
    keypoint_scores: Optional[Sequence],
    bbox_xyxy: Optional[Sequence],
    intrinsics: Optional[CameraIntrinsics],
    params: DepthParams = DEFAULT_DEPTH_PARAMS,
) -> CameraPoint:
    """One person's 3D position in the colour camera optical frame [m].

    ``person Z`` (two-stage median over keypoints) + ``(u, v)`` (bbox centre)
    + colour intrinsics -> ``(X, Y, Z)``. The Euclidean distance published as
    ``PersonSkeleton.depth`` is the returned point's
    :attr:`CameraPoint.distance`.

    Returns :data:`NO_POSITION` (all NaN) whenever any ingredient is missing:
    no depth image, no intrinsics, no valid person Z, or no usable bbox.

    Raises:
        ValueError: ``depth_image`` does not have the intrinsics' ``(height,
            width)``, i.e. it is not aligned to the colour stream these
            intrinsics describe. That is a wiring bug, not a missing
            measurement, so it is not hidden behind NaN.
    """
    if depth_image is None or intrinsics is None or not intrinsics.is_valid():
        return NO_POSITION

    image = np.asarray(depth_image)
    if image.ndim == 2 and image.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth image is {image.shape[1]}x{image.shape[0]} but the colour "
            f"intrinsics are for {intrinsics.width}x{intrinsics.height}; the "
            "depth image must be aligned to the colour stream"
        )

    person_z = compute_person_z(image, keypoints_xy, keypoint_scores, params)
    if not is_valid_depth(person_z):
        return NO_POSITION

    pixel = bbox_center_pixel(bbox_xyxy, intrinsics.width, intrinsics.height)
    if pixel is None:
        return NO_POSITION

    u, v = pixel
    return deproject_pixel_to_point(u, v, person_z, intrinsics)
