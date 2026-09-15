"""Shared skeleton drawing used by both the ROS debug topic and saved files.

This is the single drawing implementation in the package.  ``rtmo_node`` calls
:func:`draw_skeleton_overlay` once per frame and then (a) publishes the result
on ``/skeleton_detection/visualization_image`` (downscaled) and/or (b) writes
it to disk at full resolution, so the file you open in VS Code and the live
topic show exactly the same annotations.

The functions here take duck-typed "person" objects: anything exposing
``person_id`` (int), ``score`` (float), ``bbox`` ([x, y, width, height]) and
``joints`` (51 floats, [x, y, conf] * 17) works.  ``PersonSkeleton`` messages
satisfy that, so no ROS message import is needed here.  ``depth`` (float,
meters, NaN = unavailable) is read when present and rendered as
``Depth: N/A`` when it is missing or NaN.

``person_id`` is drawn as-is.  When tracking is enabled it is a persistent
BoT-SORT track id; when tracking is off it is the frame-local detection index.
The caller selects the matching footer text with :func:`legend_for`.

All coordinates are in the ORIGINAL source image coordinate system; the canvas
is the untouched source image, so nothing has to be rescaled.
"""

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .coco_keypoints import (
    COCO_CONNECTIONS,
    COCO_KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
)
from .person_depth import NO_DEPTH, format_depth


# BGR colours.
COLOR_BBOX = (0, 255, 0)          # green
COLOR_LEFT = (255, 160, 0)        # blue-ish   -> left side of the body
COLOR_RIGHT = (0, 160, 255)       # orange     -> right side of the body
COLOR_CENTER = (0, 255, 255)      # yellow     -> midline / cross-body links
COLOR_TEXT = (255, 255, 255)      # white on a dark plate
COLOR_TEXT_PLATE = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Rendered onto every saved/published overlay so a reviewer looking at the file
# alone knows what the ID actually means. Use :func:`legend_for` to pick the
# right one -- the meaning of person_id depends on whether tracking is enabled.
UNTRACKED_LEGEND = "ID = per-frame detection index (NOT stable across frames)"
TRACKED_LEGEND = "ID = persistent BoT-SORT track ID"
TRACKED_REID_LEGEND = "ID = persistent BoT-SORT track ID (motion + ReID)"
DEFAULT_LEGEND = UNTRACKED_LEGEND


def legend_for(tracking_enabled: bool, with_reid: bool = False) -> str:
    """Legend text matching the id semantics actually in force."""
    if not tracking_enabled:
        return UNTRACKED_LEGEND
    return TRACKED_REID_LEGEND if with_reid else TRACKED_LEGEND


def _side_color(keypoint_index: int) -> Tuple[int, int, int]:
    name = COCO_KEYPOINT_NAMES[keypoint_index]
    if name.startswith("left_"):
        return COLOR_LEFT
    if name.startswith("right_"):
        return COLOR_RIGHT
    return COLOR_CENTER


def _connection_color(index_a: int, index_b: int) -> Tuple[int, int, int]:
    color_a = _side_color(index_a)
    color_b = _side_color(index_b)
    # Cross-body links (shoulders, hips, eyes...) get the neutral colour.
    return color_a if color_a == color_b else COLOR_CENTER


def _scaled_metrics(image_height: int) -> Tuple[int, int, float]:
    """Line thickness, joint radius and font scale for this image size."""
    thickness = max(1, int(round(image_height / 300.0)))
    radius = max(2, int(round(image_height / 180.0)))
    font_scale = max(0.4, min(1.0, image_height / 700.0))
    return thickness, radius, font_scale


def _text_extent(text: str, font_scale: float, thickness: int) -> Tuple[int, int, int]:
    (text_width, text_height), baseline = cv2.getTextSize(
        text, FONT, font_scale, thickness
    )
    return text_width, text_height, baseline


def _draw_text_with_plate(
    canvas: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float,
    thickness: int,
    text_color: Tuple[int, int, int] = COLOR_TEXT,
    top_limit: int = 0,
) -> int:
    """Draw text on a filled dark plate; returns the plate's bottom y.

    ``origin`` is the baseline-ish bottom-left corner of the text.  The plate
    is kept inside the image and never pushed above ``top_limit``, which is how
    the per-person labels avoid overlapping the title banner.
    """
    text_width, text_height, baseline = _text_extent(text, font_scale, thickness)
    x, y = origin
    x = max(0, min(x, canvas.shape[1] - text_width - 4))
    y = max(top_limit + text_height + baseline + 4, min(y, canvas.shape[0] - 4))
    plate_top = y - text_height - baseline - 2
    plate_bottom = y + baseline - 1
    cv2.rectangle(
        canvas,
        (x - 2, plate_top),
        (x + text_width + 2, plate_bottom),
        COLOR_TEXT_PLATE,
        cv2.FILLED,
    )
    cv2.putText(
        canvas, text, (x, y - baseline + 1), FONT, font_scale, text_color,
        thickness, cv2.LINE_AA
    )
    return plate_bottom


def draw_skeleton_overlay(
    image_bgr: np.ndarray,
    persons: Sequence,
    joint_score_threshold: float = 0.3,
    draw_joint_scores: bool = False,
    title: Optional[str] = None,
    legend: Optional[str] = DEFAULT_LEGEND,
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the detections drawn on top.

    Args:
        image_bgr: the ORIGINAL source frame (BGR, unmodified resolution).
        persons: objects with ``person_id``/``score``/``bbox``/``joints``,
            and optionally ``depth`` (meters; NaN or absent -> "N/A").
        joint_score_threshold: joints below this confidence are not drawn, and
            a connection is drawn only when BOTH endpoints are above it.
        draw_joint_scores: print each drawn joint's confidence next to it.
            Off by default: it is unreadable on small people.
        title: optional one-line banner drawn at the top left (frame info).
        legend: optional one-line footer; defaults to the "not a tracking ID"
            reminder.  Pass ``None`` to omit.
    """
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    thickness, radius, font_scale = _scaled_metrics(height)
    label_thickness = max(1, thickness - 1)

    # Title first: person labels then know which band to stay clear of.
    title_bottom = 0
    if title:
        title_bottom = _draw_text_with_plate(
            canvas, title, (6, 22), font_scale, label_thickness
        )

    for person in persons:
        x, y, box_width, box_height = (float(value) for value in person.bbox)
        top_left = (int(round(x)), int(round(y)))
        bottom_right = (int(round(x + box_width)), int(round(y + box_height)))
        cv2.rectangle(canvas, top_left, bottom_right, COLOR_BBOX, thickness)

        joints = list(person.joints)

        # Connections first so the joint dots stay on top of the limb lines.
        for index_a, index_b in COCO_CONNECTIONS:
            if (
                joints[index_a * 3 + 2] < joint_score_threshold
                or joints[index_b * 3 + 2] < joint_score_threshold
            ):
                continue
            point_a = (
                int(round(joints[index_a * 3])),
                int(round(joints[index_a * 3 + 1])),
            )
            point_b = (
                int(round(joints[index_b * 3])),
                int(round(joints[index_b * 3 + 1])),
            )
            cv2.line(
                canvas, point_a, point_b, _connection_color(index_a, index_b),
                thickness, cv2.LINE_AA
            )

        for joint_index in range(NUM_COCO_KEYPOINTS):
            joint_x, joint_y, confidence = joints[joint_index * 3: joint_index * 3 + 3]
            if confidence < joint_score_threshold:
                continue
            center = (int(round(joint_x)), int(round(joint_y)))
            cv2.circle(canvas, center, radius, _side_color(joint_index), cv2.FILLED)
            cv2.circle(canvas, center, radius, (0, 0, 0), 1, cv2.LINE_AA)
            if draw_joint_scores:
                cv2.putText(
                    canvas, f"{confidence:.2f}",
                    (center[0] + radius + 1, center[1] - radius),
                    FONT, font_scale * 0.5, COLOR_TEXT, 1, cv2.LINE_AA
                )

        # Person label above the box; if that would land in the title band,
        # drop it just inside the top edge of the box instead so it always
        # stays visually attached to the right person.
        # getattr keeps the duck-typed contract: a person object without a
        # depth field simply shows "N/A" instead of raising.
        depth = float(getattr(person, "depth", NO_DEPTH))
        label = (
            f"ID {person.person_id}  score={float(person.score):.2f}"
            f" | Depth: {format_depth(depth)}"
        )
        _, label_height, label_baseline = _text_extent(
            label, font_scale, label_thickness
        )
        label_y = top_left[1] - 6
        if label_y - label_height - label_baseline - 2 < title_bottom:
            label_y = top_left[1] + label_height + label_baseline + 8
        _draw_text_with_plate(
            canvas,
            label,
            (top_left[0], label_y),
            font_scale,
            label_thickness,
            top_limit=title_bottom,
        )

    if legend:
        footer = f"{legend} | joint conf >= {joint_score_threshold:.2f}"
        _draw_text_with_plate(canvas, footer, (6, height - 8), font_scale * 0.75, 1)

    return canvas


def joint_visibility_summary(
    persons: Sequence, joint_score_threshold: float = 0.3
) -> List[int]:
    """Per-person count of joints above the threshold (handy for logging)."""
    counts: List[int] = []
    for person in persons:
        confidences = list(person.joints)[2::3]
        counts.append(sum(1 for value in confidences if value >= joint_score_threshold))
    return counts


class VisualizationWriter:
    """Writes rendered overlays to disk as image files.

    Kept next to the drawing code because it is a visualization *sink*, not
    node logic. The live ROS topic is the other sink; both consume the overlay
    produced by :func:`draw_skeleton_overlay`, which stays the single drawing
    implementation.

    Naming convention: ``frame_<frame_index padded to 6>_rtmo.<ext>``, e.g.
    ``frame_000000_rtmo.jpg``. ``frame_index`` resets when the node restarts,
    so files never collide within a run, but a NEW run overwrites the previous
    run's files -- logged as a warning rather than done silently.
    """

    def __init__(self, output_dir: str, image_format: str = "jpg", logger=None) -> None:
        self.output_dir = output_dir
        self.image_format = str(image_format).lstrip(".").lower()
        if self.image_format not in ("jpg", "jpeg", "png"):
            raise ValueError(
                "visualization_image_format must be one of 'jpg', 'jpeg', 'png'; "
                f"got '{self.image_format}'"
            )
        self._logger = logger
        self._prepare_output_dir()

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            print(f"[{level}] {message}", flush=True)
            return
        try:
            getattr(self._logger, level)(message)
        except Exception:  # noqa: BLE001
            print(f"[{level}] {message}", flush=True)

    def _prepare_output_dir(self) -> None:
        """Create the output directory tree, widening permissions we create.

        The default lives inside the bind-mounted source tree, so files written
        here appear directly on the host and can be opened from the VS Code SSH
        file explorer -- no rqt, X11 or display forwarding involved.
        """
        created = []
        candidate = os.path.abspath(self.output_dir)
        while candidate and not os.path.isdir(candidate):
            created.append(candidate)
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create visualization_output_dir '{self.output_dir}': {exc}"
            ) from exc

        for directory in created:
            self._relax_permissions(directory, 0o777)

        if not os.access(self.output_dir, os.W_OK):
            raise RuntimeError(
                f"visualization_output_dir '{self.output_dir}' is not writable"
            )

    def write(self, overlay_bgr: np.ndarray, frame_index: int) -> Optional[str]:
        """Write one annotated frame; returns the path, or None on failure."""
        filename = f"frame_{frame_index:06d}_rtmo.{self.image_format}"
        path = os.path.join(self.output_dir, filename)

        if os.path.exists(path):
            self._log(
                "warning",
                f"Overwriting existing {path} (left over from an earlier run)",
            )

        try:
            written = cv2.imwrite(path, overlay_bgr)
        except cv2.error as exc:
            self._log("error", f"Failed to write {path}: {exc}")
            return None

        if not written:
            self._log("error", f"cv2.imwrite returned False for {path}")
            return None

        self._relax_permissions(path, 0o666)
        return path

    @staticmethod
    def _relax_permissions(path: str, mode: int) -> None:
        """Make container-written files manageable from the host.

        The container runs as root, so anything it writes into the bind mount
        is root-owned on the host. Widening the mode keeps the offline workflow
        friction-free. Best effort: skipped when not root or when the
        filesystem refuses chmod.
        """
        if os.geteuid() != 0:
            return
        try:
            os.chmod(path, mode)
        except OSError:
            pass
