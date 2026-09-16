"""Direct Intel RealSense capture, in-process, with a latest-frame-wins buffer.

Milestone 2 exists because pushing 848x480@60 ``sensor_msgs/Image`` frames
(~1.2 MB each, ~70 MB/s) between two processes over Fast DDS is a bottleneck
in this environment.  This module lets the RTMO node open the D456 itself with
the RealSense SDK, so an RGB frame goes

    librealsense -> numpy array -> RTMO

with no serialization, no DDS, and no intermediate ROS Image topic.

Threading model
---------------
A dedicated capture thread owns the ``rs.pipeline`` and does nothing but pull
frames and drop them into a single-slot buffer.  The consumer (the RTMO worker
thread) takes whatever is in that slot.  There is no queue and no backlog:

    LATEST FRAME WINS.

If the consumer is slower than the camera, the newest frame simply overwrites
the previous unconsumed one and the overwritten frame is counted as *dropped*.
Latency therefore stays bounded at roughly one camera period plus one
inference, instead of growing without limit.

Depth, and why it is ALIGNED TO COLOUR
--------------------------------------
When ``enable_depth`` is set (the default), a Z16 depth stream is opened
alongside the colour stream and every frameset is pushed through
``rs.align(rs.stream.color)`` **in the capture loop** before the arrays are
copied out.  This is the ONLY place in the package where depth-to-colour
alignment happens.

It has to happen, because the D4xx depth and colour sensors sit at different
positions on the module and have different intrinsics: the raw depth frame is
in the depth sensor's own pixel grid.  RTMO runs on the COLOUR image, so its
keypoints are colour pixels, and indexing an unaligned depth frame with them
would read the wrong part of the scene.  After ``rs.align`` the depth image has
the colour frame's size and intrinsics, and ``depth[v, u]`` is the distance at
colour pixel ``(u, v)`` -- which is exactly what
:mod:`skeleton_detection.person_depth` assumes.

The array is carried in :class:`CapturedFrame` in RAW Z16 units; the metric
conversion uses :attr:`RealSenseCapture.depth_scale` (meters per unit) and is
done by the depth module, not here.

Each aligned depth value is the optical-axis Z of that pixel IN THE COLOUR
CAMERA frame (``rs.align`` reprojects the depth points through the
depth->colour extrinsics), so the matching intrinsics for 3D deprojection are
the COLOUR stream's. They are read once in :meth:`RealSenseCapture.start` and
exposed as :attr:`RealSenseCapture.camera_intrinsics`.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .person_depth import CameraIntrinsics


@dataclass
class CapturedFrame:
    """One colour frame handed from the capture thread to the consumer."""

    image_bgr: np.ndarray
    # ROS clock nanoseconds sampled immediately after the frame was received.
    capture_time_ns: int
    # Monotonic capture counter, starting at 0.
    sequence: int
    # RealSense hardware/driver timestamp in milliseconds. Recorded for
    # diagnostics only: it lives in the camera's clock domain, NOT in the ROS
    # clock domain, and is deliberately never mixed into the ROS header.
    hardware_timestamp_ms: float
    # (H, W) uint16 Z16 depth ALIGNED TO THE COLOUR FRAME: same size as
    # image_bgr, same pixel coordinates, RAW units (multiply by
    # RealSenseCapture.depth_scale for meters). None when depth is disabled or
    # when this particular frameset carried no depth frame.
    depth_image: Optional[np.ndarray] = None


@dataclass
class CaptureStats:
    captured: int = 0
    dropped: int = 0
    errors: int = 0
    first_frame_time: Optional[float] = None
    last_frame_time: Optional[float] = None

    def capture_fps(self) -> float:
        if (
            self.first_frame_time is None
            or self.last_frame_time is None
            or self.captured < 2
        ):
            return 0.0
        span = self.last_frame_time - self.first_frame_time
        return (self.captured - 1) / span if span > 0 else 0.0


class RealSenseCaptureError(RuntimeError):
    """Raised when the camera cannot be opened or configured as requested."""


class RealSenseCapture:
    """Owns an ``rs.pipeline`` and publishes the newest colour frame.

    Args:
        width/height/fps: requested colour profile. Validated against the
            device's advertised profiles before the pipeline is started, so a
            wrong request fails with the supported list instead of a generic
            SDK error.
        color_format: ``"bgr8"`` (default) or ``"rgb8"``. The D456 advertises
            both; ``bgr8`` is requested so the array handed to MMPose is
            already in the BGR order the RTMO config expects (its data
            preprocessor uses mean=[0,0,0], std=[1,1,1] and no bgr_to_rgb), and
            no per-frame colour conversion is needed.
        enable_depth: also open the Z16 depth stream (same
            width/height/fps as colour) and align every frameset to the colour
            frame, so ``CapturedFrame.depth_image`` can be indexed with colour
            pixel coordinates. Costs one alignment per captured frame.
        serial: optional device serial number, for multi-camera setups.
        clock_ns: callable returning ROS-clock nanoseconds; injected by the
            node so capture timestamps come from the same clock as the
            published headers.
    """

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 60,
        color_format: str = "bgr8",
        enable_depth: bool = True,
        serial: str = "",
        clock_ns=None,
        wait_timeout_ms: int = 5000,
        logger=None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.color_format = str(color_format).lower()
        self.enable_depth = bool(enable_depth)
        self.serial = str(serial)
        self.wait_timeout_ms = int(wait_timeout_ms)
        self._clock_ns = clock_ns or (lambda: time.time_ns())
        self._logger = logger

        if self.color_format not in ("bgr8", "rgb8"):
            raise RealSenseCaptureError(
                f"color_format must be 'bgr8' or 'rgb8', got '{self.color_format}'"
            )

        self._pipeline = None
        self._profile = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._slot: Optional[CapturedFrame] = None
        self._frame_available = threading.Event()
        self._sequence = 0
        self.stats = CaptureStats()
        self.device_info: Dict[str, str] = {}
        # Raw rs.intrinsics of the ACTIVE colour stream (kept for its
        # distortion model/coeffs), and the same numbers as a pyrealsense2-free
        # CameraIntrinsics for person_depth. Both None until start().
        self.color_intrinsics = None
        self.camera_intrinsics: Optional[CameraIntrinsics] = None
        # Meters per raw Z16 unit, read from the device once the pipeline is
        # up (~0.001 on a D4xx). 0.0 while depth is disabled/not started.
        self.depth_scale = 0.0
        self._align = None
        self._started = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            print(f"[{level}] {message}", flush=True)
            return
        try:
            getattr(self._logger, level)(message)
        except Exception:  # noqa: BLE001
            # The rclpy context may already be torn down (Ctrl+C path).
            print(f"[{level}] {message}", flush=True)

    @staticmethod
    def _import_sdk():
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RealSenseCaptureError(
                "pyrealsense2 is not installed in this environment. It provides "
                "the direct (non-ROS) RealSense capture path required by "
                "input_mode='realsense'. Install the version matching the "
                "system librealsense2, e.g. "
                "`pip install pyrealsense2==2.58.3.10794`."
            ) from exc
        return rs

    def describe_device(self) -> Dict[str, str]:
        """Enumerate the device and record name/serial/firmware/usb."""
        rs = self._import_sdk()
        context = rs.context()
        devices = list(context.query_devices())
        if not devices:
            raise RealSenseCaptureError(
                "No RealSense device found. Check that the camera is connected "
                "and that the container can see it (the /dev bind mount)."
            )

        device = None
        if self.serial:
            for candidate in devices:
                if candidate.get_info(rs.camera_info.serial_number) == self.serial:
                    device = candidate
                    break
            if device is None:
                found = [d.get_info(rs.camera_info.serial_number) for d in devices]
                raise RealSenseCaptureError(
                    f"RealSense serial '{self.serial}' not found; connected: {found}"
                )
        else:
            device = devices[0]

        def info(field, default="unknown"):
            return device.get_info(field) if device.supports(field) else default

        self.device_info = {
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "product_id": info(rs.camera_info.product_id),
            "usb_type": info(rs.camera_info.usb_type_descriptor),
            "sdk_version": rs.__version__,
        }
        self._device = device
        return self.device_info

    def _supported_profiles(self, stream) -> List[Tuple[int, int, int, str]]:
        """(width, height, fps, format) tuples advertised for one stream type."""
        profiles = []
        for sensor in self._device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != stream:
                    continue
                video = profile.as_video_stream_profile()
                profiles.append(
                    (
                        video.width(),
                        video.height(),
                        video.fps(),
                        str(profile.format()).replace("format.", ""),
                    )
                )
        return sorted(set(profiles))

    def supported_color_profiles(self) -> List[Tuple[int, int, int, str]]:
        """(width, height, fps, format) tuples advertised by the colour sensor."""
        rs = self._import_sdk()
        return self._supported_profiles(rs.stream.color)

    def supported_depth_profiles(self) -> List[Tuple[int, int, int, str]]:
        """(width, height, fps, format) tuples advertised by the depth sensor."""
        rs = self._import_sdk()
        return self._supported_profiles(rs.stream.depth)

    def _validate_requested_profile(self) -> None:
        """Fail with the supported list rather than guessing a fallback."""
        self._require_profile(
            self.supported_color_profiles(),
            (self.width, self.height, self.fps, self.color_format),
            "colour",
        )
        if self.enable_depth:
            # Depth is requested at the colour resolution; rs.align resamples
            # it onto the colour grid anyway, and matching sizes keeps the
            # alignment cheap.
            self._require_profile(
                self.supported_depth_profiles(),
                (self.width, self.height, self.fps, "z16"),
                "depth",
            )

    def _require_profile(self, available, requested, label: str) -> None:
        if requested in available:
            return
        width, height, fps, fmt = requested
        same_size = sorted(
            {(w, h, f, p) for (w, h, f, p) in available if w == width and h == height}
        )
        raise RealSenseCaptureError(
            f"Requested {label} profile {width}x{height}@{fps} {fmt} is not "
            f"supported by this device/SDK.\n"
            f"  Profiles at this resolution: {same_size or 'none'}\n"
            f"  Change realsense_width/height/fps/color_format to a supported "
            f"combination"
            + (
                " (or set realsense_enable_depth:=false to run without depth)"
                if label == "depth"
                else ""
            )
            + "; nothing is silently substituted."
        )

    def _build_config(self):
        """Single place where streams are declared (colour, and optionally depth)."""
        rs = self._import_sdk()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        color_format = rs.format.bgr8 if self.color_format == "bgr8" else rs.format.rgb8
        config.enable_stream(
            rs.stream.color, self.width, self.height, color_format, self.fps
        )
        if self.enable_depth:
            # Same geometry as colour; rs.align(rs.stream.color) in
            # _capture_loop is what actually puts it on the colour pixel grid.
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
        return config

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        rs = self._import_sdk()
        self.describe_device()
        self._validate_requested_profile()

        self._pipeline = rs.pipeline()
        config = self._build_config()
        try:
            self._profile = self._pipeline.start(config)
        except RuntimeError as exc:
            self._pipeline = None
            raise RealSenseCaptureError(
                f"Failed to start the RealSense pipeline: {exc}"
            ) from exc

        video_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        # COLOUR intrinsics, read once: RTMO pixels are colour pixels and the
        # depth is aligned to colour, so these (not the depth sensor's) are the
        # ones that deproject (u, v, Z) correctly.
        self.color_intrinsics = video_profile.get_intrinsics()
        self.camera_intrinsics = CameraIntrinsics(
            fx=float(self.color_intrinsics.fx),
            fy=float(self.color_intrinsics.fy),
            cx=float(self.color_intrinsics.ppx),
            cy=float(self.color_intrinsics.ppy),
            width=int(self.color_intrinsics.width),
            height=int(self.color_intrinsics.height),
        )
        self._started = True

        if self.enable_depth:
            # THE depth-to-colour alignment object. Created once; applied to
            # every frameset in _capture_loop.
            self._align = rs.align(rs.stream.color)
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            if self.depth_scale <= 0.0:
                raise RealSenseCaptureError(
                    f"Device reported a non-positive depth scale "
                    f"({self.depth_scale}); refusing to publish depths that "
                    "cannot be converted to meters."
                )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="realsense_capture", daemon=True
        )
        self._thread.start()

        self._log(
            "info",
            f"RealSense started: {self.device_info['name']} "
            f"(serial={self.device_info['serial']}, "
            f"fw={self.device_info['firmware']}, "
            f"usb={self.device_info['usb_type']}, "
            f"sdk={self.device_info['sdk_version']}) "
            f"colour {video_profile.width()}x{video_profile.height()}@"
            f"{video_profile.fps()} {self.color_format}"
            + (
                f" + depth z16 aligned to colour "
                f"(depth_scale={self.depth_scale:.6f} m/unit)"
                if self.enable_depth
                else " (depth disabled)"
            ),
        )

    def stop(self) -> None:
        """Stop the thread and release the camera. Safe to call twice."""
        self._stop_event.set()
        self._frame_available.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                self._log("warn", "RealSense capture thread did not exit in 5s")
        self._thread = None

        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
                self._log("info", "RealSense pipeline stopped; device released")
            except RuntimeError as exc:
                self._log("warn", f"Error stopping RealSense pipeline: {exc}")
        self._pipeline = None
        self._align = None
        self._started = False

    # ------------------------------------------------------------------
    # capture thread
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(self.wait_timeout_ms)
            except RuntimeError as exc:
                # Timeout or transient USB error: keep going unless stopping.
                if self._stop_event.is_set():
                    break
                self.stats.errors += 1
                self._log("warn", f"RealSense wait_for_frames failed: {exc}")
                continue

            # DEPTH-TO-COLOUR ALIGNMENT -- the one and only place it happens.
            # After this the depth frame has the colour frame's size and
            # intrinsics, so RTMO's colour-space keypoints index it directly.
            if self._align is not None:
                frames = self._align.process(frames)

            color = frames.get_color_frame()
            if not color:
                continue

            # Copy out of the SDK buffer: the underlying memory is recycled as
            # soon as the frame object is released, and copying lets the SDK
            # reuse its pool immediately (~1.2 MB, sub-millisecond).
            image = np.asanyarray(color.get_data()).copy()

            depth_image = None
            if self._align is not None:
                depth = frames.get_depth_frame()
                # A frameset can legitimately arrive without depth; the frame
                # is still delivered and the consumer reports depth as NaN.
                if depth:
                    depth_image = np.asanyarray(depth.get_data()).copy()

            now_ns = self._clock_ns()

            captured = CapturedFrame(
                image_bgr=image,
                capture_time_ns=now_ns,
                sequence=self._sequence,
                hardware_timestamp_ms=color.get_timestamp(),
                depth_image=depth_image,
            )
            self._sequence += 1

            now = time.perf_counter()
            if self.stats.first_frame_time is None:
                self.stats.first_frame_time = now
            self.stats.last_frame_time = now
            self.stats.captured += 1

            with self._lock:
                if self._slot is not None:
                    # The consumer never took the previous frame: latest wins.
                    self.stats.dropped += 1
                self._slot = captured
            self._frame_available.set()

    # ------------------------------------------------------------------
    # consumer side
    # ------------------------------------------------------------------
    def get_latest(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Take the newest frame, or None if none arrived within ``timeout``."""
        if not self._frame_available.wait(timeout):
            return None
        with self._lock:
            frame = self._slot
            self._slot = None
            self._frame_available.clear()
        return frame

    @property
    def running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()
