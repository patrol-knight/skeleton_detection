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

Depth readiness
---------------
No depth stream is enabled in this milestone, but the structure does not block
it: :meth:`RealSenseCapture._build_config` is the single place where streams
are declared, the pipeline profile is retained, and colour intrinsics are
exposed.  Adding an aligned depth stream later means enabling it there, adding
an ``rs.align`` step in the capture loop, and carrying the depth array in
:class:`CapturedFrame` -- not rewriting the architecture.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


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
        serial: str = "",
        clock_ns=None,
        wait_timeout_ms: int = 5000,
        logger=None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.color_format = str(color_format).lower()
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
        self.color_intrinsics = None
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

    def supported_color_profiles(self) -> List[Tuple[int, int, int, str]]:
        """(width, height, fps, format) tuples advertised by the colour sensor."""
        rs = self._import_sdk()
        profiles = []
        for sensor in self._device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
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

    def _validate_requested_profile(self) -> None:
        """Fail with the supported list rather than guessing a fallback."""
        available = self.supported_color_profiles()
        requested = (self.width, self.height, self.fps, self.color_format)
        if requested in available:
            return

        same_size = sorted(
            {
                (w, h, f, fmt)
                for (w, h, f, fmt) in available
                if w == self.width and h == self.height
            }
        )
        raise RealSenseCaptureError(
            f"Requested colour profile {self.width}x{self.height}@{self.fps} "
            f"{self.color_format} is not supported by this device/SDK.\n"
            f"  Profiles at this resolution: {same_size or 'none'}\n"
            f"  Change realsense_width/height/fps/color_format to a supported "
            f"combination; nothing is silently substituted."
        )

    def _build_config(self):
        """Single place where streams are declared (depth would be added here)."""
        rs = self._import_sdk()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        color_format = rs.format.bgr8 if self.color_format == "bgr8" else rs.format.rgb8
        config.enable_stream(
            rs.stream.color, self.width, self.height, color_format, self.fps
        )
        # Future depth work goes here, e.g.:
        #   config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        # plus an rs.align(rs.stream.color) applied in _capture_loop.
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
        self.color_intrinsics = video_profile.get_intrinsics()
        self._started = True

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
            f"{video_profile.fps()} {self.color_format}",
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

            color = frames.get_color_frame()
            if not color:
                continue

            # Copy out of the SDK buffer: the underlying memory is recycled as
            # soon as the frame object is released, and copying lets the SDK
            # reuse its pool immediately (~1.2 MB, sub-millisecond).
            image = np.asanyarray(color.get_data()).copy()
            now_ns = self._clock_ns()

            captured = CapturedFrame(
                image_bgr=image,
                capture_time_ns=now_ns,
                sequence=self._sequence,
                hardware_timestamp_ms=color.get_timestamp(),
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
