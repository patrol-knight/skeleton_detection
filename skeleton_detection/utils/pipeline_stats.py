"""Throughput and latency accounting for the perception pipeline.

Keeps the per-frame costs separate on purpose::

    rtmo_inference_ms   RTMO-M forward + result parsing
    tracking_ms         BoT-SORT association (+ ReID embedding when enabled)

Merging tracking into inference would hide exactly the number this module
exists to measure, so they are never summed into one field.

Only constant-size state is kept for the whole run (counters and first/last
timestamps). Per-frame timings are kept only for the current periodic-log
window, and only when that log is enabled.

Frame-drop accounting lives in
:class:`~skeleton_detection.input.realsense_capture.CaptureStats`: a frame is *dropped* when a newer camera frame replaced it in
the latest-frame-wins slot before the inference loop consumed it.
"""

import time
from typing import Dict, List, Optional


class PipelineStats:
    """Counters for one run of the pipeline.

    ``skeleton_fps`` is measured from the FIRST processed frame so model
    loading and camera start-up do not depress the reported throughput.

    ``window_enabled=False`` (periodic log disabled) skips all window
    bookkeeping; the run-wide counters used by :meth:`summary` still update.
    """

    def __init__(self, window_enabled: bool = True) -> None:
        self.window_enabled = bool(window_enabled)

        self.processed = 0
        self.persons_seen = 0
        self.active_tracks_total = 0

        self.first_process_time: Optional[float] = None
        self.last_process_time: Optional[float] = None
        self.first_inference_ms: Optional[float] = None

        # Rolling window for the periodic log line; reset by take_window().
        self._window_start = time.perf_counter()
        self._window_processed = 0
        self._window_inference_ms: List[float] = []
        self._window_tracking_ms: List[float] = []

        # Visualization images actually rendered + published (rate limited).
        self.visualizations = 0
        self._window_visualizations = 0
        self.first_visualization_time: Optional[float] = None
        self.last_visualization_time: Optional[float] = None

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def record_frame(
        self,
        inference_ms: float,
        tracking_ms: float,
        person_count: int,
        active_tracks: int = 0,
    ) -> None:
        now = time.perf_counter()
        if self.first_process_time is None:
            self.first_process_time = now
        self.last_process_time = now

        self.processed += 1
        self.persons_seen += person_count
        self.active_tracks_total += active_tracks

        if self.first_inference_ms is None:
            self.first_inference_ms = inference_ms

        if not self.window_enabled:
            return
        self._window_processed += 1
        self._window_inference_ms.append(inference_ms)
        if tracking_ms > 0.0:
            self._window_tracking_ms.append(tracking_ms)

    def record_visualization(self) -> None:
        now = time.perf_counter()
        if self.first_visualization_time is None:
            self.first_visualization_time = now
        self.last_visualization_time = now
        self.visualizations += 1
        if self.window_enabled:
            self._window_visualizations += 1

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def take_window(self) -> Dict[str, float]:
        """Rates/means since the previous call; resets the window."""
        now = time.perf_counter()
        span = now - self._window_start
        window = {
            "count": self._window_processed,
            "skeleton_fps": self._window_processed / span if span > 0 else 0.0,
            "visualization_fps": (
                self._window_visualizations / span if span > 0 else 0.0
            ),
            "inference_ms": (
                sum(self._window_inference_ms) / len(self._window_inference_ms)
                if self._window_inference_ms
                else 0.0
            ),
            "tracking_ms": (
                sum(self._window_tracking_ms) / len(self._window_tracking_ms)
                if self._window_tracking_ms
                else 0.0
            ),
        }
        self._window_start = now
        self._window_processed = 0
        self._window_visualizations = 0
        self._window_inference_ms = []
        self._window_tracking_ms = []
        return window

    def skeleton_fps(self) -> float:
        if (
            self.first_process_time is None
            or self.last_process_time is None
            or self.processed < 2
        ):
            return 0.0
        span = self.last_process_time - self.first_process_time
        return (self.processed - 1) / span if span > 0 else 0.0

    def visualization_fps(self) -> float:
        if (
            self.first_visualization_time is None
            or self.last_visualization_time is None
            or self.visualizations < 2
        ):
            return 0.0
        span = self.last_visualization_time - self.first_visualization_time
        return (self.visualizations - 1) / span if span > 0 else 0.0

    def processing_window_sec(self) -> float:
        if self.first_process_time is None or self.last_process_time is None:
            return 0.0
        return self.last_process_time - self.first_process_time

    def summary(self) -> Dict[str, object]:
        """Run-wide aggregate, built from constant-size counters only."""
        return {
            "processed": self.processed,
            "elapsed_sec": self.processing_window_sec(),
            "skeleton_fps": self.skeleton_fps(),
            "first_inference_ms": self.first_inference_ms or 0.0,
            "persons_seen": self.persons_seen,
            "mean_persons": (
                self.persons_seen / self.processed if self.processed else 0.0
            ),
            "mean_active_tracks": (
                self.active_tracks_total / self.processed if self.processed else 0.0
            ),
        }
