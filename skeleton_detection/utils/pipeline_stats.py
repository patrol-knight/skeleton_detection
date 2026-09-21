"""Throughput and latency accounting for the perception pipeline.

Keeps the three per-frame costs separate on purpose::

    rtmo_inference_ms   RTMO-M forward + result parsing
    tracking_ms         BoT-SORT association (+ ReID embedding when enabled)
    total_processing_ms inference + tracking + message build + publish

Merging tracking into inference would hide exactly the number this milestone
exists to measure, so they are never summed into one field.

Frame-drop accounting lives in
:class:`~skeleton_detection.input.realsense_capture.CaptureStats`: a frame is *dropped* when a newer camera frame replaced it in
the latest-frame-wins slot before the inference loop consumed it.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _Series:
    """A growing list of millisecond samples with percentile helpers."""

    samples: List[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        self.samples.append(value_ms)

    def summary(self, skip: int = 0) -> Dict[str, float]:
        values = self.samples[skip:]
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)],
            "min": ordered[0],
            "max": ordered[-1],
            "count": float(len(values)),
        }


class PipelineStats:
    """Counters for one run of the pipeline.

    ``skeleton_fps`` is measured from the FIRST processed frame so model
    loading and camera start-up do not depress the reported throughput.
    """

    def __init__(self) -> None:
        self.processed = 0
        self.inference = _Series()
        self.tracking = _Series()
        self.total = _Series()
        self.persons_seen = 0
        self.active_tracks_total = 0

        self.first_process_time: Optional[float] = None
        self.last_process_time: Optional[float] = None
        self.first_inference_ms: Optional[float] = None

        # Rolling window for the periodic log line.
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
        total_ms: float,
        person_count: int,
        active_tracks: int = 0,
    ) -> None:
        now = time.perf_counter()
        if self.first_process_time is None:
            self.first_process_time = now
        self.last_process_time = now

        self.processed += 1
        self.inference.add(inference_ms)
        self.total.add(total_ms)
        if tracking_ms > 0.0:
            self.tracking.add(tracking_ms)
        self.persons_seen += person_count
        self.active_tracks_total += active_tracks

        if self.first_inference_ms is None:
            self.first_inference_ms = inference_ms

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

    def summary(self, warmup_frames: int = 1) -> Dict[str, object]:
        """Aggregate result; the first frame is excluded from the steady-state
        numbers because it pays the one-off CUDA warm-up."""
        return {
            "processed": self.processed,
            "elapsed_sec": self.processing_window_sec(),
            "skeleton_fps": self.skeleton_fps(),
            "first_inference_ms": self.first_inference_ms or 0.0,
            "inference": self.inference.summary(warmup_frames),
            "tracking": self.tracking.summary(warmup_frames),
            "total": self.total.summary(warmup_frames),
            "persons_seen": self.persons_seen,
            "mean_persons": (
                self.persons_seen / self.processed if self.processed else 0.0
            ),
            "mean_active_tracks": (
                self.active_tracks_total / self.processed if self.processed else 0.0
            ),
        }
