#!/usr/bin/env python3
"""Benchmark the effective ROS 2 topic FPS of the Skeleton Detection pipeline.

Standalone, read-only utility. It subscribes to two topics of an ALREADY
RUNNING stack and measures how fast messages actually arrive at this node:

    /camera/camera/rgbd          realsense2_camera_msgs/msg/RGBD  (camera input)
    /skeleton_detection/frame    patrolknight_msgs/msg/SkeletonFrame (output)

It never starts, stops, reconfigures or publishes to anything; it only adds one
extra subscriber per topic. RealSense and ZED both publish the same RGBD topic
(the ZED via zed_adapter), so ``--camera`` is a label for titles, terminal
output and filenames only -- it never changes which topic is subscribed to.

Measurement
-----------
Every message's arrival time at this node is recorded (``time.monotonic()``).
Every ``--sample-interval`` seconds (0.1 s), the rolling FPS of each topic is

    messages received in (t - window, t]  /  window

which captures real throughput and short-term variation without the noise of
``1 / latest_inter-frame_interval``.

Warm-up
-------
Before ``t = 0`` the script waits for the first message on BOTH topics and then
keeps receiving for one full ``--window`` (pre-roll). The rolling windows are
therefore already populated at ``t = 0``, so no sample -- and no summary
statistic -- is dragged down by a half-empty window or by DDS discovery time.
Message counts and duration cover only ``t >= 0``.

Messages are taken serialized (``raw=True``): the multi-megabyte RGBD payload
is never deserialized in Python, so the benchmark itself cannot become the
bottleneck that lowers the measured camera FPS.

Usage, in an environment with ROS 2 Humble, realsense2_camera_msgs,
patrolknight_msgs and matplotlib::

    python3 scripts/benchmark_pipeline_fps.py --camera zed --duration 30
    python3 scripts/benchmark_pipeline_fps.py --camera realsense --duration 0   # until Ctrl+C

The host has none of these, but the iot-skeleton image does. Run it from the
repository root in a throwaway container on the IoT stack's network (same
network and ROS_DOMAIN_ID as the skeleton service; no GPU or devices)::

    docker run --rm -it --network iot_ros-net \
        -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} -e HOME=/tmp \
        --user "$(id -u):$(id -g)" -v /etc/localtime:/etc/localtime:ro \
        -v "$PWD":/ws -w /ws iot-skeleton:latest \
        python3 scripts/benchmark_pipeline_fps.py --camera zed --duration 30

Writes ``benchmark_results/<camera>_fps_<YYYYmmdd_HHMMSS>.{csv,png}``.
"""

import argparse
import csv
import signal
import statistics
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

#: Defaults discovered from this repository:
#: ``DEFAULT_RGBD_TOPIC`` in skeleton_detection/input/ros_camera_subscriber.py
#: and ``output_topic`` in skeleton_detection/iot_node.py / config/*.yaml.
DEFAULT_INPUT_TOPIC = "/camera/camera/rgbd"
DEFAULT_OUTPUT_TOPIC = "/skeleton_detection/frame"
EXPECTED_INPUT_TYPE = "realsense2_camera_msgs/msg/RGBD"
EXPECTED_OUTPUT_TYPE = "patrolknight_msgs/msg/SkeletonFrame"

CAMERA_LABELS = {"zed": "ZED", "realsense": "RealSense"}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark_results"

#: Plot colours: slots 1 and 2 of a CVD-validated categorical palette.
INPUT_COLOR = "#2a78d6"
OUTPUT_COLOR = "#eb6834"
INK_PRIMARY = "#1f1f1e"
INK_SECONDARY = "#5f5e5a"
GRID_COLOR = "#e4e3df"
SURFACE_COLOR = "#fcfcfb"


class BenchmarkError(RuntimeError):
    """A precondition (environment, topic, first message) was not met."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark camera RGBD input and SkeletonFrame output FPS "
        "of a running Skeleton Detection pipeline (read-only)."
    )
    parser.add_argument(
        "--camera",
        required=True,
        choices=sorted(CAMERA_LABELS),
        help="Camera backend label for titles/filenames. Does NOT change topics.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Benchmark length in seconds; 0 runs until Ctrl+C (default: 30).",
    )
    parser.add_argument(
        "--input-topic",
        default=DEFAULT_INPUT_TOPIC,
        help=f"Camera RGBD topic (default: {DEFAULT_INPUT_TOPIC}).",
    )
    parser.add_argument(
        "--output-topic",
        default=DEFAULT_OUTPUT_TOPIC,
        help=f"SkeletonFrame topic (default: {DEFAULT_OUTPUT_TOPIC}).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="Rolling FPS window in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Seconds between FPS samples / CSV rows (default: 0.1).",
    )
    parser.add_argument(
        "--topic-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for each topic to appear and deliver its first "
        "message (default: 15).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for CSV/PNG results (default: {DEFAULT_RESULTS_DIR}).",
    )
    args = parser.parse_args(argv)
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    if args.window <= 0:
        parser.error("--window must be > 0")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be > 0")
    if args.topic_timeout <= 0:
        parser.error("--topic-timeout must be > 0")
    return args


def import_pyplot():
    """Import matplotlib with the headless Agg backend, or fail up front.

    Checked before any measurement so a missing package is reported
    immediately instead of after a 30 s run.
    """
    try:
        import matplotlib
    except ImportError as exc:
        raise BenchmarkError(
            "matplotlib is not available in this Python environment. Run the "
            "script inside the iot-skeleton image (it ships matplotlib); see "
            "the module docstring. Nothing was installed or changed."
        ) from exc
    matplotlib.use("Agg")  # no display needed; works over SSH
    import matplotlib.pyplot as plt

    return plt


def import_rclpy():
    try:
        import rclpy
        from rclpy.signals import SignalHandlerOptions
    except ImportError as exc:
        raise BenchmarkError(
            "rclpy is not importable. Source ROS 2 Humble and the message "
            "workspaces first, or run inside the iot-skeleton image (see the "
            "module docstring)."
        ) from exc
    return rclpy, SignalHandlerOptions


class TopicRate:
    """Arrival timestamps of one topic and its rolling-window FPS."""

    def __init__(self, label: str, topic: str, window: float) -> None:
        self.label = label
        self.topic = topic
        self.window = window
        self.arrivals: Deque[float] = deque()
        self.first_arrival: Optional[float] = None
        self.count_since_start = 0
        self.start_time: Optional[float] = None

    def on_message(self, _msg: bytes) -> None:
        now = time.monotonic()
        self.arrivals.append(now)
        if self.first_arrival is None:
            self.first_arrival = now
        if self.start_time is not None and now >= self.start_time:
            self.count_since_start += 1

    def fps(self, now: float) -> float:
        cutoff = now - self.window
        while self.arrivals and self.arrivals[0] <= cutoff:
            self.arrivals.popleft()
        return len(self.arrivals) / self.window


class PipelineFpsBenchmark:
    def __init__(self, args: argparse.Namespace, rclpy, signal_options) -> None:
        self.args = args
        self.rclpy = rclpy
        self.stop_requested = False
        self.interrupted = False

        # rclpy's own SIGINT handling would shut the context down mid-loop;
        # take over SIGINT/SIGTERM so Ctrl+C just ends the measurement and the
        # collected data is still written.
        rclpy.init(signal_handler_options=signal_options.NO)
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)

        self.node = rclpy.create_node("pipeline_fps_benchmark")
        self.input = TopicRate("Camera RGBD Input", args.input_topic, args.window)
        self.output = TopicRate(
            "Skeleton Detection Output", args.output_topic, args.window
        )

        self.elapsed: List[float] = []
        self.input_fps: List[float] = []
        self.output_fps: List[float] = []
        self.duration = 0.0

    def _request_stop(self, _signum, _frame) -> None:
        self.stop_requested = True
        self.interrupted = True

    def _spin(self, timeout: float = 0.01) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout)

    # -- preconditions ---------------------------------------------------

    def _wait_for_publisher(self, rate: TopicRate, expected_type: str) -> str:
        """Wait until ``rate.topic`` has a publisher; return its type string.

        Publishers are checked rather than topic names: the skeleton node's
        own RGBD subscription makes the input topic show up in the graph even
        when no camera is publishing.
        """
        deadline = time.monotonic() + self.args.topic_timeout
        while not self.stop_requested:
            if self.node.count_publishers(rate.topic) > 0:
                types = dict(self.node.get_topic_names_and_types()).get(
                    rate.topic, []
                )
                if types:
                    if expected_type not in types:
                        print(
                            f"[warn] {rate.topic} has type {types}, expected "
                            f"{expected_type}; measuring it anyway.",
                            file=sys.stderr,
                        )
                        return types[0]
                    return expected_type
            if time.monotonic() >= deadline:
                raise BenchmarkError(
                    f"No publisher on {rate.topic} ({rate.label}) within "
                    f"{self.args.topic_timeout:g} s. Is the IoT stack / "
                    "Skeleton Detection running, on the same ROS_DOMAIN_ID and "
                    "Docker network? Override the topic with "
                    "--input-topic/--output-topic if it differs."
                )
            self._spin(0.1)
        raise KeyboardInterrupt

    def _subscribe(self, rate: TopicRate, type_name: str) -> None:
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from rosidl_runtime_py.utilities import get_message

        try:
            msg_type = get_message(type_name)
        except (AttributeError, ModuleNotFoundError, ValueError) as exc:
            raise BenchmarkError(
                f"Message type {type_name} of {rate.topic} is not available in "
                "this environment; source the workspace that provides it."
            ) from exc

        # BEST_EFFORT matches both the camera's sensor-data publisher and the
        # RELIABLE SkeletonFrame publisher, and never applies back-pressure to
        # the running pipeline.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(
            msg_type, rate.topic, rate.on_message, qos, raw=True
        )

    def _wait_for_first_messages(self) -> None:
        deadline = time.monotonic() + self.args.topic_timeout
        while not self.stop_requested:
            silent = [
                r for r in (self.input, self.output) if r.first_arrival is None
            ]
            if not silent:
                return
            if time.monotonic() >= deadline:
                names = ", ".join(f"{r.topic} ({r.label})" for r in silent)
                raise BenchmarkError(
                    f"Publisher found but no message received within "
                    f"{self.args.topic_timeout:g} s on: {names}. Check that the "
                    "camera is streaming and the QoS/network allow delivery."
                )
            self._spin(0.05)
        raise KeyboardInterrupt

    def _pre_roll(self) -> None:
        end = time.monotonic() + self.args.window
        while not self.stop_requested and time.monotonic() < end:
            self._spin(0.01)
        if self.stop_requested:
            raise KeyboardInterrupt

    # -- measurement -----------------------------------------------------

    def run(self) -> None:
        args = self.args
        print(f"Waiting for publishers (timeout {args.topic_timeout:g} s)...")
        in_type = self._wait_for_publisher(self.input, EXPECTED_INPUT_TYPE)
        out_type = self._wait_for_publisher(self.output, EXPECTED_OUTPUT_TYPE)
        print(f"  input : {self.input.topic}  [{in_type}]")
        print(f"  output: {self.output.topic}  [{out_type}]")

        self._subscribe(self.input, in_type)
        self._subscribe(self.output, out_type)
        self._wait_for_first_messages()
        self._pre_roll()

        t0 = time.monotonic()
        self.input.start_time = t0
        self.output.start_time = t0
        # Arrivals before t0 stay in the windows (pre-roll) but not the counts.
        self.input.count_since_start = 0
        self.output.count_since_start = 0

        limit = "until Ctrl+C" if args.duration == 0 else f"for {args.duration:g} s"
        print(f"Measuring {limit} (window {args.window:g} s)...")

        next_sample = t0
        next_report = t0 + 5.0
        while not self.stop_requested:
            now = time.monotonic()
            if args.duration > 0 and now - t0 >= args.duration:
                break
            if now >= next_sample:
                self._sample(now - t0, now)
                next_sample += args.sample_interval
                if next_sample <= now:  # fell behind; skip missed slots
                    next_sample = now + args.sample_interval
            if now >= next_report:
                print(
                    f"  t={now - t0:6.1f}s  camera {self.input_fps[-1]:6.1f} FPS"
                    f"  skeleton {self.output_fps[-1]:6.1f} FPS"
                )
                next_report += 5.0
            self._spin(min(0.01, max(0.0, next_sample - time.monotonic())))

        self.duration = time.monotonic() - t0
        if self.interrupted:
            print("\nInterrupted; saving collected data.")

    def _sample(self, elapsed: float, now: float) -> None:
        self.elapsed.append(elapsed)
        self.input_fps.append(self.input.fps(now))
        self.output_fps.append(self.output.fps(now))

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self.rclpy.ok():
                self.rclpy.shutdown()


# -- outputs -------------------------------------------------------------


def write_csv(path: Path, bench: PipelineFpsBenchmark) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["elapsed_sec", "camera_fps", "skeleton_fps"])
        for t, cam, skel in zip(bench.elapsed, bench.input_fps, bench.output_fps):
            writer.writerow([f"{t:.3f}", f"{cam:.3f}", f"{skel:.3f}"])


def write_plot(plt, path: Path, bench: PipelineFpsBenchmark, camera: str) -> None:
    args = bench.args
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    series = (
        (bench.input, bench.input_fps, INPUT_COLOR),
        (bench.output, bench.output_fps, OUTPUT_COLOR),
    )
    for rate, values, color in series:
        mean = statistics.fmean(values)
        ax.plot(
            bench.elapsed,
            values,
            color=color,
            linewidth=2,
            label=f"{rate.label}  (mean {mean:.1f} FPS)",
        )

    peak = max(max(bench.input_fps), max(bench.output_fps), 1.0)
    ax.set_ylim(0, peak * 1.15)
    ax.set_xlim(0, max(bench.elapsed[-1], args.sample_interval))

    ax.set_title(
        f"{CAMERA_LABELS[camera]} Skeleton Detection Pipeline FPS",
        loc="left",
        fontsize=14,
        color=INK_PRIMARY,
        pad=22,
    )
    ax.text(
        0,
        1.02,
        f"{args.input_topic} → {args.output_topic}  ·  "
        f"{args.window:g} s rolling window  ·  {bench.duration:.1f} s",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("Elapsed time (s)", color=INK_SECONDARY)
    ax.set_ylabel("FPS (messages / s)", color=INK_SECONDARY)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    legend = ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    legend.get_frame().set_edgecolor(GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def print_summary(
    bench: PipelineFpsBenchmark, camera: str, csv_path: Path, png_path: Path
) -> None:
    print()
    print("=" * 64)
    print(f"Camera backend : {CAMERA_LABELS[camera]}")
    print(f"Duration       : {bench.duration:.1f} s"
          + ("  (stopped with Ctrl+C)" if bench.interrupted else ""))
    print(f"Rolling window : {bench.args.window:g} s, "
          f"{len(bench.elapsed)} samples every {bench.args.sample_interval:g} s")
    print("-" * 64)
    print(f"{'':28}{'msgs':>7}{'mean':>7}{'median':>8}{'min':>7}{'max':>7}")
    for rate, values in (
        (bench.input, bench.input_fps),
        (bench.output, bench.output_fps),
    ):
        print(
            f"{rate.label:28}{rate.count_since_start:>7d}"
            f"{statistics.fmean(values):>7.1f}{statistics.median(values):>8.1f}"
            f"{min(values):>7.1f}{max(values):>7.1f}"
        )
    print("-" * 64)
    print(f"PNG : {png_path}")
    print(f"CSV : {csv_path}")
    print("=" * 64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # Progress lines must appear live even without a TTY (docker run, nohup).
    sys.stdout.reconfigure(line_buffering=True)
    try:
        plt = import_pyplot()
        rclpy, signal_options = import_rclpy()
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench = PipelineFpsBenchmark(args, rclpy, signal_options)
    print(f"Pipeline FPS benchmark - camera backend: {CAMERA_LABELS[args.camera]}")
    try:
        bench.run()
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted before measurement started; nothing to save.")
        return 130
    finally:
        bench.shutdown()

    if not bench.elapsed:
        print("No samples were collected; nothing to save.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / f"{args.camera}_fps_{stamp}"
    csv_path = base.with_suffix(".csv")
    png_path = base.with_suffix(".png")
    write_csv(csv_path, bench)
    write_plot(plt, png_path, bench, args.camera)
    print_summary(bench, args.camera, csv_path, png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
