#!/usr/bin/env bash
# Run scripts/benchmark_pipeline_fps.py inside the already-running Skeleton
# Detection container and copy its PNG/CSV results back to the host.
#
#   ./scripts/run_fps_benchmark.sh <zed|realsense> [duration_sec=30] [extra benchmark args...]
#   ./scripts/run_fps_benchmark.sh zed 0        # until Ctrl+C
#
# Read-only towards the pipeline: it only copies the script into the
# container's /tmp, runs it there (the image already has ROS 2, the message
# packages and matplotlib), copies results to <repo>/benchmark_results/ and
# removes its temporary files from the container. Existing results are kept.
#
# SKELETON_CONTAINER overrides the container name (default: iot-skeleton-1).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_SCRIPT="$REPO_ROOT/scripts/benchmark_pipeline_fps.py"
RESULTS_DIR="$REPO_ROOT/benchmark_results"
CONTAINER="${SKELETON_CONTAINER:-iot-skeleton-1}"

usage() {
  echo "Usage: $(basename "$0") <zed|realsense> [duration_sec=30] [extra benchmark args...]" >&2
  echo "       duration 0 runs until Ctrl+C" >&2
}

camera="${1:-}"
duration="${2:-30}"
case "$camera" in
  zed|realsense) ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac
if [[ ! "$duration" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Invalid duration '$duration' (expected seconds >= 0)" >&2
  exit 2
fi
shift $(( $# >= 2 ? 2 : 1 ))

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Benchmark script not found: $BENCH_SCRIPT" >&2
  exit 1
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]]; then
  echo "Container '$CONTAINER' is not running. Start the IoT stack first" >&2
  echo "(e.g. ./launch.sh up camera:=$camera in the iot repository)." >&2
  exit 1
fi

# Unique per run, so concurrent or earlier runs are never mixed up.
run_id="$(date +%Y%m%d_%H%M%S)_$$"
remote_script="/tmp/benchmark_pipeline_fps.py"
remote_out="/tmp/fps_benchmark_$run_id"

cleanup() {
  docker exec "$CONTAINER" rm -rf "$remote_script" "$remote_out" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker cp "$BENCH_SCRIPT" "$CONTAINER:$remote_script" >/dev/null || {
  echo "Failed to copy the benchmark script into '$CONTAINER'" >&2
  exit 1
}

# The container runs in UTC; hand it the host's current UTC offset as a POSIX
# TZ (sign inverted by POSIX) so result filenames carry host local time.
offset="$(date +%z)"
sign="${offset:0:1}"
[[ "$sign" == "+" ]] && posix_sign="-" || posix_sign="+"
host_tz="LOCAL${posix_sign}${offset:1:2}:${offset:3:2}"

# A TTY lets Ctrl+C reach the benchmark as SIGINT, which makes it stop and
# save. The wrapper itself only notes the interrupt and carries on to copy.
tty_flag=()
[[ -t 0 && -t 1 ]] && tty_flag=(-t)
trap 'echo; echo "[wrapper] interrupt received; waiting for the benchmark to save..." >&2' INT

docker exec -i "${tty_flag[@]}" -e TZ="$host_tz" "$CONTAINER" bash -c '
  source /opt/ros/humble/setup.bash
  for ws in /opt/patrolknight_msgs /opt/skeleton_detection; do
    [ -f "$ws/setup.bash" ] && source "$ws/setup.bash"
  done
  exec python3 "$@"
' bash "$remote_script" --camera "$camera" --duration "$duration" \
  --output-dir "$remote_out" "$@"
status=$?
trap - INT

# Copy back whatever was saved, even when the benchmark failed or was stopped.
mapfile -t files < <(docker exec "$CONTAINER" ls -1 "$remote_out" 2>/dev/null)
if (( ${#files[@]} == 0 )); then
  echo "[wrapper] No result files were produced." >&2
  (( status == 0 )) && status=1
  exit "$status"
fi

mkdir -p "$RESULTS_DIR"
echo
echo "Results copied to the host:"
for f in "${files[@]}"; do
  if docker cp "$CONTAINER:$remote_out/$f" "$RESULTS_DIR/$f" >/dev/null; then
    echo "  $RESULTS_DIR/$f"
  else
    echo "  [failed to copy] $f" >&2
    (( status == 0 )) && status=1
  fi
done

exit "$status"
