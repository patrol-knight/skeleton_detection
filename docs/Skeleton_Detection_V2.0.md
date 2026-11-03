# Skeleton Detection V2.0 — previous pipeline (summary)

V2.0 is the pipeline before per-joint 3D output. Its message is still
published in V2.1 as the **legacy debug output**; see
[Skeleton Detection V2.1](Skeleton_Detection_V2.1.md) for the current
contract.

## Pipeline

```mermaid
flowchart LR
    IN["RGB-D frame<br/>colour + aligned depth + CameraInfo"]
    RTMO["RTMO-M<br/>17 COCO joints (u, v, conf)"]
    TRK["BoT-SORT + OSNet ReID<br/>person_id"]
    PZ["person Z<br/>median of 3x3 joint medians<br/>(joints with conf ≥ 0.3)"]
    POS["position<br/>bbox centre deprojected at person Z"]
    MSG["SkeletonFrame<br/>/skeleton_detection/frame"]
    VIS["2D overlay<br/>/skeleton_detection/visualization_image"]

    IN --> RTMO --> TRK --> PZ --> POS --> MSG
    MSG --> VIS
```

- One process, one pass per frame: RTMO → tracking → person depth → message
  → optional overlay.
- 3D information is **per person only**: one point at the bbox centre.
  Joints stay 2D image pixels.

## Legacy interface

`/skeleton_detection/frame` in V2.0 (now `/skeleton_detection/debug_frame`):
`patrolknight_msgs/msg/SkeletonFrame`.

| Field | Type | Meaning |
|---|---|---|
| `header` | `std_msgs/Header` | capture stamp + optical `frame_id`, copied from the input |
| `frame_index` | `int32` | processed-frame counter |
| `persons[].person_id` | `int32` | track ID `>= 0`, or `-1` when untracked (still published) |
| `persons[].score` | `float32` | person confidence |
| `persons[].bbox` | `float32[4]` | `[x, y, w, h]` image pixels, unclipped |
| `persons[].joints` | `float32[]` | 51 floats `[u0, v0, c0, ...]`, image pixels |
| `persons[].connections` | `int32[]` | 19 COCO edges, 38 ints |
| `persons[].position` | `geometry_msgs/Point` | bbox centre in meters, optical frame (x right, y down, z forward); NaN if unavailable |
| `persons[].depth` | `float32` | Euclidean `sqrt(x²+y²+z²)` of `position`; NaN if unavailable |

## Limits that motivated V2.1

- No joint-level 3D: consumers had to deproject UVs themselves, without the
  depth image.
- Untracked (`-1`) detections mixed into the robot-facing output.
- Person position is a bbox-centre point, not a body point.
