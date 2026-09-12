int32 person_id
float32 score
float32[4] bbox
float32[] joints
int32[] connections
geometry_msgs/Point position
# Estimated distance of this person from the camera, in METERS.
# NaN means "no valid depth was available for this person" (no depth stream,
# or no visible keypoint with a usable depth sample). It is never 0.
float32 depth
