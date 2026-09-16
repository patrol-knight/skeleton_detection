int32 person_id
float32 score
float32[4] bbox
float32[] joints
int32[] connections
# This person's 3D position in METERS in the colour camera OPTICAL frame
# (header.frame_id, default camera_color_optical_frame):
#   x = toward image right, y = toward image down, z = forward (optical axis).
# z is the robust RealSense Z-depth; x/y come from deprojecting the bbox centre
# with the colour intrinsics. All three are NaN when unavailable, never 0.
geometry_msgs/Point position
# EUCLIDEAN straight-line distance from the camera optical origin to this
# person, in METERS: sqrt(x^2 + y^2 + z^2) of position. This is NOT the raw
# RealSense depth value (that is position.z). NaN when unavailable, never 0.
float32 depth
