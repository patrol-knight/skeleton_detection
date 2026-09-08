"""RTMO-M model ownership and inference.

Owns the model, the checkpoint-loading workaround and the result parsing.
Knows nothing about ROS: it takes a BGR frame and returns a list of
:class:`PersonDetection`.

Coordinate convention
---------------------
Everything here stays in **xyxy**, in the ORIGINAL source image coordinate
system.  MMPose's bottom-up estimator already maps its predictions out of the
padded 640x640 model input back into image space, and BoT-SORT also wants
xyxy, so no conversion happens until
:mod:`skeleton_detection.message_builder` turns the detection into a ROS
message (where the published convention is ``[x, y, width, height]``).
"""

import os.path as osp
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .coco_keypoints import COCO_KEYPOINT_NAMES, NUM_COCO_KEYPOINTS


@dataclass
class PersonDetection:
    """One detected person, before any ROS conversion.

    Attributes:
        bbox_xyxy: ``(4,)`` float32 ``[x1, y1, x2, y2]`` in original image
            pixels. Not clipped to the image.
        score: person/instance confidence in [0, 1].
        keypoints_xy: ``(17, 2)`` float32 COCO keypoints in image pixels.
        keypoint_scores: ``(17,)`` float32 per-joint confidence.
        track_id: persistent BoT-SORT id once
            :class:`~skeleton_detection.tracking.SkeletonTracker` has run.
            ``None`` means "not tracked" -- either tracking is disabled, or the
            tracker did not return a track for this detection this frame.
        detection_index: position of this detection in the frame's list. This
            is what BoxMOT echoes back as ``det_ind``, and it is how a track is
            joined to the exact skeleton that produced it.
    """

    bbox_xyxy: np.ndarray
    score: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    detection_index: int
    track_id: Optional[int] = None

    @property
    def person_id(self) -> int:
        """Id to publish: the track id when tracked, else the frame-local index."""
        return self.track_id if self.track_id is not None else self.detection_index


def install_checkpoint_loader_workaround() -> None:
    """Allow mmengine to load the RTMO checkpoint under torch >= 2.6.

    torch 2.6 flipped the ``torch.load`` ``weights_only`` default to ``True``.
    mmengine 0.10.7's local checkpoint loader never passes the argument and the
    RTMO checkpoint stores numpy objects the restricted unpickler cannot
    rebuild.  Override the registered scheme through mmengine's public API
    instead of patching site-packages (see docker/PACKAGES.md, finding 4).
    """
    import torch
    from mmengine.runner import CheckpointLoader

    @CheckpointLoader.register_scheme(prefixes="", force=True)
    def _load_from_local_trusted(filename, map_location):
        filename = osp.expanduser(filename)
        if not osp.isfile(filename):
            raise FileNotFoundError(f"{filename} can not be found.")
        return torch.load(filename, map_location=map_location, weights_only=False)


class RTMOInferenceError(RuntimeError):
    """Raised when the model cannot be built as requested."""


class RTMOInference:
    """Loads RTMO-M once and runs bottom-up inference on BGR frames."""

    def __init__(
        self,
        model_config: str,
        checkpoint: str,
        device: str = "cuda:0",
        person_score_threshold: float = 0.3,
        logger=None,
    ) -> None:
        self.model_config = model_config
        self.checkpoint = checkpoint
        self.device = device
        self.person_score_threshold = float(person_score_threshold)
        self._logger = logger
        self.load_seconds = 0.0

        self._validate_paths()
        self._check_device()
        self._build()

    # ------------------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            print(f"[{level}] {message}", flush=True)
            return
        try:
            getattr(self._logger, level)(message)
        except Exception:  # noqa: BLE001 - context may already be torn down
            print(f"[{level}] {message}", flush=True)

    def _validate_paths(self) -> None:
        if not osp.isfile(self.model_config):
            raise RTMOInferenceError(f"RTMO config not found: {self.model_config}")
        if not osp.isfile(self.checkpoint):
            raise RTMOInferenceError(f"RTMO checkpoint not found: {self.checkpoint}")

    def _check_device(self) -> None:
        import torch

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RTMOInferenceError(
                    "CUDA is not available inside this container, but device="
                    f"'{self.device}' was requested. Start the container with "
                    "GPU access or set the 'device' parameter to 'cpu' explicitly."
                )
            index = torch.cuda.current_device()
            self._log(
                "info",
                f"CUDA available: using {torch.cuda.get_device_name(index)} "
                f"(device='{self.device}', torch {torch.__version__})",
            )
        else:
            self._log("warning", f"Running RTMO on '{self.device}' (no CUDA)")

    def _build(self) -> None:
        install_checkpoint_loader_workaround()

        # Imported here (not at module import time) so the checkpoint-loader
        # override is registered first and import failures are reported with
        # full node context.
        from mmpose.apis import inference_bottomup, init_model

        self._inference_bottomup = inference_bottomup

        start = time.time()
        self.model = init_model(self.model_config, self.checkpoint, device=self.device)
        self.load_seconds = time.time() - start
        self._log(
            "info",
            f"Loaded RTMO-M in {self.load_seconds:.2f}s "
            f"(config={self.model_config}, checkpoint={self.checkpoint})",
        )
        self._verify_keypoint_layout()

    def _verify_keypoint_layout(self) -> None:
        """Fail loudly if the checkpoint is not COCO-17 in the expected order."""
        dataset_meta = getattr(self.model, "dataset_meta", None) or {}
        id2name = dataset_meta.get("keypoint_id2name")
        if not id2name:
            self._log(
                "warning",
                "Model dataset_meta has no keypoint_id2name; assuming COCO-17 order",
            )
            return

        names = [id2name[index] for index in sorted(id2name)]
        if names != COCO_KEYPOINT_NAMES:
            raise RTMOInferenceError(
                "Model keypoint layout does not match the COCO-17 order this "
                f"package publishes.\n  model: {names}\n  expected: "
                f"{COCO_KEYPOINT_NAMES}"
            )
        self._log(
            "info", f"Keypoint layout verified: COCO-17 ({NUM_COCO_KEYPOINTS} joints)"
        )

    # ------------------------------------------------------------------
    def infer(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        """Run RTMO on one BGR frame and return parsed detections.

        The official MMPose bottom-up API owns resize/pad/normalisation; the
        BGR array is handed over untouched (the RTMO config uses
        ``mean=[0,0,0]``, ``std=[1,1,1]`` and no ``bgr_to_rgb``).

        A ``NotImplementedError`` from the stubbed ``mmcv._ext`` propagates
        deliberately: this environment ships mmcv-lite, and a real native-op
        call must fail loudly rather than yield a fabricated result.
        """
        results = self._inference_bottomup(self.model, frame_bgr)
        return self._parse(results)

    def _parse(self, results) -> List[PersonDetection]:
        """Convert RTMO ``PoseDataSample`` predictions to PersonDetection.

        ``pred_instances`` (already in original-image coordinates) provides
        ``bboxes`` (N,4 xyxy), ``scores`` (N,), ``keypoints`` (N,17,2) and
        ``keypoint_scores`` (N,17).
        """
        detections: List[PersonDetection] = []
        if not results:
            return detections

        pred = results[0].pred_instances
        keypoints = np.asarray(pred.keypoints, dtype=np.float32)
        keypoint_scores = np.asarray(pred.keypoint_scores, dtype=np.float32)
        scores = np.asarray(pred.scores, dtype=np.float32)
        bboxes = np.asarray(pred.bboxes, dtype=np.float32)

        for index in range(keypoints.shape[0]):
            score = float(scores[index])
            if score < self.person_score_threshold:
                continue

            person_keypoints = keypoints[index]
            if person_keypoints.shape[0] != NUM_COCO_KEYPOINTS:
                self._log(
                    "error",
                    f"Expected {NUM_COCO_KEYPOINTS} keypoints, got "
                    f"{person_keypoints.shape[0]}; skipping detection {index}",
                )
                continue

            detections.append(
                PersonDetection(
                    bbox_xyxy=bboxes[index].astype(np.float32),
                    score=score,
                    keypoints_xy=person_keypoints,
                    keypoint_scores=keypoint_scores[index],
                    # Assigned after filtering so the index always matches this
                    # list's position -- which is what BoxMOT's det_ind refers to.
                    detection_index=len(detections),
                )
            )
        return detections
