"""RTMO-M model loading, inference and result parsing (no ROS code).

Boxes stay xyxy in original image pixels until the ROS message is built.
"""

import contextlib
import io
import os.path as osp
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..utils.coco_keypoints import COCO_KEYPOINT_NAMES, NUM_COCO_KEYPOINTS
from .depth_estimation import NO_POSITION, CameraPoint

UNTRACKED_PERSON_ID = -1


@dataclass
class PersonDetection:
    """One detected person, before any ROS conversion.

    Attributes:
        bbox_xyxy: ``(4,)`` float32 ``[x1, y1, x2, y2]`` in original image
            pixels. Not clipped to the image.
        score: person/instance confidence in [0, 1].
        keypoints_xy: ``(17, 2)`` float32 COCO keypoints in image pixels.
        keypoint_scores: ``(17,)`` float32 per-joint confidence.
        track_id: persistent BoT-SORT id, ``None`` when untracked.
        detection_index: index in the frame's list; BoxMOT's ``det_ind``.
            Never published.
        position: ``(x, y, z)`` [m] in the colour optical frame; all-NaN when
            unavailable.
    """

    bbox_xyxy: np.ndarray
    score: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    detection_index: int
    track_id: Optional[int] = None
    position: CameraPoint = NO_POSITION

    @property
    def person_id(self) -> int:
        """The persistent track id, or ``-1`` when there is none."""
        return self.track_id if self.track_id is not None else UNTRACKED_PERSON_ID

    @property
    def depth(self) -> float:
        """Euclidean distance [m], not Z-depth (that is ``position.z``)."""
        return self.position.distance


def install_checkpoint_loader_workaround() -> None:
    """Load the RTMO checkpoint under torch >= 2.6 (``weights_only=False``).

    mmengine 0.10.7 doesn't pass ``weights_only`` and the checkpoint holds
    numpy objects the restricted unpickler can't rebuild.
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
        self.device_name = device
        self.person_score_threshold = float(person_score_threshold)
        self._logger = logger
        self.load_seconds = 0.0

        self._validate_paths()
        self._check_device()
        self._build()

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
            self.device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            self._log("debug", f"torch {torch.__version__}, device={self.device}")
        else:
            self._log("warning", f"Running RTMO on '{self.device}' (no CUDA)")

    def _build(self) -> None:
        install_checkpoint_loader_workaround()

        from mmpose.apis import inference_bottomup, init_model

        self._inference_bottomup = inference_bottomup

        start = time.time()
        # mmengine print()s "Loads checkpoint by ..." with no logger to silence.
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = init_model(self.model_config, self.checkpoint, device=self.device)
        self.load_seconds = time.time() - start
        self._log(
            "info",
            f"RTMO-M loaded in {self.load_seconds:.2f}s on {self.device_name}",
        )
        self._log("debug", f"config={self.model_config}, checkpoint={self.checkpoint}")
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
            "debug", f"Keypoint layout verified: COCO-17 ({NUM_COCO_KEYPOINTS} joints)"
        )

    def infer(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        """Run RTMO on one BGR frame and return parsed detections."""
        results = self._inference_bottomup(self.model, frame_bgr)
        return self._parse(results)

    def _parse(self, results) -> List[PersonDetection]:
        """Convert RTMO ``PoseDataSample`` predictions to PersonDetection."""
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
                    # after filtering, so it matches BoxMOT's det_ind
                    detection_index=len(detections),
                )
            )
        return detections
