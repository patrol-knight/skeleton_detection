import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2


class FrameArtifactsWriter:
    def __init__(
        self,
        output_video_path: str,
        output_metadata_path: str,
        frame_size: Tuple[int, int],
        fps: float,
        write_video: bool = True,
    ) -> None:
        self.output_video_path = output_video_path
        self.output_metadata_path = output_metadata_path
        self.frame_size = frame_size
        self.fps = fps
        self.write_video = write_video
        self._writer: Optional[cv2.VideoWriter] = None
        self._metadata_handle = None
        self._metadata_frame_count = 0

    def _ensure_parent_dirs(self) -> None:
        for path in (self.output_video_path, self.output_metadata_path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def _ensure_metadata_file(self) -> None:
        if self._metadata_handle is not None:
            return

        self._ensure_parent_dirs()
        self._metadata_handle = open(self.output_metadata_path, "w+", encoding="utf-8")
        payload = {
            "fps": self.fps,
            "frame_size": [self.frame_size[0], self.frame_size[1]],
            "frames": [],
        }
        json.dump(payload, self._metadata_handle, indent=4)
        self._metadata_handle.flush()

    def _ensure_writer(self) -> None:
        if not self.write_video or self._writer is not None:
            return

        self._ensure_parent_dirs()
        self._writer = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.frame_size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_video_path}")

    def append(
        self,
        frame_index: int,
        timestamp_sec: float,
        annotated_frame_bgr,
        persons: Sequence[Dict[str, Any]],
        timings: Dict[str, float],
    ) -> None:
        self._ensure_parent_dirs()
        if self.write_video:
            self._ensure_writer()
            self._writer.write(annotated_frame_bgr)

        self._ensure_metadata_file()
        frame_payload = {
            "frame_index": frame_index,
            "timestamp": timestamp_sec,
            "persons": list(persons),
            "timings": timings,
        }
        frame_json = json.dumps(frame_payload, indent=8)
        assert self._metadata_handle is not None
        self._metadata_handle.seek(0, os.SEEK_END)

        # Replace the closing empty/non-empty frames suffix with the next entry.
        suffix = "\n    ]\n}"
        self._metadata_handle.seek(self._metadata_handle.tell() - len(suffix))
        if self._metadata_frame_count == 0:
            self._metadata_handle.write("\n")
        else:
            self._metadata_handle.write(",\n")
        self._metadata_handle.write(frame_json)
        self._metadata_handle.write(suffix)
        self._metadata_handle.truncate()
        self._metadata_handle.flush()
        self._metadata_frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self._metadata_handle is not None:
            self._metadata_handle.close()
            self._metadata_handle = None
