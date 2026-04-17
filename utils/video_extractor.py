from pathlib import Path
from typing import Callable, Optional

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 1.0,
    max_frames: int = 0,
    log_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Extract frames from a video at a fixed time interval.

    Args:
        video_path:   Path to the input video file.
        output_dir:   Directory where extracted JPEG frames will be saved.
        interval_sec: Time gap between consecutive extracted frames (seconds).
        max_frames:   Hard cap on the number of frames extracted (0 = no limit).
        log_callback: Optional function called with progress messages.

    Returns:
        Number of frames actually written to disk.

    Raises:
        RuntimeError: If the video file cannot be opened.
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    frame_step = max(1, round(fps * interval_sec))
    expected = total_frames // frame_step
    if max_frames > 0:
        expected = min(expected, max_frames)

    log(f"Video info: {Path(video_path).name}")
    log(f"  Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    log(f"  FPS        : {fps:.2f}")
    log(f"  Duration   : {duration_sec:.1f}s  ({total_frames} frames)")
    log(f"  Interval   : {interval_sec}s  → every {frame_step} frame(s)")
    log(f"  Expected   : ~{expected} frame(s) to extract")

    count = 0
    frame_idx = 0
    log_every = max(1, expected // 10)  # log at ~10% intervals

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        out_path = out / f"frame_{count:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

        if count % log_every == 0 or count == expected:
            t = frame_idx / fps
            log(f"  [{count}/{expected}] t={t:.1f}s → {out_path.name}")

        if max_frames > 0 and count >= max_frames:
            break

        frame_idx += frame_step

    cap.release()
    log(f"Extraction complete: {count} frame(s) saved to {output_dir}")
    return count
