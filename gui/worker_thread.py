import threading
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from pipeline.pipeline import PhotogrammetryPipeline
from utils.video_extractor import extract_frames


class WorkerThread(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)   # output_dir
    error = pyqtSignal(str)
    aborted = pyqtSignal()

    def __init__(
        self,
        image_dir: str,
        workspace_dir: str,
        config_path: str,
        video_path: str = None,
        frame_interval: float = 1.0,
        max_frames: int = 0,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.workspace_dir = workspace_dir
        self.config_path = config_path
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.abort_event = threading.Event()

    def abort(self):
        """Signal the running subprocess to stop immediately."""
        self.abort_event.set()

    def run(self):
        try:
            image_dir = self.image_dir

            # ── Video mode: extract frames first ─────────────────────────────
            if self.video_path:
                if self.abort_event.is_set():
                    self.aborted.emit()
                    return
                frames_dir = str(Path(self.workspace_dir) / "frames")
                self.log.emit("── Extracting frames from video...")
                count = extract_frames(
                    video_path=self.video_path,
                    output_dir=frames_dir,
                    interval_sec=self.frame_interval,
                    max_frames=self.max_frames,
                    log_callback=self.log.emit,
                )
                if count == 0:
                    self.error.emit("No frames were extracted from the video.")
                    return
                image_dir = frames_dir

            pipeline = PhotogrammetryPipeline(self.config_path)
            output_dir = pipeline.run(
                image_dir=image_dir,
                workspace_dir=self.workspace_dir,
                progress_callback=self.progress.emit,
                log_callback=self.log.emit,
                abort_event=self.abort_event,
            )
            self.finished.emit(output_dir)
        except RuntimeError as e:
            if "ABORTED" in str(e):
                self.aborted.emit()
            else:
                self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))
