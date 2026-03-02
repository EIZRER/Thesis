import threading
from PyQt5.QtCore import QThread, pyqtSignal
from pipeline.pipeline import PhotogrammetryPipeline


class WorkerThread(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)   # output_dir
    error = pyqtSignal(str)
    aborted = pyqtSignal()

    def __init__(self, image_dir: str, workspace_dir: str, config_path: str):
        super().__init__()
        self.image_dir = image_dir
        self.workspace_dir = workspace_dir
        self.config_path = config_path
        self.abort_event = threading.Event()

    def abort(self):
        """Signal the running subprocess to stop immediately."""
        self.abort_event.set()

    def run(self):
        try:
            pipeline = PhotogrammetryPipeline(self.config_path)
            output_dir = pipeline.run(
                image_dir=self.image_dir,
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
