import threading
from pathlib import Path
from typing import Callable
from utils.command_runner import run_command
from core.exceptions import COLMAPError


class COLMAPRunner:
    def __init__(self, colmap_exe: str, use_gpu: bool = True):
        bat_path = Path(colmap_exe)
        exe_path = bat_path.parent / "bin" / "colmap.exe"
        if bat_path.suffix.lower() == ".bat" and exe_path.exists():
            self.colmap_exe = str(exe_path)
        else:
            self.colmap_exe = str(bat_path)
        self.use_gpu = use_gpu

    def _run(self, args: list, log_callback: Callable = None, abort_event: threading.Event = None):
        cmd = [self.colmap_exe] + args
        try:
            run_command(cmd, log_callback=log_callback, abort_event=abort_event)
        except RuntimeError as e:
            raise COLMAPError(str(e))

    def create_database(self, database_path: str, log_callback=None, abort_event=None):
        self._run(["database_creator", "--database_path", str(database_path)], log_callback, abort_event)

    def extract_features(self, database_path: str, image_path: str, log_callback=None, abort_event=None):
        self._run([
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", "PINHOLE",
        ], log_callback, abort_event)

    def match_features(self, database_path: str, log_callback=None, abort_event=None):
        self._run(["exhaustive_matcher", "--database_path", str(database_path)], log_callback, abort_event)

    def run_mapper(self, database_path: str, image_path: str, output_path: str, log_callback=None, abort_event=None):
        self._run([
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
        ], log_callback, abort_event)

    def convert_model_to_txt(self, input_path: str, output_path: str, log_callback=None, abort_event=None):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self._run([
            "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", "TXT",
        ], log_callback, abort_event)
